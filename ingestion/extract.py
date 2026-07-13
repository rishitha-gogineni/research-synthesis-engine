"""Extract structured metadata from paper abstracts with a cheap LLM."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from shared.schemas import EnrichedPaper, Paper


DEFAULT_MODEL = "gpt-4o-mini"
NOT_SPECIFIED = "not specified"


class ExtractedFields(BaseModel):
    """Fields produced by the LLM before merging back into paper metadata."""

    main_contribution: str = Field(..., min_length=1)
    methodology: str = Field(..., min_length=1)
    dataset_used: str = Field(..., min_length=1)
    key_result: str = Field(..., min_length=1)
    limitations: str = Field(..., min_length=1)


LlmExtractor = Callable[[Paper], dict[str, Any] | str]


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines from a local .env file without an extra dependency."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_raw_papers(path: Path) -> list[Paper]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Paper.model_validate(item) for item in payload]


def load_existing_enriched(path: Path) -> list[EnrichedPaper]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EnrichedPaper.model_validate(item) for item in payload]


def write_enriched(path: Path, papers: list[EnrichedPaper]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [paper.model_dump() for paper in papers]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_error(path: Path, paper: Paper, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "topic": paper.topic,
        "error": error,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_prompt(paper: Paper) -> list[dict[str, str]]:
    system_prompt = (
        "You extract structured research metadata from academic abstracts. "
        "Use only evidence stated in the title and abstract. Do not guess. "
        f'If the abstract does not specify a field, write "{NOT_SPECIFIED}". '
        "Return strict JSON only."
    )
    user_prompt = f"""
Extract these exact JSON fields from the paper:
- main_contribution
- methodology
- dataset_used
- key_result
- limitations

Rules:
- Use concise phrases or one short sentence per field.
- Do not infer datasets, results, or limitations that are not stated.
- If a dataset is not mentioned, use "{NOT_SPECIFIED}".
- If a key result is not mentioned, use "{NOT_SPECIFIED}".
- If limitations are not mentioned, use "{NOT_SPECIFIED}".
- Return only a JSON object with the five fields.

Title:
{paper.title}

Abstract:
{paper.abstract}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_llm_output(output: dict[str, Any] | str) -> ExtractedFields:
    if isinstance(output, dict):
        payload = output
    else:
        text = output.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        payload = json.loads(text)

    return ExtractedFields.model_validate(payload)


def call_openai_extractor(client: OpenAI, model: str, paper: Paper) -> dict[str, Any] | str:
    response = client.chat.completions.create(
        model=model,
        messages=build_prompt(paper),
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=350,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty response")
    return content


def enrich_paper(paper: Paper, extractor: LlmExtractor, retries: int = 1) -> EnrichedPaper:
    last_error: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            fields = parse_llm_output(extractor(paper))
            return EnrichedPaper(**paper.model_dump(), **fields.model_dump())
        except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
            last_error = exc

    raise ValueError(f"Extraction failed after {retries + 1} attempt(s): {last_error}")


def stable_paper_key(paper: Paper) -> str:
    return paper.paper_id or f"{paper.title.lower()}::{paper.year}"


def run_extraction(
    input_path: Path,
    output_path: Path,
    errors_path: Path,
    model: str,
    limit: Optional[int],
    topics: Optional[list[str]],
    retries: int,
    delay_seconds: float,
    force: bool,
) -> tuple[int, int, int]:
    raw_papers = load_raw_papers(input_path)
    if topics:
        topic_set = set(topics)
        raw_papers = [paper for paper in raw_papers if paper.topic in topic_set]

    enriched = [] if force else load_existing_enriched(output_path)
    existing_keys = {stable_paper_key(paper) for paper in enriched}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before running extraction.")

    client = OpenAI(api_key=api_key)

    candidates = [paper for paper in raw_papers if force or stable_paper_key(paper) not in existing_keys]
    if limit is not None:
        candidates = candidates[:limit]

    success_count = 0
    skipped_count = len(enriched)
    failure_count = 0

    for index, paper in enumerate(candidates, start=1):
        try:
            extractor = lambda item: call_openai_extractor(client, model, item)
            enriched_paper = enrich_paper(paper, extractor=extractor, retries=retries)
            enriched.append(enriched_paper)
            existing_keys.add(stable_paper_key(paper))
            success_count += 1
            write_enriched(output_path, enriched)
            print(f"[{index}/{len(candidates)}] enriched: {paper.title[:90]}")
        except Exception as exc:
            failure_count += 1
            append_error(errors_path, paper, str(exc))
            print(f"[{index}/{len(candidates)}] skipped after failure: {paper.title[:90]}")
            if "insufficient_quota" in str(exc):
                raise RuntimeError(
                    "OpenAI API returned insufficient_quota. Check billing/credits for this API key "
                    "before rerunning extraction."
                ) from exc

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return success_count, skipped_count, failure_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw_papers.json"))
    parser.add_argument("--output", type=Path, default=Path("data/enriched_papers_final.json"))
    parser.add_argument("--errors", type=Path, default=Path("data/extraction_errors.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Process only this many not-yet-enriched papers.")
    parser.add_argument(
        "--topic",
        action="append",
        help="Only process this topic. Repeat for multiple topics. Defaults to all topics.",
    )
    parser.add_argument("--retries", type=int, default=1, help="Retry malformed/invalid LLM outputs this many times.")
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="Ignore existing enriched output and reprocess from scratch.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")

    success_count, skipped_count, failure_count = run_extraction(
        input_path=args.input,
        output_path=args.output,
        errors_path=args.errors,
        model=args.model,
        limit=args.limit,
        topics=args.topic,
        retries=args.retries,
        delay_seconds=args.delay_seconds,
        force=args.force,
    )
    print(
        "Extraction complete: "
        f"{success_count} enriched, {skipped_count} already present, {failure_count} failed"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None
