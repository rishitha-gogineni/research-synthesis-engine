"""Create OpenAI embeddings for enriched papers and truncate them to 1024 dims."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from shared.schemas import EnrichedPaper


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
FULL_DIMENSIONS = 3072
TRUNCATED_DIMENSIONS = 1024


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


def stable_paper_key(paper: EnrichedPaper) -> str:
    return paper.paper_id or f"{paper.title.lower()}::{paper.year}"


def load_enriched_papers(path: Path) -> list[EnrichedPaper]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EnrichedPaper.model_validate(item) for item in payload]


def load_existing_embeddings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_embeddings(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def build_embedding_text(paper: EnrichedPaper) -> str:
    authors = ", ".join(paper.authors[:8]) if paper.authors else "not specified"
    year = str(paper.year) if paper.year is not None else "not specified"
    return "\n".join(
        [
            f"Title: {paper.title}",
            f"Topic: {paper.topic}",
            f"Authors: {authors}",
            f"Year: {year}",
            f"Abstract: {paper.abstract}",
            f"Main contribution: {paper.main_contribution}",
            f"Methodology: {paper.methodology}",
            f"Dataset used: {paper.dataset_used}",
            f"Key result: {paper.key_result}",
            f"Limitations: {paper.limitations}",
        ]
    )


def truncate_embedding(embedding: list[float], dimensions: int = TRUNCATED_DIMENSIONS) -> list[float]:
    if len(embedding) < dimensions:
        raise ValueError(f"Embedding has {len(embedding)} dims, cannot truncate to {dimensions}")
    return embedding[:dimensions]


def embed_texts(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def build_embedding_record(
    paper: EnrichedPaper,
    full_embedding: list[float],
    embedding_text: str,
    model: str,
    dimensions: int,
) -> dict:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "topic": paper.topic,
        "year": paper.year,
        "citation_count": paper.citation_count,
        "embedding_model": model,
        "full_embedding_dimensions": len(full_embedding),
        "embedding_dimensions": dimensions,
        "embedding": truncate_embedding(full_embedding, dimensions),
        "embedding_text": embedding_text,
        "metadata": paper.model_dump(),
    }


def run_embedding(
    input_path: Path,
    output_path: Path,
    model: str,
    dimensions: int,
    batch_size: int,
    limit: Optional[int],
    force: bool,
    delay_seconds: float,
) -> tuple[int, int]:
    papers = load_enriched_papers(input_path)
    existing = [] if force else load_existing_embeddings(output_path)
    existing_ids = {record["paper_id"] for record in existing if record.get("paper_id")}

    candidates = [paper for paper in papers if force or stable_paper_key(paper) not in existing_ids]
    if limit is not None:
        candidates = candidates[:limit]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before running embeddings.")

    client = OpenAI(api_key=api_key)
    embedded_count = 0

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        texts = [build_embedding_text(paper) for paper in batch]
        embeddings = embed_texts(client, model, texts)

        for paper, text, full_embedding in zip(batch, texts, embeddings):
            record = build_embedding_record(
                paper=paper,
                full_embedding=full_embedding,
                embedding_text=text,
                model=model,
                dimensions=dimensions,
            )
            existing.append(record)
            embedded_count += 1
            print(f"[{embedded_count}/{len(candidates)}] embedded: {paper.title[:90]}")

        write_embeddings(output_path, existing)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return embedded_count, len(existing) - embedded_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/enriched_papers_final.json"))
    parser.add_argument("--output", type=Path, default=Path("data/embedded_papers.json"))
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=int, default=TRUNCATED_DIMENSIONS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.dimensions <= 0:
        raise ValueError("--dimensions must be greater than 0")

    embedded_count, skipped_count = run_embedding(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
        delay_seconds=args.delay_seconds,
    )
    print(f"Embedding complete: {embedded_count} embedded, {skipped_count} already present")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None

