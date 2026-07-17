"""Generate grounded research briefs from unified retrieval results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from pydantic import ValidationError

from retrieval.confidence import assess_confidence, load_response
from retrieval.index_qdrant import load_env_file
from retrieval.unified_search import run_unified_search
from shared.schemas import (
    BriefTheme,
    ConfidenceAssessment,
    EvidenceSource,
    ResearchBrief,
    UnifiedSearchResponse,
)


DEFAULT_SYNTHESIS_MODEL = "gpt-4o-mini"
MAX_SOURCE_TEXT_CHARS = 900
MAX_SOURCES = 12
BriefGenerator = Callable[[str], str]


class SynthesisError(RuntimeError):
    """Raised when a grounded brief cannot be generated or parsed."""


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())


def source_score(result: Any) -> float:
    for attr in ("blended_score", "rerank_score", "hybrid_score", "dense_score", "sparse_score"):
        value = getattr(result, attr, None)
        if value is not None:
            return max(0.0, min(1.0, float(value)))
    return 0.0


def source_text_from_result(result: Any) -> str:
    if text := clean_text(getattr(result, "text", None)):
        return text[:MAX_SOURCE_TEXT_CHARS]

    parts = [
        clean_text(getattr(result, "main_contribution", None)),
        clean_text(getattr(result, "methodology", None)),
        clean_text(getattr(result, "dataset_used", None)),
        clean_text(getattr(result, "key_result", None)),
        clean_text(getattr(result, "limitations", None)),
        clean_text(getattr(result, "abstract", None)),
    ]
    joined = " ".join(part for part in parts if part)
    return joined[:MAX_SOURCE_TEXT_CHARS]


def evidence_source_from_result(result: Any, fallback_index: int) -> EvidenceSource | None:
    evidence_text = source_text_from_result(result)
    if not evidence_text:
        return None

    paper_id = getattr(result, "paper_id", None)
    chunk_id = getattr(result, "chunk_id", None)
    if chunk_id:
        source_id = f"chunk:{chunk_id}"
    elif paper_id:
        source_id = f"paper:{paper_id}"
    else:
        source_id = f"result:{fallback_index}"

    return EvidenceSource(
        source_id=source_id,
        title=getattr(result, "title", "Untitled source"),
        topic=getattr(result, "topic", "Unknown"),
        paper_id=paper_id,
        chunk_id=chunk_id,
        year=getattr(result, "year", None),
        citation_count=getattr(result, "citation_count", 0) or 0,
        evidence_text=evidence_text,
        score=round(source_score(result), 6),
    )


def collect_evidence_sources(response: UnifiedSearchResponse, max_sources: int = MAX_SOURCES) -> list[EvidenceSource]:
    candidates = list(response.paper_results) + list(response.chunk_results)
    ranked = sorted(candidates, key=source_score, reverse=True)

    sources: list[EvidenceSource] = []
    seen: set[str] = set()
    for index, result in enumerate(ranked, start=1):
        source = evidence_source_from_result(result, index)
        if source is None or source.source_id in seen:
            continue
        seen.add(source.source_id)
        sources.append(source)
        if len(sources) >= max_sources:
            break
    return sources


def build_synthesis_prompt(query: str, sources: list[EvidenceSource]) -> str:
    source_blocks = []
    for source in sources:
        source_blocks.append(
            "\n".join(
                [
                    f"SOURCE_ID: {source.source_id}",
                    f"TITLE: {source.title}",
                    f"TOPIC: {source.topic}",
                    f"YEAR: {source.year or 'unknown'}",
                    f"CITATIONS: {source.citation_count}",
                    f"EVIDENCE: {source.evidence_text}",
                ]
            )
        )

    joined_sources = "\n\n".join(source_blocks)
    return f"""You are generating a grounded research brief for a research synthesis engine.

User question:
{query}

Use only the retrieved sources below. Do not add outside facts. If the retrieved evidence does not support a point, say so in limitations or open_problems. Cite source IDs exactly as provided.

Retrieved sources:
{joined_sources}

Return only valid JSON with this exact shape:
{{
  "direct_answer": "2-4 sentence answer grounded in the sources",
  "themes": [
    {{
      "theme": "short theme name",
      "summary": "what the sources collectively say about this theme",
      "supporting_source_ids": ["source:id"]
    }}
  ],
  "evidence_bullets": ["specific evidence point with source IDs"],
  "limitations": ["what the retrieved evidence does not establish"],
  "open_problems": ["research gap or follow-up question grounded in the evidence"]
}}
"""


def parse_brief_payload(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise SynthesisError("brief generator did not return valid JSON") from None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SynthesisError(f"brief generator returned malformed JSON: {exc}") from exc


def build_guarded_brief(
    query: str,
    confidence: ConfidenceAssessment,
    sources: list[EvidenceSource],
) -> ResearchBrief:
    return ResearchBrief(
        query=query,
        status="skipped_low_confidence",
        confidence_decision=confidence.decision,
        direct_answer="",
        themes=[],
        evidence_bullets=[],
        limitations=[confidence.reason],
        open_problems=[confidence.recommended_action],
        sources=sources,
        warning="Research brief generation skipped because retrieval confidence was below the synthesis threshold.",
    )


def build_research_brief(
    response: UnifiedSearchResponse,
    *,
    confidence: ConfidenceAssessment | None = None,
    generator: BriefGenerator | None = None,
    model: str = DEFAULT_SYNTHESIS_MODEL,
) -> ResearchBrief:
    confidence = confidence or assess_confidence(response)
    sources = collect_evidence_sources(response)

    if confidence.decision != "sufficient_evidence":
        return build_guarded_brief(response.query, confidence, sources)

    if not sources:
        raise SynthesisError("cannot generate a research brief without evidence sources")

    prompt = build_synthesis_prompt(response.query, sources)
    raw_text = generator(prompt) if generator else call_openai_generator(prompt, model=model)
    payload = parse_brief_payload(raw_text)

    try:
        themes = [BriefTheme(**item) for item in payload.get("themes", [])]
        return ResearchBrief(
            query=response.query,
            status="generated",
            confidence_decision=confidence.decision,
            direct_answer=payload.get("direct_answer", ""),
            themes=themes,
            evidence_bullets=list(payload.get("evidence_bullets", [])),
            limitations=list(payload.get("limitations", [])),
            open_problems=list(payload.get("open_problems", [])),
            sources=sources,
        )
    except (TypeError, ValidationError) as exc:
        raise SynthesisError(f"brief payload failed schema validation: {exc}") from exc


def call_openai_generator(
    prompt: str,
    *,
    model: str = DEFAULT_SYNTHESIS_MODEL,
    env_file: Path = Path(".env"),
) -> str:
    load_env_file(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SynthesisError("OPENAI_API_KEY is missing. Add it to .env before live brief generation.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON for a grounded research brief."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise SynthesisError("OpenAI returned an empty brief")
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run unified retrieval for this query before generating a brief.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a saved UnifiedSearchResponse JSON file.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_SYNTHESIS_MODEL)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        response = load_response(args.input)
    elif args.query:
        response = run_unified_search(args.query, top_k=args.top_k, apply_reranking=not args.no_rerank)
    else:
        raise SynthesisError("provide either a query or --input path")

    if args.env_file != Path(".env"):
        generator = lambda prompt: call_openai_generator(prompt, model=args.model, env_file=args.env_file)
        brief = build_research_brief(response, generator=generator, model=args.model)
    else:
        brief = build_research_brief(response, model=args.model)
    print(brief.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except SynthesisError as exc:
        raise SystemExit(f"Error: {exc}") from None
