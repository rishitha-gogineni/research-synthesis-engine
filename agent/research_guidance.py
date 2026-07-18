"""Combined Day 19 research guidance service."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.guidance_common import GuidanceGenerator
from agent.open_problems import build_open_problems_report, open_problems_to_text
from agent.reading_path import build_reading_path, reading_path_to_text
from retrieval.confidence import assess_confidence, load_response
from retrieval.unified_search import run_unified_search
from shared.schemas import ConfidenceAssessment, ResearchGuidanceResponse, UnifiedSearchResponse


DEFAULT_TOP_K = 8


class ResearchGuidanceError(RuntimeError):
    """Raised when combined research guidance cannot be generated."""


def build_research_guidance(
    response: UnifiedSearchResponse,
    *,
    confidence: ConfidenceAssessment | None = None,
    reading_generator: GuidanceGenerator | None = None,
    problems_generator: GuidanceGenerator | None = None,
    max_papers: int = DEFAULT_TOP_K,
    max_problems: int = 6,
) -> ResearchGuidanceResponse:
    confidence = confidence or assess_confidence(response)
    warnings: list[str] = []

    if confidence.decision != "sufficient_evidence":
        warnings.append(confidence.reason)
        warnings.append(confidence.recommended_action)
        return ResearchGuidanceResponse(
            question=response.query,
            confidence=confidence,
            reading_path=None,
            open_problems=None,
            warnings=warnings,
        )

    reading_path = build_reading_path(response, confidence=confidence, generator=reading_generator, max_papers=max_papers)
    open_problems = build_open_problems_report(
        response,
        confidence=confidence,
        generator=problems_generator,
        max_problems=max_problems,
    )
    warnings.extend(reading_path.limitations)
    warnings.extend(open_problems.corpus_limitations)
    warnings.extend(open_problems.evidence_gaps)

    return ResearchGuidanceResponse(
        question=response.query,
        confidence=confidence,
        reading_path=reading_path,
        open_problems=open_problems,
        warnings=list(dict.fromkeys(warnings)),
    )


def guidance_to_text(response: ResearchGuidanceResponse) -> str:
    lines = [f"Research guidance for: {response.question}", f"Confidence decision: {response.confidence.decision}"]
    if response.reading_path:
        lines.append("\n" + reading_path_to_text(response.reading_path))
    if response.open_problems:
        lines.append("\n" + open_problems_to_text(response.open_problems))
    if response.warnings:
        lines.append("\nWarnings:")
        lines.extend(f"- {warning}" for warning in response.warnings)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run unified retrieval for this question before guidance generation.")
    parser.add_argument("--query", dest="query_option", help="Run unified retrieval for this question before guidance generation.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a saved UnifiedSearchResponse JSON file.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-problems", type=int, default=6)
    parser.add_argument("--readable", action="store_true", help="Print readable text instead of JSON.")
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = args.query_option or args.query
    if args.input:
        response = load_response(args.input)
    elif query:
        response = run_unified_search(query, top_k=args.top_k, apply_reranking=not args.no_rerank)
    else:
        raise ResearchGuidanceError("provide either --query, a positional query, or --input path")

    guidance = build_research_guidance(response, max_papers=args.top_k, max_problems=args.max_problems)
    print(guidance_to_text(guidance) if args.readable else guidance.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except ResearchGuidanceError as exc:
        raise SystemExit(f"Error: {exc}") from None
