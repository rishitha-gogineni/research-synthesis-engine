"""Build evidence matrices from unified retrieval results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from retrieval.confidence import load_response
from retrieval.unified_search import run_unified_search
from shared.schemas import EvidenceMatrix, EvidenceMatrixRow, ResearchBrief, UnifiedSearchResponse


DEFAULT_MAX_ROWS = 10
MISSING_VALUE = "not stated in retrieved evidence"


class EvidenceMatrixError(RuntimeError):
    """Raised when an evidence matrix input cannot be loaded or parsed."""


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())


def normalized_value(value: str | None) -> str:
    cleaned = clean_text(value)
    if not cleaned or cleaned.lower() in {"not specified", "not stated in abstract", "none", "n/a"}:
        return MISSING_VALUE
    return cleaned


def score_for_result(result: Any) -> float:
    for attr in ("blended_score", "rerank_score", "hybrid_score", "dense_score", "sparse_score"):
        value = getattr(result, attr, None)
        if value is not None:
            return max(0.0, min(1.0, float(value)))
    return 0.0


def strength_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def source_id_for_result(result: Any, fallback_index: int) -> str:
    if chunk_id := getattr(result, "chunk_id", None):
        return f"chunk:{chunk_id}"
    if paper_id := getattr(result, "paper_id", None):
        return f"paper:{paper_id}"
    return f"result:{fallback_index}"


def snippet_from_result(result: Any, max_chars: int = 260) -> str:
    for attr in ("text", "key_result", "main_contribution", "abstract"):
        value = clean_text(getattr(result, attr, None))
        if value:
            return value[:max_chars]
    return ""


def claim_from_result(result: Any) -> str:
    if contribution := clean_text(getattr(result, "main_contribution", None)):
        return contribution[:180]
    if section := clean_text(getattr(result, "section_hint", None)):
        return f"Evidence from {getattr(result, 'title', 'retrieved source')} ({section})"
    return f"Evidence from {getattr(result, 'title', 'retrieved source')}"


def row_from_result(result: Any, fallback_index: int) -> EvidenceMatrixRow:
    score = score_for_result(result)
    return EvidenceMatrixRow(
        claim=claim_from_result(result),
        supporting_papers=[getattr(result, "title", "Untitled source")],
        source_ids=[source_id_for_result(result, fallback_index)],
        methodology=normalized_value(getattr(result, "methodology", None)),
        dataset=normalized_value(getattr(result, "dataset_used", None)),
        key_result=normalized_value(getattr(result, "key_result", None)),
        limitation=normalized_value(getattr(result, "limitations", None)),
        evidence_strength=strength_from_score(score),
        evidence_snippet=snippet_from_result(result),
    )


def row_sort_key(result: Any) -> float:
    return score_for_result(result)


def build_evidence_matrix(
    response: UnifiedSearchResponse,
    *,
    brief: ResearchBrief | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> EvidenceMatrix:
    del brief  # The current matrix is evidence-first; briefs can consume it later without changing retrieval semantics.
    candidates = list(response.paper_results) + list(response.chunk_results)
    ranked = sorted(candidates, key=row_sort_key, reverse=True)

    rows: list[EvidenceMatrixRow] = []
    seen: set[str] = set()
    for index, result in enumerate(ranked, start=1):
        source_id = source_id_for_result(result, index)
        if source_id in seen:
            continue
        seen.add(source_id)
        rows.append(row_from_result(result, index))
        if len(rows) >= max_rows:
            break

    matrix = EvidenceMatrix(query=response.query, rows=rows)
    matrix.markdown = matrix_to_markdown(matrix)
    return matrix


def escape_cell(value: str) -> str:
    return clean_text(value).replace("|", "\\|")


def matrix_to_markdown(matrix: EvidenceMatrix) -> str:
    if not matrix.rows:
        return "No retrieved evidence rows available."

    headers = ["Claim", "Sources", "Methodology", "Dataset", "Key Result", "Limitation", "Strength"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in matrix.rows:
        cells = [
            row.claim,
            ", ".join(row.source_ids),
            row.methodology,
            row.dataset,
            row.key_result,
            row.limitation,
            row.evidence_strength,
        ]
        lines.append("| " + " | ".join(escape_cell(cell) for cell in cells) + " |")
    return "\n".join(lines)


def load_brief(path: Path) -> ResearchBrief:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ResearchBrief(**payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise EvidenceMatrixError(f"failed to load research brief from {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run unified retrieval for this query before building a matrix.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a saved UnifiedSearchResponse JSON file.")
    parser.add_argument("--brief", type=Path, default=None, help="Optional saved ResearchBrief JSON file.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--markdown", action="store_true", help="Print the evidence matrix as a Markdown table.")
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        response = load_response(args.input)
    elif args.query:
        response = run_unified_search(args.query, top_k=args.top_k, apply_reranking=not args.no_rerank)
    else:
        raise EvidenceMatrixError("provide either a query or --input path")

    brief = load_brief(args.brief) if args.brief else None
    matrix = build_evidence_matrix(response, brief=brief, max_rows=args.top_k)
    if args.markdown:
        print(matrix.markdown)
    else:
        print(matrix.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except EvidenceMatrixError as exc:
        raise SystemExit(f"Error: {exc}") from None
