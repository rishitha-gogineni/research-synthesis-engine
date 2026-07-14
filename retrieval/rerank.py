"""Local reranking and citation-aware blended scoring for retrieval candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_WEIGHT = 0.75
DEFAULT_CITATION_WEIGHT = 0.25
DEFAULT_TEXT_CHAR_LIMIT = 2500


class CrossEncoderLike(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        """Return one relevance score for each query/candidate text pair."""


def load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> CrossEncoderLike:
    """Load the local cross-encoder lazily so tests do not download a model."""

    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def candidate_to_text(candidate: dict[str, Any], *, max_chars: int = DEFAULT_TEXT_CHAR_LIMIT) -> str:
    """Build the text shown to the reranker from a paper or chunk candidate."""

    fields = [
        candidate.get("title"),
        candidate.get("topic"),
        candidate.get("section_hint"),
        candidate.get("text"),
        candidate.get("abstract"),
        candidate.get("main_contribution"),
        candidate.get("methodology"),
        candidate.get("dataset_used"),
        candidate.get("key_result"),
        candidate.get("limitations"),
    ]
    text = "\n".join(str(field).strip() for field in fields if field and str(field).strip())
    return text[:max_chars]


def normalize_scores(values: Sequence[float]) -> list[float]:
    """Min-max normalize scores to 0..1 while keeping equal scores stable."""

    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(maximum, minimum):
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def normalized_citation_scores(candidates: Sequence[dict[str, Any]]) -> list[float]:
    citation_logs = [math.log1p(max(int(candidate.get("citation_count") or 0), 0)) for candidate in candidates]
    maximum = max(citation_logs, default=0.0)
    if maximum <= 0:
        return [0.0 for _ in citation_logs]
    return [value / maximum for value in citation_logs]


def score_with_cross_encoder(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    model: CrossEncoderLike | None = None,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> list[float]:
    """Score candidates with a local cross-encoder."""

    if not query.strip():
        raise ValueError("query must not be empty")
    if not candidates:
        return []

    reranker = model or load_cross_encoder(model_name)
    pairs = [(query, candidate_to_text(candidate)) for candidate in candidates]
    return [float(score) for score in reranker.predict(pairs)]


def attach_rerank_scores(
    candidates: list[dict[str, Any]],
    raw_scores: Sequence[float],
) -> list[dict[str, Any]]:
    """Attach raw and normalized rerank scores without mutating input candidates."""

    if len(candidates) != len(raw_scores):
        raise ValueError("candidates and raw_scores must have the same length")

    normalized = normalize_scores([float(score) for score in raw_scores])
    enriched = []
    for candidate, raw_score, normalized_score in zip(candidates, raw_scores, normalized):
        enriched.append(
            {
                **candidate,
                "rerank_raw_score": float(raw_score),
                "rerank_score": round(float(normalized_score), 6),
            }
        )
    return sorted(enriched, key=lambda candidate: candidate["rerank_score"], reverse=True)


def apply_citation_blended_scores(
    candidates: list[dict[str, Any]],
    *,
    rerank_weight: float = DEFAULT_RERANK_WEIGHT,
    citation_weight: float = DEFAULT_CITATION_WEIGHT,
) -> list[dict[str, Any]]:
    """Blend rerank relevance with log-normalized citation count."""

    if rerank_weight < 0 or citation_weight < 0:
        raise ValueError("score weights must be non-negative")
    total_weight = rerank_weight + citation_weight
    if total_weight <= 0:
        raise ValueError("at least one score weight must be positive")

    normalized_rerank_weight = rerank_weight / total_weight
    normalized_citation_weight = citation_weight / total_weight
    citation_scores = normalized_citation_scores(candidates)

    enriched = []
    for candidate, citation_score in zip(candidates, citation_scores):
        rerank_score = float(candidate.get("rerank_score") or 0.0)
        blended_score = (normalized_rerank_weight * rerank_score) + (normalized_citation_weight * citation_score)
        enriched.append(
            {
                **candidate,
                "citation_score": round(citation_score, 6),
                "blended_score": round(blended_score, 6),
                "score_breakdown": {
                    "rerank_score": round(rerank_score, 6),
                    "citation_score": round(citation_score, 6),
                    "rerank_weight": round(normalized_rerank_weight, 6),
                    "citation_weight": round(normalized_citation_weight, 6),
                },
            }
        )

    return sorted(
        enriched,
        key=lambda candidate: (
            candidate["blended_score"],
            candidate.get("rerank_score") or 0.0,
            candidate.get("citation_count") or 0,
        ),
        reverse=True,
    )


def rerank_and_blend(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    model: CrossEncoderLike | None = None,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    top_k: int | None = None,
    rerank_weight: float = DEFAULT_RERANK_WEIGHT,
    citation_weight: float = DEFAULT_CITATION_WEIGHT,
) -> list[dict[str, Any]]:
    """Run cross-encoder reranking and citation-aware blended scoring."""

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    raw_scores = score_with_cross_encoder(query, candidates, model=model, model_name=model_name)
    reranked = attach_rerank_scores(candidates, raw_scores)
    blended = apply_citation_blended_scores(
        reranked,
        rerank_weight=rerank_weight,
        citation_weight=citation_weight,
    )
    return blended[:top_k] if top_k is not None else blended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Research question used for reranking.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file containing candidate dictionaries.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_CROSS_ENCODER_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = json.loads(args.input.read_text(encoding="utf-8"))
    results = rerank_and_blend(args.query, candidates, model_name=args.model, top_k=args.top_k)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
