"""Local reranking and citation-aware blended scoring for retrieval candidates."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_WEIGHT = 0.75
DEFAULT_CITATION_WEIGHT = 0.25
DEFAULT_TEXT_CHAR_LIMIT = 2500
FALLBACK_SCORE_KEYS = ("hybrid_score", "dense_score", "sparse_score", "blended_score", "rerank_score")
AGENT_QUERY_TERMS = {"agent", "agents", "autonomous", "task", "tasks", "tool", "tools", "api", "apis", "execute", "execution", "perform", "workflow", "workflows"}
AGENT_EVIDENCE_TERMS = {"autonomous", "agent", "agents", "planning", "planner", "tool", "tools", "api", "apis", "execution", "execute", "action", "actions", "environment", "feedback", "workflow", "workflows", "taskmatrix", "restgpt"}
AGENT_WEAK_EXAMPLE_TERMS = {"role-playing", "role playing", "debate"}



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


def normalized_terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def is_agent_task_query(query: str) -> bool:
    terms = normalized_terms(query)
    return bool({"agent", "agents", "autonomous"} & terms and AGENT_QUERY_TERMS & terms)


def agent_task_intent_boost(query: str, candidate: dict[str, Any]) -> float:
    if not is_agent_task_query(query):
        return 0.0

    fields = [
        candidate.get("title"),
        candidate.get("section_hint"),
        candidate.get("text"),
        candidate.get("abstract"),
        candidate.get("main_contribution"),
        candidate.get("methodology"),
        candidate.get("key_result"),
    ]
    evidence_text = "\n".join(str(field).strip() for field in fields if field and str(field).strip()).lower()[:4000]
    topic = str(candidate.get("topic") or "").lower()
    bonus = 0.0
    if "ai agents" in topic or "tool use" in topic:
        bonus += 0.03
    if "survey" in evidence_text or "autonomous agent" in evidence_text or "autonomous agents" in evidence_text:
        bonus += 0.04
    if any(term in evidence_text for term in ("tool", "tools", "api", "apis", "taskmatrix", "restgpt")):
        bonus += 0.05
    if any(term in evidence_text for term in ("planning", "planner", "execution", "execute", "actions", "environment", "feedback", "workflow")):
        bonus += 0.04
    if any(term in evidence_text for term in AGENT_WEAK_EXAMPLE_TERMS):
        bonus -= 0.03
    return round(max(0.0, min(0.12, bonus)), 6)


def apply_query_intent_boosts(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boosted = []
    for candidate in candidates:
        bonus = agent_task_intent_boost(query, candidate)
        if bonus <= 0:
            boosted.append(candidate)
            continue
        base_score = float(candidate.get("blended_score") or 0.0)
        score_breakdown = dict(candidate.get("score_breakdown") or {})
        score_breakdown["intent_boost"] = bonus
        boosted.append(
            {
                **candidate,
                "blended_score": round(min(1.0, base_score + bonus), 6),
                "score_breakdown": score_breakdown,
            }
        )
    return sorted(
        boosted,
        key=lambda candidate: (
            candidate.get("blended_score") or 0.0,
            candidate.get("rerank_score") or 0.0,
            candidate.get("citation_count") or 0,
        ),
        reverse=True,
    )


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


def fallback_retrieval_scores(candidates: list[dict[str, Any]]) -> list[float]:
    """Use existing retrieval scores when the optional local reranker is unavailable."""

    scores: list[float] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates):
        available_scores = []
        for key in FALLBACK_SCORE_KEYS:
            value = candidate.get(key)
            if isinstance(value, int | float):
                available_scores.append(float(value))
        scores.append(max(available_scores) if available_scores else float(total - index))
    return scores


def mark_rerank_fallback(candidates: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
    """Annotate fallback-ranked candidates without exposing provider internals."""

    return [
        {
            **candidate,
            "rerank_fallback": reason,
        }
        for candidate in candidates
    ]


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

    fallback_reason = None
    try:
        raw_scores = score_with_cross_encoder(query, candidates, model=model, model_name=model_name)
    except Exception:
        if model is not None:
            raise
        fallback_reason = "cross_encoder_unavailable"
        raw_scores = fallback_retrieval_scores(candidates)

    reranked = attach_rerank_scores(candidates, raw_scores)
    if fallback_reason:
        reranked = mark_rerank_fallback(reranked, fallback_reason)
    blended = apply_citation_blended_scores(
        reranked,
        rerank_weight=rerank_weight,
        citation_weight=citation_weight,
    )
    boosted = apply_query_intent_boosts(query, blended)
    return boosted[:top_k] if top_k is not None else boosted


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
