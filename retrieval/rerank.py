"""Local reranking and citation-aware blended scoring for retrieval candidates."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_WEIGHT = 0.75
DEFAULT_CITATION_WEIGHT = 0.25
DEFAULT_TEXT_CHAR_LIMIT = 2500
DEFAULT_MMR_LAMBDA = 0.72
FALLBACK_SCORE_KEYS = ("hybrid_score", "dense_score", "sparse_score", "blended_score", "rerank_score")
AGENT_QUERY_TERMS = {"agent", "agents", "autonomous", "task", "tasks", "tool", "tools", "api", "apis", "execute", "execution", "perform", "workflow", "workflows"}
AGENT_EVIDENCE_TERMS = {"autonomous", "agent", "agents", "planning", "planner", "tool", "tools", "api", "apis", "execution", "execute", "action", "actions", "environment", "feedback", "workflow", "workflows", "taskmatrix", "restgpt"}
AGENT_WEAK_EXAMPLE_TERMS = {"role-playing", "role playing", "debate"}



class CrossEncoderLike(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        """Return one relevance score for each query/candidate text pair."""


@lru_cache(maxsize=4)
def load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> CrossEncoderLike:
    """Load the local cross-encoder, cached by model_name.

    Without caching, every call to score_with_cross_encoder() (i.e. once per
    query during evaluation) reconstructs the model from scratch -- the
    "Loading weights" log line repeating dozens of times during a 50-query
    evaluation run is exactly this happening. lru_cache keeps one instance
    alive per model_name for the life of the process, so the (fast, local)
    model load only happens once. Tests that need to avoid loading a real
    model at all should keep passing a fake `model=` directly to
    score_with_cross_encoder rather than calling this function.
    """

    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def normalized_terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def split_into_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace."""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def compress_text_for_query(text: str, query: str, *, budget_chars: int) -> str:
    """Keep the sentences most relevant to the query instead of blindly truncating a prefix.

    Full-text chunks can be long enough that a fixed-length prefix cut ends up
    dropping the one sentence that actually answers the question, especially
    for chunks whose relevant content sits mid-passage rather than at the
    start. This scores each sentence by query-term overlap, greedily keeps the
    highest-scoring sentences within the character budget, and reassembles
    them in their original order so the result still reads naturally.

    Falls back to plain prefix truncation when there is nothing to compress
    (text already fits), nothing to score against (no query terms), or no
    sentence actually overlaps the query (scoring would be arbitrary).
    """

    if budget_chars <= 0:
        return ""
    if len(text) <= budget_chars:
        return text

    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return text[:budget_chars]

    query_terms = normalized_terms(query)
    if not query_terms:
        return text[:budget_chars]

    scored = [
        (len(normalized_terms(sentence) & query_terms), index)
        for index, sentence in enumerate(sentences)
    ]
    if not any(score > 0 for score, _ in scored):
        return text[:budget_chars]

    selected: set[int] = set()
    used_chars = 0
    for score, index in sorted(scored, key=lambda item: item[0], reverse=True):
        if score <= 0:
            break
        sentence = sentences[index]
        added_chars = len(sentence) + (1 if selected else 0)
        if selected and used_chars + added_chars > budget_chars:
            continue
        selected.add(index)
        used_chars += added_chars
        if used_chars >= budget_chars:
            break

    if not selected:
        return text[:budget_chars]

    compressed = " ".join(sentences[index] for index in sorted(selected))
    return compressed[:budget_chars]


def candidate_to_text(
    candidate: dict[str, Any],
    *,
    max_chars: int = DEFAULT_TEXT_CHAR_LIMIT,
    query: str | None = None,
) -> str:
    """Build the text shown to the reranker from a paper or chunk candidate.

    Without a `query`, this preserves the original behavior: join every
    available field and cut off at `max_chars`. When a `query` is supplied,
    the long free-text `text` field (full-text chunk content) is compressed
    with `compress_text_for_query` instead of being blindly prefix-truncated,
    so budget is spent on the sentences most likely to matter for this
    specific question.
    """

    raw_text = candidate.get("text")
    text_field = raw_text

    if query and query.strip() and raw_text and str(raw_text).strip():
        other_fields = [
            candidate.get("title"),
            candidate.get("topic"),
            candidate.get("section_hint"),
            candidate.get("abstract"),
            candidate.get("main_contribution"),
            candidate.get("methodology"),
            candidate.get("dataset_used"),
            candidate.get("key_result"),
            candidate.get("limitations"),
        ]
        other_chars = sum(len(str(field).strip()) + 1 for field in other_fields if field and str(field).strip())
        remaining_budget = max(max_chars - other_chars, 0)
        text_field = compress_text_for_query(str(raw_text).strip(), query, budget_chars=remaining_budget)

    fields = [
        candidate.get("title"),
        candidate.get("topic"),
        candidate.get("section_hint"),
        text_field,
        candidate.get("abstract"),
        candidate.get("main_contribution"),
        candidate.get("methodology"),
        candidate.get("dataset_used"),
        candidate.get("key_result"),
        candidate.get("limitations"),
    ]
    text = "\n".join(str(field).strip() for field in fields if field and str(field).strip())
    return text[:max_chars]


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


DEFAULT_CROSS_ENCODER_SCORE_MIN = -10.0
DEFAULT_CROSS_ENCODER_SCORE_MAX = 10.0


def normalize_scores_batch_relative(values: Sequence[float]) -> list[float]:
    """Min-max normalize scores within the current candidate batch.

    Kept for direct ablations because this was the original behavior. It is
    sensitive to candidate-pool composition: adding weaker candidates changes
    the batch min/max and therefore rescales every existing candidate.
    """

    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(maximum, minimum):
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def normalize_scores(
    values: Sequence[float],
    *,
    score_min: float = DEFAULT_CROSS_ENCODER_SCORE_MIN,
    score_max: float = DEFAULT_CROSS_ENCODER_SCORE_MAX,
) -> list[float]:
    """Normalize cross-encoder scores against a fixed plausible score range.

    This avoids batch-relative min/max scaling, so a candidate's normalized
    score does not change just because more candidates were added to the same
    reranking batch. Values outside the fixed range are clipped.
    """

    if not values:
        return []
    if score_max <= score_min:
        raise ValueError("score_max must be greater than score_min")

    span = score_max - score_min
    normalized = []
    for value in values:
        clipped = min(score_max, max(score_min, float(value)))
        normalized.append((clipped - score_min) / span)
    return normalized


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
    pairs = [(query, candidate_to_text(candidate, query=query)) for candidate in candidates]
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


def candidate_tokens(candidate: dict[str, Any]) -> set[str]:
    """Build a content fingerprint for diversity comparison, not exact matching."""

    parts = [
        candidate.get("title"),
        candidate.get("topic"),
        candidate.get("text"),
        candidate.get("abstract"),
        candidate.get("main_contribution"),
    ]
    text = " ".join(str(part) for part in parts if part).lower()
    return {token for token in re.findall(r"[a-z][a-z0-9-]{2,}", text) if len(token) > 3}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def select_mmr_candidates(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    lambda_weight: float = DEFAULT_MMR_LAMBDA,
) -> list[dict[str, Any]]:
    """Pick top_k candidates trading off relevance against redundancy.

    Plain top-K-by-score truncation can crowd a result set with several
    near-duplicate high-scoring chunks/papers on the same narrow sub-topic,
    pushing out a genuinely relevant result on a different sub-topic --
    exactly the failure mode multi-document queries like "compare X, Y, and
    Z" hit. Candidates are assumed to already be sorted by blended_score
    descending; the top-scored candidate is always kept, and each subsequent
    pick balances its own score against how redundant it is with what's
    already been selected.
    """

    if top_k <= 0:
        return []
    if len(candidates) <= top_k:
        return list(candidates)

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    token_cache = {id(candidate): candidate_tokens(candidate) for candidate in remaining}

    while remaining and len(selected) < top_k:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        def mmr_score(candidate: dict[str, Any]) -> tuple[float, float]:
            diversity_penalty = max(
                jaccard_similarity(token_cache[id(candidate)], token_cache[id(chosen)]) for chosen in selected
            )
            relevance = float(candidate.get("blended_score") or 0.0)
            score = (lambda_weight * relevance) - ((1.0 - lambda_weight) * diversity_penalty)
            return score, relevance

        best = max(remaining, key=mmr_score)
        remaining.remove(best)
        selected.append(best)

    return selected


def rerank_and_blend(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    model: CrossEncoderLike | None = None,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    top_k: int | None = None,
    rerank_weight: float = DEFAULT_RERANK_WEIGHT,
    citation_weight: float = DEFAULT_CITATION_WEIGHT,
    apply_mmr: bool = False,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
) -> list[dict[str, Any]]:
    """Run cross-encoder reranking and citation-aware blended scoring.

    apply_mmr defaults to False based on a measured evaluation result, not a
    guess: enabling it dropped Recall@5 from 0.65 to 0.62 and Recall@10 from
    0.76 to 0.72 on this project's 36-query labeled eval set (see
    docs/DECISIONS.md). Most queries in that set are answered by multiple
    chunks from the *same* paper, and MMR's redundancy penalty actively
    discards the second and third correct chunks from that paper in favor of
    "more diverse" but less relevant ones. It only clearly helps a minority
    of queries that need several genuinely different papers (e.g.
    "compare X, Y, and Z"). Leave this off unless you have evidence for your
    own query mix that the tradeoff goes the other way.
    """

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

    if top_k is None:
        return boosted
    if apply_mmr:
        return select_mmr_candidates(boosted, top_k=top_k, lambda_weight=mmr_lambda)
    return boosted[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Research question used for reranking.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file containing candidate dictionaries.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--mmr", action="store_true", help="Enable MMR diversity selection instead of plain top-K by score (measured to hurt recall on this project's eval set by default -- see docs/DECISIONS.md).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = json.loads(args.input.read_text(encoding="utf-8"))
    results = rerank_and_blend(args.query, candidates, model_name=args.model, top_k=args.top_k, apply_mmr=args.mmr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()