"""CRAG-style confidence guardrail for unified retrieval results."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from retrieval.unified_search import run_unified_search
from shared.schemas import ConfidenceAssessment, UnifiedSearchResponse


SUFFICIENT_THRESHOLD = 0.72
BROADEN_THRESHOLD = 0.45
CLARIFY_ROUTE_CONFIDENCE_THRESHOLD = 0.6
MIN_STRONG_RESULTS = 2
QUERY_SUPPORT_MINIMUM = 0.18
QUERY_SPECIFICITY_MINIMUM = 1
QUERY_STOPWORDS = {
    "about", "after", "against", "answer", "answers", "are", "can", "compare", "corpus",
    "does", "explain", "from", "give", "highly", "indexed", "into", "main", "paper",
    "papers", "policies", "question", "recent", "research", "results", "should", "show",
    "that", "their", "there", "these", "they", "this", "those", "used", "user", "what",
    "when", "where", "which", "while", "with", "without", "about", "tell", "query", "test",
}


class ConfidenceError(RuntimeError):
    """Raised when confidence assessment input cannot be parsed."""


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def all_results(response: UnifiedSearchResponse) -> list[Any]:
    return list(response.paper_results) + list(response.chunk_results)


def candidate_score(candidate: Any) -> float:
    for attr in ("blended_score", "rerank_score", "hybrid_score", "dense_score"):
        value = getattr(candidate, attr, None)
        if value is not None:
            return clamp(float(value))
    return 0.0


def topic_values(results: list[Any]) -> list[str]:
    return [str(topic).strip().lower() for result in results if (topic := getattr(result, "topic", None))]


def result_ids(results: list[Any], attr: str) -> set[str]:
    return {str(value) for result in results if (value := getattr(result, attr, None))}


def query_terms(query: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", query.lower())
    terms: list[str] = []
    for token in tokens:
        normalized = token.strip("-+")
        if len(normalized) < 4 or normalized in QUERY_STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms


def candidate_text(candidate: Any) -> str:
    fields = [
        getattr(candidate, "title", None),
        getattr(candidate, "topic", None),
        getattr(candidate, "abstract", None),
        getattr(candidate, "text", None),
        getattr(candidate, "main_contribution", None),
        getattr(candidate, "methodology", None),
        getattr(candidate, "dataset_used", None),
        getattr(candidate, "key_result", None),
        getattr(candidate, "limitations", None),
    ]
    return " ".join(str(field).lower() for field in fields if field)


def score_query_support(response: UnifiedSearchResponse, terms: list[str]) -> tuple[float, list[str]]:
    if not terms:
        return 0.0, ["query_support=0.00: no specific query terms available"]
    results = all_results(response)[:10]
    if not results:
        return 0.0, ["query_support=0.00: no results available for query-term check"]
    combined = " ".join(candidate_text(result) for result in results)
    matched = [term for term in terms if term in combined]
    score = len(matched) / len(terms)
    shown = ", ".join(matched[:5]) if matched else "none"
    return clamp(score), [f"query_support={score:.2f}: matched query terms: {shown}"]


def score_result_count(count: int) -> float:
    if count <= 0:
        return 0.0
    return clamp(count / 5)


def score_consistency(scores: list[float]) -> float:
    if not scores:
        return 0.0
    strong = sum(1 for score in scores[:5] if score >= 0.55)
    return clamp(strong / min(len(scores), 5))


def score_topic_agreement(response: UnifiedSearchResponse) -> tuple[float, list[str]]:
    results = all_results(response)
    signals: list[str] = []
    topics = topic_values(results)
    if not topics:
        return 0.0, ["topic_agreement=0.00: no result topics available"]

    most_common_topic, most_common_count = Counter(topics).most_common(1)[0]
    concentration = most_common_count / len(topics)
    signals.append(f"topic_concentration={concentration:.2f}: dominant topic '{most_common_topic}'")

    if response.route.route == "hybrid_both":
        paper_topics = set(topic_values(list(response.paper_results)))
        chunk_topics = set(topic_values(list(response.chunk_results)))
        paper_ids = result_ids(list(response.paper_results), "paper_id")
        chunk_paper_ids = result_ids(list(response.chunk_results), "paper_id")
        has_both_sets = bool(response.paper_results and response.chunk_results)
        has_overlap = bool((paper_topics & chunk_topics) or (paper_ids & chunk_paper_ids))
        if has_both_sets and has_overlap:
            signals.append("hybrid_agreement=1.00: paper and chunk evidence overlap by topic or paper id")
            return 1.0, signals
        if has_both_sets:
            signals.append("hybrid_agreement=0.45: paper and chunk evidence both exist but do not overlap")
            return 0.45, signals
        signals.append("hybrid_agreement=0.20: hybrid route is missing one result set")
        return 0.2, signals

    return clamp(concentration), signals


def score_route_confidence(response: UnifiedSearchResponse) -> float:
    return clamp(float(response.route.confidence))


def confidence_components(response: UnifiedSearchResponse) -> dict[str, Any]:
    results = all_results(response)
    scores = sorted((candidate_score(result) for result in results), reverse=True)
    agreement_score, agreement_signals = score_topic_agreement(response)
    terms = query_terms(response.query)
    query_support_score, query_support_signals = score_query_support(response, terms)
    return {
        "result_count": len(results),
        "top_score": scores[0] if scores else 0.0,
        "route_confidence": score_route_confidence(response),
        "count_score": score_result_count(len(results)),
        "consistency_score": score_consistency(scores),
        "agreement_score": agreement_score,
        "query_terms": terms,
        "query_specificity": len(terms),
        "query_support_score": query_support_score,
        "agreement_signals": agreement_signals,
        "query_support_signals": query_support_signals,
    }


def weighted_confidence(components: dict[str, Any]) -> float:
    score = (
        0.35 * components["top_score"]
        + 0.20 * components["route_confidence"]
        + 0.10 * components["count_score"]
        + 0.12 * components["consistency_score"]
        + 0.13 * components["agreement_score"]
        + 0.10 * components["query_support_score"]
    )
    return round(clamp(score), 6)


def decide(response: UnifiedSearchResponse, components: dict[str, Any], confidence_score: float) -> tuple[str, str, str]:
    result_count = components["result_count"]
    top_score = components["top_score"]
    route_confidence = components["route_confidence"]
    query_specificity = components["query_specificity"]
    query_support_score = components["query_support_score"]

    if result_count == 0:
        return (
            "insufficient_evidence",
            "No retrieval results were returned, so synthesis would be unsupported.",
            "Do not generate an answer; broaden retrieval or ask the user for a narrower question.",
        )
    if route_confidence < CLARIFY_ROUTE_CONFIDENCE_THRESHOLD and confidence_score < SUFFICIENT_THRESHOLD:
        return (
            "ask_clarifying_question",
            "The router was uncertain and retrieved evidence is not strong enough for synthesis.",
            "Ask a clarifying question before generating an answer.",
        )
    if query_specificity < QUERY_SPECIFICITY_MINIMUM:
        return (
            "ask_clarifying_question",
            "The question does not contain enough specific research terms to verify retrieved evidence.",
            "Ask the user to name the paper, method, dataset, or research topic they want analyzed.",
        )
    if query_support_score < QUERY_SUPPORT_MINIMUM:
        return (
            "insufficient_evidence",
            "Retrieved results do not contain the specific terms needed to support the user question.",
            "Do not generate an answer; tell the user the indexed corpus does not provide enough evidence for this question.",
        )
    if result_count < MIN_STRONG_RESULTS or top_score < 0.35 or confidence_score < BROADEN_THRESHOLD:
        return (
            "broaden_search",
            "Retrieved evidence is weak or too sparse for a grounded answer.",
            "Broaden retrieval, increase top-k, or use the hybrid_both route before synthesis.",
        )
    if confidence_score >= SUFFICIENT_THRESHOLD:
        return (
            "sufficient_evidence",
            "Retrieved evidence is strong enough to proceed to grounded synthesis.",
            "Proceed to research brief generation using the retrieved evidence.",
        )
    return (
        "broaden_search",
        "Retrieved evidence is partially relevant but below the synthesis confidence threshold.",
        "Run an expanded retrieval pass before generating an answer.",
    )


def assess_confidence(response: UnifiedSearchResponse) -> ConfidenceAssessment:
    components = confidence_components(response)
    confidence_score = weighted_confidence(components)
    decision, reason, recommended_action = decide(response, components, confidence_score)

    signals = [
        f"top_score={components['top_score']:.2f}",
        f"route_confidence={components['route_confidence']:.2f}",
        f"result_count={components['result_count']}",
        f"count_score={components['count_score']:.2f}",
        f"consistency_score={components['consistency_score']:.2f}",
        f"agreement_score={components['agreement_score']:.2f}",
        f"query_support_score={components['query_support_score']:.2f}",
        f"query_specificity={components['query_specificity']}",
        *components["agreement_signals"],
        *components["query_support_signals"],
    ]

    return ConfidenceAssessment(
        query=response.query,
        route=response.route.route,
        confidence_score=confidence_score,
        decision=decision,
        reason=reason,
        recommended_action=recommended_action,
        signals=signals,
        result_count=components["result_count"],
        top_score=round(components["top_score"], 6),
        route_confidence=round(components["route_confidence"], 6),
    )


def load_response(path: Path) -> UnifiedSearchResponse:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return UnifiedSearchResponse(**payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ConfidenceError(f"failed to load unified response from {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run unified retrieval for this query before assessing confidence.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a saved UnifiedSearchResponse JSON file.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        response = load_response(args.input)
    elif args.query:
        response = run_unified_search(args.query, top_k=args.top_k, apply_reranking=not args.no_rerank)
    else:
        raise ConfidenceError("provide either a query or --input path")
    assessment = assess_confidence(response)
    print(assessment.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except ConfidenceError as exc:
        raise SystemExit(f"Error: {exc}") from None
