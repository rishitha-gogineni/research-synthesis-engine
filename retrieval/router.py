"""Rule-based routing for research questions.

The router chooses which retrieval path should handle a user question:
paper-level retrieval, chunk-level retrieval, both retrieval levels, or a
metadata-oriented filter path.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

from pydantic import ValidationError

from shared.schemas import QueryRoute, QueryRouteName


MIN_CONFIDENT_SCORE = 2

PAPER_LEVEL_SIGNALS = {
    "approach": "broad approach question",
    "approaches": "broad approach question",
    "overview": "overview question",
    "survey": "survey question",
    "trend": "trend question",
    "trends": "trend question",
    "theme": "theme question",
    "themes": "theme question",
    "landscape": "research landscape question",
    "summarize": "summary question",
    "summary": "summary question",
    "taxonomy": "taxonomy question",
    "main": "main ideas question",
}

CHUNK_LEVEL_SIGNALS = {
    "dataset": "dataset detail requested",
    "datasets": "dataset detail requested",
    "benchmark": "benchmark detail requested",
    "benchmarks": "benchmark detail requested",
    "metric": "metric detail requested",
    "metrics": "metric detail requested",
    "methodology": "methodology detail requested",
    "method": "method detail requested",
    "methods": "method detail requested",
    "limitation": "limitation detail requested",
    "limitations": "limitation detail requested",
    "result": "result detail requested",
    "results": "result detail requested",
    "evaluation": "evaluation detail requested",
    "evaluate": "evaluation detail requested",
    "experiment": "experiment detail requested",
    "experiments": "experiment detail requested",
    "ablation": "ablation detail requested",
    "evidence": "evidence detail requested",
}

HYBRID_BOTH_SIGNALS = {
    "compare": "comparison needs broad papers and detailed evidence",
    "comparison": "comparison needs broad papers and detailed evidence",
    "contrast": "comparison needs broad papers and detailed evidence",
    "versus": "comparison needs broad papers and detailed evidence",
    "vs": "comparison needs broad papers and detailed evidence",
    "tradeoff": "tradeoff question needs both retrieval levels",
    "tradeoffs": "tradeoff question needs both retrieval levels",
    "difference": "difference question needs both retrieval levels",
    "differences": "difference question needs both retrieval levels",
    "pros": "pros/cons question needs both retrieval levels",
    "cons": "pros/cons question needs both retrieval levels",
}

METADATA_FILTER_SIGNALS = {
    "top": "ranking/filtering by metadata requested",
    "top-cited": "ranking/filtering by citation count requested",
    "cited": "citation metadata requested",
    "citation": "citation metadata requested",
    "citations": "citation metadata requested",
    "recent": "recency metadata requested",
    "latest": "recency metadata requested",
    "newest": "recency metadata requested",
    "oldest": "year metadata requested",
    "year": "year metadata requested",
    "after": "year filter requested",
    "before": "year filter requested",
    "between": "year range filter requested",
    "list": "list/filter request",
    "show": "list/filter request",
}

PHRASE_SIGNALS: list[tuple[QueryRouteName, str, str, int]] = [
    ("hybrid_both", "compare", "comparison needs broad papers and detailed evidence", 2),
    ("hybrid_both", "versus", "comparison needs broad papers and detailed evidence", 2),
    ("paper_level", "main approaches", "broad approach question", 2),
    ("paper_level", "research themes", "theme question", 2),
    ("paper_level", "state of the art", "overview question", 2),
    ("chunk_level", "key result", "result detail requested", 2),
    ("chunk_level", "key results", "result detail requested", 2),
    ("chunk_level", "used to evaluate", "evaluation detail requested", 2),
    ("metadata_filter", "top cited", "ranking by citation count requested", 2),
    ("metadata_filter", "most cited", "ranking by citation count requested", 2),
    ("metadata_filter", "after 20", "year filter requested", 2),
    ("metadata_filter", "before 20", "year filter requested", 2),
    ("hybrid_both", "compare and contrast", "comparison needs broad papers and detailed evidence", 3),
]

ROUTE_REASONS: dict[QueryRouteName, str] = {
    "paper_level": "The query asks for broad themes, approaches, trends, or an overview, so paper-level retrieval is the right first path.",
    "chunk_level": "The query asks for detailed evidence such as datasets, metrics, methods, results, or limitations, so full-text chunk retrieval is the right first path.",
    "hybrid_both": "The query either needs both broad paper context and detailed evidence, or it is ambiguous enough that using both retrieval levels is safer.",
    "metadata_filter": "The query asks for ranking, listing, citation, recency, or year-based filtering, so metadata filtering is the right first path.",
}


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def tokenize(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", query.lower())


def score_signals(query: str) -> tuple[dict[QueryRouteName, int], dict[QueryRouteName, list[str]]]:
    normalized = normalize_query(query)
    tokens = tokenize(normalized)
    scores: dict[QueryRouteName, int] = defaultdict(int)
    signals: dict[QueryRouteName, list[str]] = defaultdict(list)

    signal_groups: list[tuple[QueryRouteName, dict[str, str]]] = [
        ("paper_level", PAPER_LEVEL_SIGNALS),
        ("chunk_level", CHUNK_LEVEL_SIGNALS),
        ("hybrid_both", HYBRID_BOTH_SIGNALS),
        ("metadata_filter", METADATA_FILTER_SIGNALS),
    ]

    for route, route_signals in signal_groups:
        for token in tokens:
            if token in route_signals:
                scores[route] += 1
                signals[route].append(f"{token}: {route_signals[token]}")

    for route, phrase, reason, weight in PHRASE_SIGNALS:
        if phrase in normalized:
            scores[route] += weight
            signals[route].append(f"{phrase}: {reason}")

    return dict(scores), {route: sorted(set(values)) for route, values in signals.items()}


def choose_route(scores: dict[QueryRouteName, int]) -> tuple[QueryRouteName, bool]:
    if not scores:
        return "hybrid_both", True

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_route, best_score = ranked[0]
    tied_best = [route for route, score in ranked if score == best_score]

    if best_score < MIN_CONFIDENT_SCORE or len(tied_best) > 1:
        return "hybrid_both", True

    return best_route, False


def route_confidence(route: QueryRouteName, score: int, used_fallback: bool) -> float:
    if used_fallback:
        return 0.55
    if route == "hybrid_both":
        return min(0.95, 0.72 + (score * 0.06))
    return min(0.95, 0.68 + (score * 0.08))


def validate_query(query: str) -> str:
    return QueryRoute(
        query=query,
        route="hybrid_both",
        reason="temporary validation route",
        confidence=0.0,
    ).query


def route_query(query: str) -> QueryRoute:
    """Classify a user question into one of the four retrieval routes."""

    stripped_query = validate_query(query)
    scores, signals = score_signals(stripped_query)
    route, used_fallback = choose_route(scores)
    matched_signals = signals.get(route, [])

    if used_fallback:
        matched_signals = sorted({signal for route_signals in signals.values() for signal in route_signals})
        if matched_signals:
            matched_signals.append("fallback: ambiguous or low-confidence route match")
        else:
            matched_signals = ["fallback: no confident route-specific signal matched"]

    score = scores.get(route, 0)
    return QueryRoute(
        query=stripped_query,
        route=route,
        reason=ROUTE_REASONS[route],
        confidence=route_confidence(route, score, used_fallback),
        matched_signals=matched_signals,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Research question to route.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = route_query(args.query)
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        raise SystemExit(f"Error: {exc}") from None
