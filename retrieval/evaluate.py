"""Evaluate route-aware retrieval quality on a small human-readable query set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from agent.query_rewriter import ChatTurn, QueryRewriteResult, rewrite_query
from retrieval.confidence import assess_confidence
from retrieval.unified_search import run_unified_search
from shared.schemas import ConfidenceAssessment, EvaluationQuery, UnifiedSearchResponse


DEFAULT_EVAL_QUERIES = Path("tests/fixtures/eval_queries.json")
DEFAULT_TOP_KS = (5, 10)

SearchRunner = Callable[..., UnifiedSearchResponse]
RewriteRunner = Callable[[str, list[ChatTurn]], QueryRewriteResult]
ConfidenceRunner = Callable[[UnifiedSearchResponse], ConfidenceAssessment]


class EvaluationError(RuntimeError):
    """Raised when the retrieval evaluation cannot run cleanly."""


def load_eval_queries(path: Path) -> list[EvaluationQuery]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [EvaluationQuery(**record) for record in payload]
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise EvaluationError(f"failed to load evaluation queries from {path}: {exc}") from exc


def normalize_text(value: str | None) -> str:
    return (value or "").lower()


def result_id(result: object) -> str | None:
    for attr in ("chunk_id", "paper_id"):
        value = getattr(result, attr, None)
        if value:
            return str(value)
    return None


def select_results(response: UnifiedSearchResponse, route: str) -> list[object]:
    if route == "chunk_level":
        return list(response.chunk_results)
    if route == "hybrid_both":
        return list(response.paper_results) + list(response.chunk_results)
    return list(response.paper_results)


def all_results(response: UnifiedSearchResponse) -> list[object]:
    return list(response.paper_results) + list(response.chunk_results)


def result_text(result: object) -> str:
    fields = [
        getattr(result, "title", None),
        getattr(result, "topic", None),
        getattr(result, "abstract", None),
        getattr(result, "text", None),
        getattr(result, "main_contribution", None),
        getattr(result, "methodology", None),
        getattr(result, "dataset_used", None),
        getattr(result, "key_result", None),
        getattr(result, "limitations", None),
    ]
    return " ".join(normalize_text(str(field)) for field in fields if field)


def topic_hit(results: list[object], expected_topics: list[str], top_k: int) -> bool | None:
    if not expected_topics:
        return None
    expected = {topic.lower() for topic in expected_topics}
    for result in results[:top_k]:
        topic = normalize_text(getattr(result, "topic", None))
        if topic in expected:
            return True
    return False


def keyword_hit(results: list[object], expected_keywords: list[str], top_k: int) -> bool | None:
    if not expected_keywords:
        return None
    haystack = " ".join(result_text(result) for result in results[:top_k])
    return any(keyword.lower() in haystack for keyword in expected_keywords)


def text_contains_keywords(value: str, expected_keywords: list[str]) -> bool | None:
    if not expected_keywords:
        return None
    lowered = normalize_text(value)
    return all(keyword.lower() in lowered for keyword in expected_keywords)


def eval_chat_history(query: EvaluationQuery) -> list[ChatTurn]:
    return [ChatTurn(role=turn.role, content=turn.content) for turn in query.chat_history]


def maybe_rewrite_query(
    query: EvaluationQuery,
    *,
    rewriter: RewriteRunner = rewrite_query,
    enabled: bool = True,
) -> QueryRewriteResult:
    if not enabled or not query.chat_history:
        return QueryRewriteResult(original_query=query.query, standalone_query=query.query, rewrite_used=False, method="none")
    return rewriter(query.query, eval_chat_history(query))


def id_hits(results: list[object], expected_relevant_ids: list[str], top_k: int) -> set[str]:
    expected = set(expected_relevant_ids)
    retrieved = {identifier for result in results[:top_k] if (identifier := result_id(result))}
    return expected & retrieved


def reciprocal_rank(results: list[object], expected_relevant_ids: list[str]) -> float:
    expected = set(expected_relevant_ids)
    for index, result in enumerate(results, start=1):
        identifier = result_id(result)
        if identifier in expected:
            return 1.0 / index
    return 0.0


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def evaluate_response(
    query: EvaluationQuery,
    response: UnifiedSearchResponse,
    top_ks: tuple[int, ...],
    *,
    rewrite_result: QueryRewriteResult | None = None,
    confidence: ConfidenceAssessment | None = None,
) -> dict[str, object]:
    rewrite_result = rewrite_result or QueryRewriteResult(
        original_query=query.query,
        standalone_query=response.query,
        rewrite_used=response.query != query.query,
        method="none",
    )
    route_correct = response.route.route == query.expected_route
    route_results = select_results(response, query.expected_route)
    combined_results = all_results(response)
    has_relevant_ids = bool(query.expected_relevant_ids)
    confidence_decision = confidence.decision if confidence else None

    topic_hits = {k: topic_hit(combined_results, query.expected_topics, k) for k in top_ks}
    keyword_hits = {k: keyword_hit(combined_results, query.expected_keywords, k) for k in top_ks}
    id_hit_sets = {k: id_hits(route_results, query.expected_relevant_ids, k) for k in top_ks}
    rewrite_keyword_hit = text_contains_keywords(rewrite_result.standalone_query, query.expected_standalone_keywords)
    confidence_correct = (
        confidence_decision == query.expected_confidence_decision
        if query.expected_confidence_decision is not None and confidence_decision is not None
        else None
    )

    return {
        "query": query.query,
        "category": query.category,
        "standalone_query": rewrite_result.standalone_query,
        "rewrite_used": rewrite_result.rewrite_used,
        "rewrite_keyword_hit": rewrite_keyword_hit,
        "expected_route": query.expected_route,
        "actual_route": response.route.route,
        "route_correct": route_correct,
        "expected_confidence_decision": query.expected_confidence_decision,
        "actual_confidence_decision": confidence_decision,
        "confidence_correct": confidence_correct,
        "has_relevant_ids": has_relevant_ids,
        "result_ids": [identifier for result in route_results if (identifier := result_id(result))],
        "topic_hits": topic_hits,
        "keyword_hits": keyword_hits,
        "id_hit_sets": id_hit_sets,
        "reciprocal_rank": reciprocal_rank(route_results, query.expected_relevant_ids) if has_relevant_ids else None,
    }


def summarize_evaluations(evaluations: list[dict[str, object]], top_ks: tuple[int, ...]) -> dict[str, object]:
    total = len(evaluations)
    labeled = [evaluation for evaluation in evaluations if evaluation["has_relevant_ids"]]
    labeled_count = len(labeled)
    multi_turn = [evaluation for evaluation in evaluations if evaluation.get("category") == "multi_turn"]
    out_of_corpus = [evaluation for evaluation in evaluations if evaluation.get("category") == "out_of_corpus"]
    rewrite_labeled = [evaluation for evaluation in evaluations if evaluation.get("rewrite_keyword_hit") is not None]
    confidence_labeled = [evaluation for evaluation in evaluations if evaluation.get("confidence_correct") is not None]
    fallback_labeled = [
        evaluation
        for evaluation in confidence_labeled
        if evaluation.get("expected_confidence_decision") != "sufficient_evidence"
    ]

    route_accuracy = safe_rate(sum(1 for evaluation in evaluations if evaluation["route_correct"]), total)
    rewrite_keyword_hit_rate = safe_rate(sum(1 for evaluation in rewrite_labeled if evaluation["rewrite_keyword_hit"]), len(rewrite_labeled))
    confidence_decision_accuracy = safe_rate(sum(1 for evaluation in confidence_labeled if evaluation["confidence_correct"]), len(confidence_labeled))
    crag_fallback_success_rate = safe_rate(sum(1 for evaluation in fallback_labeled if evaluation["confidence_correct"]), len(fallback_labeled))

    topic_counts = {}
    keyword_counts = {}
    recall_counts = {}
    for k in top_ks:
        topic_values = [evaluation["topic_hits"][k] for evaluation in evaluations if evaluation["topic_hits"][k] is not None]
        keyword_values = [evaluation["keyword_hits"][k] for evaluation in evaluations if evaluation["keyword_hits"][k] is not None]
        topic_counts[k] = {
            "value": safe_rate(sum(1 for value in topic_values if value), len(topic_values)),
            "n": len(topic_values),
        }
        keyword_counts[k] = {
            "value": safe_rate(sum(1 for value in keyword_values if value), len(keyword_values)),
            "n": len(keyword_values),
        }
        recall_counts[k] = {
            "value": safe_rate(sum(1 for evaluation in labeled if evaluation["id_hit_sets"][k]), labeled_count),
            "n": labeled_count,
        }

    mrr_values = [float(evaluation["reciprocal_rank"]) for evaluation in labeled]
    mrr = (sum(mrr_values) / labeled_count) if labeled_count else None

    return {
        "queries": total,
        "queries_with_relevant_ids": labeled_count,
        "queries_topic_keyword_only": total - labeled_count,
        "multi_turn_queries": len(multi_turn),
        "out_of_corpus_queries": len(out_of_corpus),
        "route_accuracy": route_accuracy,
        "rewrite_keyword_hit_rate": {"value": rewrite_keyword_hit_rate, "n": len(rewrite_labeled)},
        "confidence_decision_accuracy": {"value": confidence_decision_accuracy, "n": len(confidence_labeled)},
        "crag_fallback_success_rate": {"value": crag_fallback_success_rate, "n": len(fallback_labeled)},
        "topic_hit_rate": topic_counts,
        "keyword_hit_rate": keyword_counts,
        "recall": recall_counts,
        "mrr": {"value": mrr, "n": labeled_count},
    }


def run_evaluation(
    queries: list[EvaluationQuery],
    *,
    search_runner: SearchRunner = run_unified_search,
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS,
    apply_reranking: bool = True,
    apply_query_rewriting: bool = True,
    rewriter: RewriteRunner = rewrite_query,
    confidence_checker: ConfidenceRunner = assess_confidence,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    evaluations = []
    max_top_k = max(top_ks)
    for query in queries:
        rewrite_result = maybe_rewrite_query(query, rewriter=rewriter, enabled=apply_query_rewriting)
        response = search_runner(rewrite_result.standalone_query, top_k=max_top_k, apply_reranking=apply_reranking)
        confidence = confidence_checker(response) if query.expected_confidence_decision is not None else None
        evaluations.append(evaluate_response(query, response, top_ks, rewrite_result=rewrite_result, confidence=confidence))
    return summarize_evaluations(evaluations, top_ks), evaluations


def summary_to_text(summary: dict[str, object], top_ks: tuple[int, ...]) -> str:
    lines = [
        f"queries: {summary['queries']}",
        f"queries_with_relevant_ids: {summary['queries_with_relevant_ids']}",
        f"queries_topic_keyword_only: {summary['queries_topic_keyword_only']}",
        f"multi_turn_queries: {summary['multi_turn_queries']}",
        f"out_of_corpus_queries: {summary['out_of_corpus_queries']}",
        f"route_accuracy: {format_rate(summary['route_accuracy'])}",
        f"rewrite_keyword_hit_rate: {format_rate(summary['rewrite_keyword_hit_rate']['value'])} (contextual subset, n={summary['rewrite_keyword_hit_rate']['n']})",
        f"confidence_decision_accuracy: {format_rate(summary['confidence_decision_accuracy']['value'])} (labeled confidence subset, n={summary['confidence_decision_accuracy']['n']})",
        f"crag_fallback_success_rate: {format_rate(summary['crag_fallback_success_rate']['value'])} (expected fallback subset, n={summary['crag_fallback_success_rate']['n']})",
    ]
    for k in top_ks:
        topic = summary["topic_hit_rate"][k]
        keyword = summary["keyword_hit_rate"][k]
        recall = summary["recall"][k]
        lines.append(f"topic_hit_rate@{k}: {format_rate(topic['value'])} (sanity check, n={topic['n']})")
        lines.append(f"keyword_hit_rate@{k}: {format_rate(keyword['value'])} (sanity check, n={keyword['n']})")
        lines.append(f"recall@{k} (labeled subset, n={recall['n']}): {format_rate(recall['value'])}")
    mrr = summary["mrr"]
    lines.append(f"mrr (labeled subset, n={mrr['n']}): {format_rate(mrr['value'])}")
    return "\n".join(lines)


def parse_top_ks(value: str) -> tuple[int, ...]:
    top_ks = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not top_ks or min(top_ks) <= 0:
        raise argparse.ArgumentTypeError("top-k values must be positive integers")
    return top_ks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_EVAL_QUERIES)
    parser.add_argument("--top-ks", type=parse_top_ks, default=DEFAULT_TOP_KS)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_eval_queries(args.queries)
    summary, evaluations = run_evaluation(
        queries,
        top_ks=args.top_ks,
        apply_reranking=not args.no_rerank,
    )
    if args.json:
        print(json.dumps({"summary": summary, "evaluations": evaluations}, indent=2, default=str))
    else:
        print(summary_to_text(summary, args.top_ks))


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as exc:
        raise SystemExit(f"Error: {exc}") from None
