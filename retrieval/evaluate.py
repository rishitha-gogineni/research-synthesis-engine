"""Evaluate route-aware retrieval quality on a small human-readable query set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from agent.query_rewriter import ChatTurn, QueryRewriteResult, rewrite_query
from retrieval.confidence import assess_confidence
from retrieval.promotion import is_diversity_query
from full_text.index_chunks_qdrant import DEFAULT_COLLECTION as DEFAULT_CHUNK_COLLECTION
from retrieval.index_qdrant import DEFAULT_COLLECTION as DEFAULT_PAPER_COLLECTION
from retrieval.unified_search import (
    DEFAULT_APPLY_PROMOTION,
    DEFAULT_PROMOTION_POOL_MULTIPLIER,
    run_unified_search,
)
from shared.schemas import ConfidenceAssessment, EvaluationQuery, UnifiedSearchResponse


DEFAULT_EVAL_QUERIES = Path("tests/fixtures/eval_queries.json")
DEFAULT_PAPER_ID_ALIASES = Path("data/paper_id_aliases.json")
DEFAULT_TOP_KS = (5, 10)
DEFAULT_MERGE_RRF_K = 60

SearchRunner = Callable[..., UnifiedSearchResponse]
RewriteRunner = Callable[[str, list[ChatTurn]], QueryRewriteResult]
ConfidenceRunner = Callable[[UnifiedSearchResponse], ConfidenceAssessment]


class EvaluationError(RuntimeError):
    """Raised when the retrieval evaluation cannot run cleanly."""


@lru_cache(maxsize=1)
def load_paper_id_aliases(path: Path = DEFAULT_PAPER_ID_ALIASES) -> dict[str, str]:
    """Load duplicate-record aliases used to compare stable paper identities."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"failed to load paper ID aliases from {path}: {exc}") from exc
    aliases = payload.get("aliases", payload)
    if not isinstance(aliases, dict):
        raise EvaluationError(f"paper ID aliases at {path} must be a JSON object")
    return {str(alias): str(canonical) for alias, canonical in aliases.items()}


def canonical_identifier(value: str, aliases: dict[str, str] | None = None) -> str:
    """Resolve a paper ID alias while leaving chunk IDs and unknown IDs intact."""
    mapping = load_paper_id_aliases() if aliases is None else aliases
    return mapping.get(value, value)


def load_eval_queries(path: Path) -> list[EvaluationQuery]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [EvaluationQuery(**record) for record in payload]
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise EvaluationError(f"failed to load evaluation queries from {path}: {exc}") from exc


def normalize_text(value: str | None) -> str:
    return (value or "").lower()


def result_identifiers(result: object) -> list[str]:
    """Return every stable ID that can identify a retrieved result.

Chunk results can be labeled either by their exact chunk ID or by their parent
paper ID, so evaluation should consider both identifiers.
    """
    identifiers: list[str] = []
    for attr in ("chunk_id", "paper_id"):
        value = getattr(result, attr, None)
        if value and str(value) not in identifiers:
            identifiers.append(str(value))
        if attr == "paper_id" and value:
            canonical = canonical_identifier(str(value))
            if canonical not in identifiers:
                identifiers.append(canonical)
    return identifiers


def result_id(result: object) -> str | None:
    identifiers = result_identifiers(result)
    return identifiers[0] if identifiers else None


def merge_ranked_lists(
    papers: list[object],
    chunks: list[object],
    *,
    rrf_k: int = DEFAULT_MERGE_RRF_K,
) -> list[object]:
    """Interleave two independently-ranked lists by reciprocal rank.

    Paper scores (weighted fusion) and chunk scores (dense cosine) are on
    different scales and cannot be compared directly, so the merge uses rank
    position only. Ties -- and rank r in both lists always ties -- break toward
    papers, then toward the better rank, which makes the merge deterministic.
    """

    scored: list[tuple[float, int, int, object]] = []
    for list_order, results in enumerate((papers, chunks)):
        for rank, result in enumerate(results, start=1):
            scored.append((1.0 / (rrf_k + rank), list_order, rank, result))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [result for _, _, _, result in scored]


def select_results(response: UnifiedSearchResponse, route: str, *, merge_hybrid: bool = False, conditional_merge: bool = False) -> list[object]:
    """Return the result list that a route's metrics should be computed over.

    For hybrid_both the default is plain concatenation, which is what every
    published number for this project was measured with. That concatenation has
    a structural consequence worth stating plainly: with paper_top_k and
    chunk_top_k both equal to max(top_ks), the first max(top_ks) entries are
    *all papers*, so `results[:10]` at k=10 contains zero chunks. Every
    chunk-ID label on a hybrid_both query is therefore unreachable at k<=10 no
    matter how well retrieval performs, and raising the probe to k=20 only
    lengthens the paper prefix. That accounts for the 12 cross_topic_comparison
    failures in docs/eval_failure_analysis_v2.md, which are exactly the queries
    whose ground truth is half chunk IDs.

    Passing merge_hybrid=True instead interleaves the two lists so both levels
    compete for the visible slots. It measures the same retrieval more
    faithfully, but it is not comparable to the concatenation numbers, so it is
    opt-in rather than default.

    Passing conditional_merge=True applies the interleave ONLY when the query
    is a diversity/comparison query (detected by is_diversity_query). This
    preserves paper-first ordering for non-comparison hybrid_both queries.
    """

    if route == "chunk_level":
        return list(response.chunk_results)
    if route == "hybrid_both":
        should_merge = merge_hybrid or (conditional_merge and is_diversity_query(response.query))
        if should_merge:
            return merge_ranked_lists(list(response.paper_results), list(response.chunk_results))
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
    expected = {canonical_identifier(identifier) for identifier in expected_relevant_ids}
    retrieved = {
        canonical_identifier(identifier)
        for result in results[:top_k]
        for identifier in result_identifiers(result)
    }
    return expected & retrieved


def reciprocal_rank(results: list[object], expected_relevant_ids: list[str]) -> float:
    expected = {canonical_identifier(identifier) for identifier in expected_relevant_ids}
    for index, result in enumerate(results, start=1):
        result_ids = {canonical_identifier(identifier) for identifier in result_identifiers(result)}
        if expected & result_ids:
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


def effective_routes(query: EvaluationQuery) -> list[str]:
    """Return the preferred route plus valid route alternatives for evaluation."""
    routes = [query.expected_route, *query.acceptable_routes]
    return list(dict.fromkeys(routes))


def evaluate_response(
    query: EvaluationQuery,
    response: UnifiedSearchResponse,
    top_ks: tuple[int, ...],
    *,
    rewrite_result: QueryRewriteResult | None = None,
    confidence: ConfidenceAssessment | None = None,
    merge_hybrid: bool = False,
    conditional_merge: bool = False,
) -> dict[str, object]:
    rewrite_result = rewrite_result or QueryRewriteResult(
        original_query=query.query,
        standalone_query=response.query,
        rewrite_used=response.query != query.query,
        method="none",
    )
    accepted_routes = effective_routes(query)
    route_correct = response.route.route in accepted_routes
    route_results = select_results(response, query.expected_route, merge_hybrid=merge_hybrid, conditional_merge=conditional_merge)
    combined_results = all_results(response)
    has_relevant_ids = bool(query.expected_relevant_ids)
    confidence_decision = confidence.decision if confidence else None

    topic_hits = {k: topic_hit(combined_results, query.expected_topics, k) for k in top_ks}
    keyword_hits = {k: keyword_hit(combined_results, query.expected_keywords, k) for k in top_ks}
    # Keep evaluation records JSON-native.  The previous set values were
    # stringified by ``json.dumps(default=str)``; notably, an empty set became
    # the non-empty string ``"set()"``.  Downstream reports then counted every
    # miss as a hit because that string is truthy.
    id_hit_sets = {
        k: sorted(id_hits(route_results, query.expected_relevant_ids, k))
        for k in top_ks
    }
    expected_id_count = len(
        {canonical_identifier(identifier) for identifier in query.expected_relevant_ids}
    )
    id_hit_fractions = {
        k: (len(id_hit_sets[k]) / expected_id_count if expected_id_count else None) for k in top_ks
    }
    rewrite_keyword_hit = text_contains_keywords(rewrite_result.standalone_query, query.expected_standalone_keywords)
    confidence_correct = (
        confidence_decision == query.expected_confidence_decision
        if query.expected_confidence_decision is not None and confidence_decision is not None
        else None
    )

    return {
        "query": query.query,
        "category": query.category,
        "evaluation_focus": query.evaluation_focus,
        "rationale": query.rationale,
        "standalone_query": rewrite_result.standalone_query,
        "rewrite_used": rewrite_result.rewrite_used,
        "rewrite_keyword_hit": rewrite_keyword_hit,
        "expected_route": query.expected_route,
        "acceptable_routes": accepted_routes,
        "actual_route": response.route.route,
        "route_confidence": response.route.confidence,
        "route_matched_signals": list(response.route.matched_signals),
        "route_correct": route_correct,
        "expected_relevant_ids": list(query.expected_relevant_ids),
        "expected_confidence_decision": query.expected_confidence_decision,
        "actual_confidence_decision": confidence_decision,
        "confidence_correct": confidence_correct,
        "has_relevant_ids": has_relevant_ids,
        "result_ids": [identifier for result in route_results if (identifier := result_id(result))],
        "topic_hits": topic_hits,
        "keyword_hits": keyword_hits,
        "id_hit_sets": id_hit_sets,
        "id_hit_fractions": id_hit_fractions,
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
    focus_counts = dict(sorted(Counter(str(evaluation.get("evaluation_focus") or "unspecified") for evaluation in evaluations).items()))
    route_confusion = dict(sorted(Counter(
        f"{evaluation.get('expected_route')}->{evaluation.get('actual_route')}"
        for evaluation in evaluations
    ).items()))
    fallback_routes = sum(
        1 for evaluation in evaluations
        if any("fallback:" in str(signal) for signal in evaluation.get("route_matched_signals", []))
    )

    route_accuracy = safe_rate(sum(1 for evaluation in evaluations if evaluation["route_correct"]), total)
    rewrite_keyword_hit_rate = safe_rate(sum(1 for evaluation in rewrite_labeled if evaluation["rewrite_keyword_hit"]), len(rewrite_labeled))
    confidence_decision_accuracy = safe_rate(sum(1 for evaluation in confidence_labeled if evaluation["confidence_correct"]), len(confidence_labeled))
    crag_fallback_success_rate = safe_rate(sum(1 for evaluation in fallback_labeled if evaluation["confidence_correct"]), len(fallback_labeled))

    topic_counts = {}
    keyword_counts = {}
    hit_rate_counts = {}
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
        hit_rate_counts[k] = {
            "value": safe_rate(sum(1 for evaluation in labeled if evaluation["id_hit_sets"][k]), labeled_count),
            "n": labeled_count,
        }
        recall_fractions = [
            evaluation["id_hit_fractions"][k]
            for evaluation in labeled
            if evaluation["id_hit_fractions"][k] is not None
        ]
        recall_counts[k] = {
            "value": (sum(recall_fractions) / len(recall_fractions)) if recall_fractions else None,
            "n": len(recall_fractions),
        }

    mrr_values = [float(evaluation["reciprocal_rank"]) for evaluation in labeled]
    mrr = (sum(mrr_values) / labeled_count) if labeled_count else None

    return {
        "queries": total,
        "queries_with_relevant_ids": labeled_count,
        "queries_topic_keyword_only": total - labeled_count,
        "multi_turn_queries": len(multi_turn),
        "out_of_corpus_queries": len(out_of_corpus),
        "evaluation_focus_counts": focus_counts,
        "route_confusion": route_confusion,
        "fallback_route_count": fallback_routes,
        "route_accuracy": route_accuracy,
        "rewrite_keyword_hit_rate": {"value": rewrite_keyword_hit_rate, "n": len(rewrite_labeled)},
        "confidence_decision_accuracy": {"value": confidence_decision_accuracy, "n": len(confidence_labeled)},
        "crag_fallback_success_rate": {"value": crag_fallback_success_rate, "n": len(fallback_labeled)},
        "topic_hit_rate": topic_counts,
        "keyword_hit_rate": keyword_counts,
        "id_relevant_hit_rate": hit_rate_counts,
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
    fusion_method: str = "weighted",
    local_path: Path | None = None,
    qdrant_url: str | None = None,
    rewriter: RewriteRunner = rewrite_query,
    confidence_checker: ConfidenceRunner = assess_confidence,
    apply_promotion: bool = DEFAULT_APPLY_PROMOTION,
    pool_multiplier: int = DEFAULT_PROMOTION_POOL_MULTIPLIER,
    extended_expansions: bool = False,
    merge_hybrid: bool = False,
    conditional_merge: bool = False,
    reading_path_boost: bool = False,
    affinity_boost: bool = False,
    paper_collection: str = DEFAULT_PAPER_COLLECTION,
    chunk_collection: str = DEFAULT_CHUNK_COLLECTION,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    evaluations = []
    max_top_k = max(top_ks)
    for query in queries:
        rewrite_result = maybe_rewrite_query(query, rewriter=rewriter, enabled=apply_query_rewriting)
        response = search_runner(
            rewrite_result.standalone_query,
            top_k=max_top_k,
            apply_reranking=apply_reranking,
            fusion_method=fusion_method,
            local_path=local_path,
            qdrant_url=qdrant_url,
            apply_promotion=apply_promotion,
            pool_multiplier=pool_multiplier,
            extended_expansions=extended_expansions,
            reading_path_boost=reading_path_boost,
            affinity_boost=affinity_boost,
            paper_collection=paper_collection,
            chunk_collection=chunk_collection,
        )
        confidence = confidence_checker(response) if query.expected_confidence_decision is not None else None
        evaluations.append(
            evaluate_response(
                query,
                response,
                top_ks,
                rewrite_result=rewrite_result,
                confidence=confidence,
                merge_hybrid=merge_hybrid,
                conditional_merge=conditional_merge,
            )
        )
    return summarize_evaluations(evaluations, top_ks), evaluations


def summary_to_text(summary: dict[str, object], top_ks: tuple[int, ...]) -> str:
    lines = [
        f"queries: {summary['queries']}",
        f"queries_with_relevant_ids: {summary['queries_with_relevant_ids']}",
        f"queries_topic_keyword_only: {summary['queries_topic_keyword_only']}",
        f"multi_turn_queries: {summary['multi_turn_queries']}",
        f"out_of_corpus_queries: {summary['out_of_corpus_queries']}",
        "evaluation_focus_counts: " + ", ".join(f"{key}={value}" for key, value in sorted(summary.get("evaluation_focus_counts", {}).items())),
        f"route_accuracy: {format_rate(summary['route_accuracy'])}",
        f"rewrite_keyword_hit_rate: {format_rate(summary['rewrite_keyword_hit_rate']['value'])} (contextual subset, n={summary['rewrite_keyword_hit_rate']['n']})",
        f"confidence_decision_accuracy: {format_rate(summary['confidence_decision_accuracy']['value'])} (labeled confidence subset, n={summary['confidence_decision_accuracy']['n']})",
        f"crag_fallback_success_rate: {format_rate(summary['crag_fallback_success_rate']['value'])} (expected fallback subset, n={summary['crag_fallback_success_rate']['n']})",
    ]
    for k in top_ks:
        topic = summary["topic_hit_rate"][k]
        keyword = summary["keyword_hit_rate"][k]
        hit_rate = summary["id_relevant_hit_rate"][k]
        recall = summary["recall"][k]
        lines.append(f"topic_hit_rate@{k}: {format_rate(topic['value'])} (sanity check, n={topic['n']})")
        lines.append(f"keyword_hit_rate@{k}: {format_rate(keyword['value'])} (sanity check, n={keyword['n']})")
        lines.append(f"hit_rate@{k} (>=1 relevant id in top-{k}, labeled subset, n={hit_rate['n']}): {format_rate(hit_rate['value'])}")
        lines.append(f"recall@{k} (fraction of all relevant ids retrieved, labeled subset, n={recall['n']}): {format_rate(recall['value'])}")
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
    parser.add_argument("--fusion-method", choices=["weighted", "rrf"], default="weighted")
    parser.add_argument("--local-path", type=Path, default=None, help="Use an embedded/local Qdrant snapshot instead of a running server.")
    parser.add_argument("--qdrant-url", default=None, help="URL of a running Qdrant server (ignored if --local-path is set).")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    parser.add_argument(
        "--promotion",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_APPLY_PROMOTION,
        help="Re-order paper/chunk candidates with route-aware promotion before the top-k cut.",
    )
    parser.add_argument(
        "--pool-multiplier",
        type=int,
        default=DEFAULT_PROMOTION_POOL_MULTIPLIER,
        help="Retrieve this many times top_k internally before reducing to top_k.",
    )
    parser.add_argument(
        "--extended-expansions",
        action="store_true",
        help="Also apply the broader research-vocabulary query expansion table.",
    )
    parser.add_argument(
        "--merge-hybrid",
        action="store_true",
        help=(
            "Interleave paper and chunk results for hybrid_both instead of concatenating them. "
            "Changes metric semantics; numbers are not comparable to previous runs."
        ),
    )
    parser.add_argument(
        "--conditional-merge",
        action="store_true",
        help="Interleave paper/chunk results only for diversity/comparison queries.",
    )
    parser.add_argument(
        "--reading-path-boost",
        action="store_true",
        help="Boost high-citation surveys for reading-path queries (requires --promotion).",
    )
    parser.add_argument(
        "--affinity",
        action="store_true",
        help="Boost chunks whose parent paper was also retrieved (requires --promotion).",
    )
    parser.add_argument(
        "--paper-collection",
        default=DEFAULT_PAPER_COLLECTION,
        help="Qdrant collection for paper-level vectors.",
    )
    parser.add_argument(
        "--chunk-collection",
        default=DEFAULT_CHUNK_COLLECTION,
        help="Qdrant collection for chunk-level vectors (use e.g. research_paper_chunks_v2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_eval_queries(args.queries)
    summary, evaluations = run_evaluation(
        queries,
        top_ks=args.top_ks,
        apply_reranking=not args.no_rerank,
        fusion_method=args.fusion_method,
        local_path=args.local_path,
        qdrant_url=args.qdrant_url,
        apply_promotion=args.promotion,
        pool_multiplier=args.pool_multiplier,
        extended_expansions=args.extended_expansions,
        merge_hybrid=args.merge_hybrid,
        conditional_merge=args.conditional_merge,
        reading_path_boost=args.reading_path_boost,
        affinity_boost=args.affinity,
        paper_collection=args.paper_collection,
        chunk_collection=args.chunk_collection,
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
