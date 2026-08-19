from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agentic.planner import RoutePlan, plan_query


@dataclass(frozen=True)
class PlannerEvalCase:
    query: str
    expected_route: str
    expected_tools: tuple[str, ...]


DEFAULT_PLANNER_CASES = (
    PlannerEvalCase("What does the indexed corpus say about RAG?", "corpus", ("search_local_corpus",)),
    PlannerEvalCase("Compare current RAG papers with the indexed papers.", "hybrid", ("search_local_corpus", "search_arxiv", "search_semantic_scholar", "search_tavily")),
    PlannerEvalCase("What are the latest papers on hallucination detection?", "live", ("search_arxiv", "search_semantic_scholar", "search_tavily")),
    PlannerEvalCase("Explain the LoRA method in our papers.", "corpus", ("search_local_corpus",)),
    PlannerEvalCase("Find recent papers about agentic RAG.", "live", ("search_arxiv", "search_semantic_scholar", "search_tavily")),
    PlannerEvalCase("What is the difference between RAG and fine-tuning?", "corpus", ("search_local_corpus",)),
)


def evaluate_planner(
    cases: tuple[PlannerEvalCase, ...] = DEFAULT_PLANNER_CASES,
    *,
    planner: Callable[[str], RoutePlan] = plan_query,
) -> dict[str, Any]:
    failures = []
    route_hits = 0
    tool_hits = 0
    for case in cases:
        actual = planner(case.query)
        route_ok = actual.route == case.expected_route
        tools_ok = tuple(actual.tools) == tuple(case.expected_tools)
        route_hits += int(route_ok)
        tool_hits += int(tools_ok)
        if not route_ok or not tools_ok:
            failures.append(
                {
                    "query": case.query,
                    "expected_route": case.expected_route,
                    "actual_route": actual.route,
                    "expected_tools": list(case.expected_tools),
                    "actual_tools": list(actual.tools),
                }
            )
    total = len(cases)
    return {
        "cases": total,
        "route_accuracy": round(route_hits / total, 3) if total else 0.0,
        "tool_plan_accuracy": round(tool_hits / total, 3) if total else 0.0,
        "failures": failures,
    }


def validate_grounded_response(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = payload.get("evidence") or []
    citations = payload.get("citations") or []
    valid_citations = {
        f"source_{index}"
        for index in range(1, len(evidence) + 1)
    }
    recognized = [citation for citation in citations if citation in valid_citations]
    return {
        "evidence_count": len(evidence),
        "citation_count": len(citations),
        "recognized_citation_count": len(recognized),
        "citation_coverage": round(len(recognized) / len(citations), 3) if citations else 0.0,
        "citations_valid": bool(citations) and len(recognized) == len(citations),
    }
