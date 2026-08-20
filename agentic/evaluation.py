from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from agentic.planner import RoutePlan, plan_query


@dataclass(frozen=True)
class PlannerEvalCase:
    query: str
    expected_route: str
    expected_tools: tuple[str, ...]


@dataclass(frozen=True)
class AgenticEvalCase(PlannerEvalCase):
    case_id: str
    should_answer: bool = True


DEFAULT_PLANNER_CASES = (
    PlannerEvalCase("What does the indexed corpus say about RAG?", "corpus", ("search_local_corpus",)),
    PlannerEvalCase("Compare current RAG papers with the indexed papers.", "hybrid", ("search_local_corpus", "search_arxiv", "search_semantic_scholar", "search_tavily")),
    PlannerEvalCase("What are the latest papers on hallucination detection?", "live", ("search_arxiv", "search_semantic_scholar", "search_tavily")),
    PlannerEvalCase("Explain the LoRA method in our papers.", "corpus", ("search_local_corpus",)),
    PlannerEvalCase("Find recent papers about agentic RAG.", "live", ("search_arxiv", "search_semantic_scholar", "search_tavily")),
    PlannerEvalCase("What is the difference between RAG and fine-tuning?", "corpus", ("search_local_corpus",)),
)


def load_agentic_cases(path: str | Path) -> tuple[AgenticEvalCase, ...]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError("agentic evaluation fixture must contain a JSON list")
    cases = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each agentic evaluation case must be an object")
        cases.append(
            AgenticEvalCase(
                case_id=str(item["id"]),
                query=str(item["query"]),
                expected_route=str(item["expected_route"]),
                expected_tools=tuple(str(tool) for tool in item["expected_tools"]),
                should_answer=bool(item.get("should_answer", True)),
            )
        )
    return tuple(cases)


def evaluate_planner(
    cases: Iterable[PlannerEvalCase] = DEFAULT_PLANNER_CASES,
    *,
    planner: Callable[[str], RoutePlan] = plan_query,
) -> dict[str, Any]:
    failures = []
    route_hits = 0
    tool_hits = 0
    cases = tuple(cases)
    for case in cases:
        actual = planner(case.query)
        route_ok = actual.route == case.expected_route
        tools_ok = tuple(actual.tools) == tuple(case.expected_tools)
        route_hits += int(route_ok)
        tool_hits += int(tools_ok)
        if not route_ok or not tools_ok:
            failures.append(
                {
                    "id": getattr(case, "case_id", None),
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
    valid_citations = {f"source_{index}" for index in range(1, len(evidence) + 1)}
    recognized = [citation for citation in citations if citation in valid_citations]
    return {
        "evidence_count": len(evidence),
        "citation_count": len(citations),
        "recognized_citation_count": len(recognized),
        "citation_coverage": round(len(recognized) / len(citations), 3) if citations else 0.0,
        "citations_valid": bool(citations) and len(recognized) == len(citations),
    }


def evaluate_agentic_responses(
    cases: Iterable[AgenticEvalCase],
    responses: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = tuple(cases)
    route_hits = tool_hits = refusal_hits = 0
    tool_successes = tool_total = 0
    answered_cases = 0
    valid_citation_cases = 0
    total_citations = recognized_citations = 0
    failures = []

    for case in cases:
        response = responses.get(case.case_id) or {}
        route_ok = response.get("route") == case.expected_route
        tools_ok = tuple(response.get("planned_tools") or []) == tuple(case.expected_tools)
        status_ok = response.get("status") == "completed" and not response.get("error")
        tool_entries = list(response.get("tool_calls") or []) + list(response.get("llm_tool_calls") or [])
        if tool_entries:
            tool_total += len(tool_entries)
            tool_successes += sum(
                1 for item in tool_entries
                if (item.get("status") or "completed") not in {"failed", "budget_exhausted"}
            )
        else:
            tool_total += 1
            tool_successes += int(status_ok)
        answer_present = bool(str(response.get("answer") or "").strip())
        decision = response.get("confidence_decision")
        refusal_ok = (answer_present if case.should_answer else not answer_present) or (
            not case.should_answer and decision in {"insufficient_evidence", "ask_clarifying_question"}
        )
        citation_metrics = validate_grounded_response(response)
        route_hits += int(route_ok)
        tool_hits += int(tools_ok)
        refusal_hits += int(refusal_ok)
        if answer_present:
            answered_cases += 1
            total_citations += citation_metrics["citation_count"]
            recognized_citations += citation_metrics["recognized_citation_count"]
            valid_citation_cases += int(citation_metrics["citations_valid"])

        if not (route_ok and tools_ok and status_ok and refusal_ok):
            failures.append(
                {
                    "id": case.case_id,
                    "route_ok": route_ok,
                    "tools_ok": tools_ok,
                    "status_ok": status_ok,
                    "refusal_or_answer_ok": refusal_ok,
                    "actual_route": response.get("route"),
                    "actual_tools": response.get("planned_tools", []),
                }
            )

    total = len(cases)
    return {
        "cases": total,
        "route_accuracy": round(route_hits / total, 3) if total else 0.0,
        "tool_plan_accuracy": round(tool_hits / total, 3) if total else 0.0,
        "tool_success_rate": round(tool_successes / tool_total, 3) if tool_total else 0.0,
        "answer_or_refusal_accuracy": round(refusal_hits / total, 3) if total else 0.0,
        "citation_validity_rate": round(valid_citation_cases / answered_cases, 3) if answered_cases else 0.0,
        "citation_coverage": round(recognized_citations / total_citations, 3) if total_citations else 0.0,
        "failures": failures,
    }
