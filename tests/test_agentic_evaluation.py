import json
from pathlib import Path

from agentic.evaluation import (
    AgenticEvalCase,
    evaluate_agentic_responses,
    evaluate_planner,
    load_agentic_cases,
    validate_grounded_response,
)


def test_default_planner_benchmark_covers_all_routes():
    metrics = evaluate_planner()

    assert metrics["cases"] == 6
    assert metrics["route_accuracy"] == 1.0
    assert metrics["tool_plan_accuracy"] == 1.0
    assert metrics["failures"] == []


def test_agentic_fixture_covers_routes_and_refusals():
    root = Path(__file__).resolve().parents[1]
    cases = load_agentic_cases(root / "tests/fixtures/agentic_eval_queries.json")
    metrics = evaluate_planner(cases)

    assert len(cases) == 46
    assert {case.expected_route for case in cases} == {"corpus", "live", "hybrid"}
    assert any(not case.should_answer for case in cases)
    assert sum(case.expected_external_source for case in cases) == 16
    assert all(case.corpus_absent for case in cases if case.expected_external_source)
    assert metrics["route_accuracy"] == 1.0
    assert metrics["tool_plan_accuracy"] == 1.0
    assert metrics["failures"] == []


def test_validate_grounded_response_reports_citation_coverage():
    payload = {
        "evidence": [{"title": "Paper One"}, {"title": "Paper Two"}],
        "citations": ["source_1", "source_2"],
    }

    metrics = validate_grounded_response(payload)

    assert metrics == {
        "evidence_count": 2,
        "citation_count": 2,
        "recognized_citation_count": 2,
        "citation_coverage": 1.0,
        "citations_valid": True,
    }


def test_validate_grounded_response_flags_unknown_citations():
    metrics = validate_grounded_response(
        {"evidence": [{"title": "Paper One"}], "citations": ["source_9"]}
    )

    assert metrics["citations_valid"] is False
    assert metrics["recognized_citation_count"] == 0
    assert metrics["citation_coverage"] == 0.0


def test_recorded_response_metrics_cover_tools_answers_and_refusals():
    cases = (
        AgenticEvalCase(case_id="answer", query="answer", expected_route="corpus", expected_tools=("search_local_corpus",), should_answer=True),
        AgenticEvalCase(case_id="refusal", query="refusal", expected_route="corpus", expected_tools=("search_local_corpus",), should_answer=False),
    )
    responses = {
        "answer": {
            "status": "completed",
            "route": "corpus",
            "planned_tools": ["search_local_corpus"],
            "tool_calls": [{"tool": "search_local_corpus", "status": "completed"}],
            "answer": "Grounded answer [source_1]",
            "evidence": [{"title": "Paper One"}],
            "citations": ["source_1"],
        },
        "refusal": {
            "status": "completed",
            "route": "corpus",
            "planned_tools": ["search_local_corpus"],
            "tool_calls": [{"tool": "search_local_corpus", "status": "completed"}],
            "confidence_decision": "insufficient_evidence",
        },
    }

    metrics = evaluate_agentic_responses(cases, responses)

    assert metrics["route_accuracy"] == 1.0
    assert metrics["tool_plan_accuracy"] == 1.0
    assert metrics["tool_success_rate"] == 1.0
    assert metrics["answer_or_refusal_accuracy"] == 1.0
    assert metrics["citation_validity_rate"] == 1.0
    assert metrics["citation_coverage"] == 1.0
    assert metrics["failures"] == []


def test_recorded_response_metrics_validate_external_fallback():
    case = AgenticEvalCase(
        case_id="fallback",
        query="external",
        expected_route="corpus",
        expected_tools=("search_local_corpus",),
        response_route="hybrid",
        response_tools=(
            "search_local_corpus",
            "search_arxiv",
            "search_semantic_scholar",
            "search_tavily",
        ),
        expected_external_source=True,
    )
    response = {
        "status": "completed",
        "route": "hybrid",
        "planned_tools": list(case.response_tools),
        "tool_calls": [
            {"tool": tool, "status": "completed"}
            for tool in case.response_tools
        ],
        "answer": "Grounded external answer [source_1]",
        "evidence": [{"kind": "external", "title": "External paper"}],
        "citations": ["source_1"],
    }

    metrics = evaluate_agentic_responses((case,), {"fallback": response})

    assert metrics["route_accuracy"] == 1.0
    assert metrics["tool_plan_accuracy"] == 1.0
    assert metrics["external_source_accuracy"] == 1.0
    assert metrics["external_source_cases"] == 1
    assert metrics["failures"] == []


def test_external_only_cases_are_absent_from_current_corpus_titles():
    root = Path(__file__).resolve().parents[1]
    cases = load_agentic_cases(root / "tests/fixtures/agentic_eval_queries.json")
    papers = json.loads((root / "data/full_text_papers.json").read_text())
    titles = [str(item.get("title", "")).lower() for item in papers]
    for case in cases:
        if case.corpus_absent:
            assert case.external_reference
            needle = case.external_reference.lower()
            assert not any(needle in title for title in titles), case.external_reference
