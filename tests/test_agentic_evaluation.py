from agentic.evaluation import evaluate_planner, validate_grounded_response


def test_default_planner_benchmark_covers_all_routes():
    metrics = evaluate_planner()

    assert metrics["cases"] == 6
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
