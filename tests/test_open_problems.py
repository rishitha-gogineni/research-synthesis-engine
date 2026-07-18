import json
import subprocess
import sys

import pytest

from agent.open_problems import OpenProblemsError, build_open_problems_report, evidence_strength, extract_problem_evidence
from shared.schemas import ConfidenceAssessment, QueryRoute, RetrievedChunk, RetrievedPaper, UnifiedSearchResponse


def make_route():
    return QueryRoute(query="What are unresolved problems in hallucination detection?", route="hybrid_both", reason="test", confidence=0.95)


def make_confidence(decision="sufficient_evidence"):
    return ConfidenceAssessment(
        query="What are unresolved problems in hallucination detection?",
        route="hybrid_both",
        confidence_score=0.9,
        decision=decision,
        reason="retrieval is strong",
        recommended_action="proceed",
        signals=["top_score=0.95"],
        result_count=4,
        top_score=0.95,
        route_confidence=0.95,
    )


def make_paper(paper_id, limitation, score=0.88):
    return RetrievedPaper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        topic="LLM Evaluation & Hallucination Detection",
        year=2024,
        citation_count=50,
        abstract="Hallucination detection evaluation paper.",
        main_contribution="Studies hallucination detection.",
        methodology="Benchmark evaluation.",
        dataset_used="HaluEval",
        key_result="Detection remains difficult across domains.",
        limitations=limitation,
        blended_score=score,
        rerank_score=score,
    )


def make_chunk(chunk_id, paper_id, text, score=0.9):
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        topic="LLM Evaluation & Hallucination Detection",
        year=2024,
        citation_count=50,
        section_hint="Future Work",
        text=text,
        blended_score=score,
        rerank_score=score,
        dense_score=score,
    )


def make_response(papers=None, chunks=None):
    papers = papers or []
    chunks = chunks or []
    return UnifiedSearchResponse(
        query="What are unresolved problems in hallucination detection?",
        route=make_route(),
        paper_result_count=len(papers),
        chunk_result_count=len(chunks),
        paper_results=papers,
        chunk_results=chunks,
    )


def generator_payload(problems):
    return json.dumps(
        {
            "problems": problems,
            "recurring_limitations": ["evaluation: benchmark coverage remains limited"],
            "conflicting_findings": ["Paper p1 reports gains; however Paper p2 notes domain failures."],
            "evidence_gaps": ["Few retrieved sources cover deployment settings."],
            "corpus_limitations": ["Limited to retrieved papers."],
        }
    )


def test_repeated_limitations_are_available_for_merging():
    response = make_response(
        papers=[
            make_paper("p1", "Evaluation is limited by benchmark coverage."),
            make_paper("p2", "Evaluation is limited by benchmark coverage."),
        ]
    )

    evidence = extract_problem_evidence(__import__("agent.guidance_common", fromlist=["normalize_candidates"]).normalize_candidates(response))

    assert len(evidence) == 1
    assert evidence[0]["category_hint"] == "evaluation"


def test_source_ids_are_preserved_and_multiple_papers_are_strong():
    response = make_response(
        papers=[
            make_paper("p1", "Evaluation is limited by benchmark coverage."),
            make_paper("p2", "Generalization remains limited across domains."),
        ]
    )
    problem = {
        "title": "Cross-domain hallucination evaluation remains under-supported",
        "description": "Retrieved papers point to benchmark and generalization limits.",
        "category": "evaluation",
        "why_it_matters": "Detection systems may not transfer across domains.",
        "evidence_summary": "Two retrieved papers discuss benchmark and generalization limits.",
        "supporting_paper_ids": ["p1", "p2"],
        "supporting_source_ids": ["paper:p1", "paper:p2"],
        "evidence_snippets": ["Evaluation is limited by benchmark coverage."],
        "suggested_research_directions": ["Evaluate across more domains."],
        "confidence": 0.8,
    }

    report = build_open_problems_report(response, confidence=make_confidence(), generator=lambda _: generator_payload([problem]))

    assert report.problems[0].supporting_source_ids == ["paper:p1", "paper:p2"]
    assert report.problems[0].evidence_strength == "strong"


def test_unsupported_problem_without_source_ids_is_rejected_from_output():
    response = make_response(papers=[make_paper("p1", "Safety limitations remain unresolved.")])
    problem = {
        "title": "Unsupported problem",
        "description": "No support.",
        "category": "safety",
        "why_it_matters": "No support.",
        "evidence_summary": "No support.",
        "supporting_paper_ids": [],
        "supporting_source_ids": [],
        "evidence_snippets": [],
        "suggested_research_directions": [],
    }

    report = build_open_problems_report(response, confidence=make_confidence(), generator=lambda _: generator_payload([problem]))

    assert report.problems == []
    assert any("No validated" in item for item in report.corpus_limitations)


def test_unknown_source_ids_raise_validation_error():
    response = make_response(papers=[make_paper("p1", "Safety limitations remain unresolved.")])
    problem = {
        "title": "Safety problem",
        "description": "Supported by bad source id.",
        "category": "safety",
        "why_it_matters": "It matters.",
        "evidence_summary": "Safety limitations remain unresolved.",
        "supporting_paper_ids": ["p1"],
        "supporting_source_ids": ["paper:unknown"],
        "evidence_snippets": ["Safety limitations remain unresolved."],
        "suggested_research_directions": [],
    }

    with pytest.raises(OpenProblemsError):
        build_open_problems_report(response, confidence=make_confidence(), generator=lambda _: generator_payload([problem]))


def test_one_paper_evidence_is_not_labeled_strong():
    assert evidence_strength(["p1"], ["paper:p1"]) == "weak"
    assert evidence_strength(["p1"], ["paper:p1", "chunk:c1"]) == "moderate"


def test_no_limitation_evidence_returns_guarded_report():
    response = make_response(papers=[make_paper("p1", "not stated in abstract")])

    def should_not_run(_):
        raise AssertionError("generator should not run")

    report = build_open_problems_report(response, confidence=make_confidence(), generator=should_not_run)

    assert report.problems == []
    assert report.confidence_decision == "sufficient_evidence"
    assert any("did not contain clear limitations" in item for item in report.corpus_limitations)


def test_low_confidence_retrieval_skips_generation():
    response = make_response(papers=[make_paper("p1", "Safety limitations remain unresolved.")])

    def should_not_run(_):
        raise AssertionError("generator should not run")

    report = build_open_problems_report(response, confidence=make_confidence(decision="ask_clarifying_question"), generator=should_not_run)

    assert report.problems == []
    assert report.confidence_decision == "ask_clarifying_question"


def test_malformed_json_receives_one_retry():
    response = make_response(papers=[make_paper("p1", "Safety limitations remain unresolved.")])
    calls = []
    problem = {
        "title": "Safety limitations need stronger validation",
        "description": "Retrieved evidence states safety limitations remain unresolved.",
        "category": "safety",
        "why_it_matters": "Safety failures affect reliability.",
        "evidence_summary": "Safety limitations remain unresolved.",
        "supporting_paper_ids": ["p1"],
        "supporting_source_ids": ["paper:p1"],
        "evidence_snippets": ["Safety limitations remain unresolved."],
        "suggested_research_directions": ["Test safety failures across domains."],
    }

    def flaky_generator(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "not json"
        return generator_payload([problem])

    report = build_open_problems_report(response, confidence=make_confidence(), generator=flaky_generator)

    assert len(calls) == 2
    assert report.problems[0].title.startswith("Safety")


def test_open_problems_cli_accepts_low_confidence_input_without_openai(tmp_path):
    response = make_response()
    path = tmp_path / "response.json"
    path.write_text(response.model_dump_json())

    completed = subprocess.run(
        [sys.executable, "-m", "agent.open_problems", "--input", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["problems"] == []
    assert payload["confidence_decision"] in {"insufficient_evidence", "broaden_search", "ask_clarifying_question"}
