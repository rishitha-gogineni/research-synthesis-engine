import json
import subprocess
import sys

from agent.research_guidance import build_research_guidance
from shared.schemas import ConfidenceAssessment, QueryRoute, RetrievedPaper, UnifiedSearchResponse


def make_confidence(decision="sufficient_evidence"):
    return ConfidenceAssessment(
        query="Compare RAG and self-verification methods.",
        route="hybrid_both",
        confidence_score=0.92,
        decision=decision,
        reason="retrieval is strong",
        recommended_action="proceed",
        signals=["top_score=0.95"],
        result_count=2,
        top_score=0.95,
        route_confidence=0.95,
    )


def make_response():
    papers = [
        RetrievedPaper(
            paper_id="p1",
            title="RAG Paper",
            topic="Retrieval-Augmented Generation (RAG)",
            year=2020,
            citation_count=100,
            abstract="RAG grounds generation in retrieved evidence.",
            main_contribution="Introduces retrieval grounding.",
            methodology="Retrieval method.",
            dataset_used="not stated in abstract",
            key_result="Grounding can improve factuality.",
            limitations="Evaluation coverage is limited.",
            blended_score=0.95,
            rerank_score=0.95,
        ),
        RetrievedPaper(
            paper_id="p2",
            title="Self Verification Paper",
            topic="LLM Evaluation & Hallucination Detection",
            year=2024,
            citation_count=20,
            abstract="Self-verification checks generated claims.",
            main_contribution="Studies verification for hallucination reduction.",
            methodology="Verification method.",
            dataset_used="HaluEval",
            key_result="Verification can catch unsupported claims.",
            limitations="Generalization remains limited.",
            blended_score=0.9,
            rerank_score=0.9,
        ),
    ]
    return UnifiedSearchResponse(
        query="Compare RAG and self-verification methods.",
        route=QueryRoute(query="Compare RAG and self-verification methods.", route="hybrid_both", reason="test", confidence=0.95),
        paper_result_count=len(papers),
        chunk_result_count=0,
        paper_results=papers,
        chunk_results=[],
    )


def reading_generator(_):
    return json.dumps(
        {
            "stages": [
                {
                    "stage": "foundational",
                    "description": "Start with retrieval grounding.",
                    "papers": [
                        {
                            "paper_id": "p1",
                            "reason_to_read": "It frames retrieval grounding.",
                            "focus_points": ["grounding"],
                            "prerequisites": [],
                            "connection_to_next": "Then compare verification.",
                            "source_ids": ["paper:p1"],
                        }
                    ],
                },
                {
                    "stage": "recent_advances",
                    "description": "Then read recent verification work.",
                    "papers": [
                        {
                            "paper_id": "p2",
                            "reason_to_read": "It provides a newer verification angle.",
                            "focus_points": ["verification"],
                            "prerequisites": [],
                            "connection_to_next": None,
                            "source_ids": ["paper:p2"],
                        }
                    ],
                },
            ],
            "limitations": [],
        }
    )


def problems_generator(_):
    return json.dumps(
        {
            "problems": [
                {
                    "title": "Evaluation and generalization remain limited",
                    "description": "Retrieved papers mention limited evaluation coverage and generalization.",
                    "category": "evaluation",
                    "why_it_matters": "Comparisons are weaker when benchmarks do not generalize.",
                    "evidence_summary": "Both retrieved papers mention evaluation or generalization limits.",
                    "supporting_paper_ids": ["p1", "p2"],
                    "supporting_source_ids": ["paper:p1", "paper:p2"],
                    "evidence_snippets": ["Evaluation coverage is limited.", "Generalization remains limited."],
                    "suggested_research_directions": ["Run matched comparisons across domains."],
                }
            ],
            "recurring_limitations": [],
            "conflicting_findings": [],
            "evidence_gaps": [],
            "corpus_limitations": [],
        }
    )


def test_combined_guidance_reuses_one_response_and_preserves_source_ids():
    guidance = build_research_guidance(
        make_response(),
        confidence=make_confidence(),
        reading_generator=reading_generator,
        problems_generator=problems_generator,
    )

    assert guidance.confidence.decision == "sufficient_evidence"
    assert guidance.reading_path is not None
    assert guidance.open_problems is not None
    reading_sources = {source_id for stage in guidance.reading_path.stages for item in stage.papers for source_id in item.source_ids}
    problem_sources = set(guidance.open_problems.problems[0].supporting_source_ids)
    assert reading_sources == {"paper:p1", "paper:p2"}
    assert problem_sources == {"paper:p1", "paper:p2"}


def test_combined_guidance_low_confidence_skips_day19_generation():
    guidance = build_research_guidance(make_response(), confidence=make_confidence(decision="broaden_search"))

    assert guidance.reading_path is None
    assert guidance.open_problems is None
    assert guidance.warnings


def test_research_guidance_cli_accepts_low_confidence_input_without_openai(tmp_path):
    response = UnifiedSearchResponse(
        query="empty",
        route=QueryRoute(query="empty", route="hybrid_both", reason="test", confidence=0.2),
        paper_result_count=0,
        chunk_result_count=0,
        paper_results=[],
        chunk_results=[],
    )
    path = tmp_path / "response.json"
    path.write_text(response.model_dump_json())

    completed = subprocess.run(
        [sys.executable, "-m", "agent.research_guidance", "--input", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["reading_path"] is None
    assert payload["open_problems"] is None
    assert payload["confidence"]["decision"] == "insufficient_evidence"
