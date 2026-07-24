import json
import subprocess
import sys

import pytest

from retrieval.confidence import (
    assess_confidence,
    candidate_score,
    confidence_components,
    load_response,
    score_topic_agreement,
)
from shared.schemas import QueryRoute, RetrievedChunk, RetrievedPaper, UnifiedSearchResponse


def make_route(route="paper_level", confidence=0.9, query="hallucination benchmark evidence"):
    return QueryRoute(query=query, route=route, reason="test", confidence=confidence)


def make_paper(paper_id="p1", topic="LLM Evaluation & Hallucination Detection", score=0.9):
    return RetrievedPaper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        topic=topic,
        abstract="hallucination benchmark evidence",
        citation_count=10,
        blended_score=score,
        rerank_score=score,
    )


def make_chunk(chunk_id="c1", paper_id="p1", topic="LLM Evaluation & Hallucination Detection", score=0.85):
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title=f"Chunk {chunk_id}",
        topic=topic,
        text="TruthfulQA and HaluEval evidence",
        citation_count=5,
        blended_score=score,
        rerank_score=score,
        dense_score=score,
    )


def make_response(route="paper_level", route_confidence=0.9, papers=None, chunks=None, query="hallucination benchmark evidence"):
    papers = papers or []
    chunks = chunks or []
    return UnifiedSearchResponse(
        query=query,
        route=make_route(route, route_confidence, query=query),
        paper_result_count=len(papers),
        chunk_result_count=len(chunks),
        paper_results=papers,
        chunk_results=chunks,
    )


def test_candidate_score_prefers_blended_then_fallbacks():
    assert candidate_score(make_paper(score=0.8)) == 0.8
    assert candidate_score(RetrievedPaper(title="A", topic="RAG", hybrid_score=0.6)) == 0.6
    assert candidate_score(RetrievedChunk(title="C", topic="RAG", text="txt", dense_score=0.7)) == 0.7


def test_high_confidence_paper_results_are_sufficient():
    response = make_response(papers=[make_paper("p1", score=0.95), make_paper("p2", score=0.88), make_paper("p3", score=0.8)])

    assessment = assess_confidence(response)

    assert assessment.decision == "sufficient_evidence"
    assert assessment.confidence_score >= 0.72
    assert assessment.result_count == 3
    assert any("top_score" in signal for signal in assessment.signals)


def test_no_results_are_insufficient():
    assessment = assess_confidence(make_response())

    assert assessment.decision == "insufficient_evidence"
    assert assessment.confidence_score == 0.18
    assert assessment.result_count == 0


def test_low_score_sparse_results_broaden_search():
    response = make_response(papers=[make_paper("p1", score=0.2)])

    assessment = assess_confidence(response)

    assert assessment.decision == "broaden_search"
    assert "weak" in assessment.reason.lower() or "sparse" in assessment.reason.lower()


def test_ambiguous_route_with_weak_evidence_asks_clarifying_question():
    response = make_response(route="hybrid_both", route_confidence=0.55, papers=[make_paper("p1", score=0.45)], chunks=[])

    assessment = assess_confidence(response)

    assert assessment.decision == "ask_clarifying_question"
    assert assessment.route_confidence == 0.55


def test_hybrid_both_agreement_boosts_confidence():
    response = make_response(
        route="hybrid_both",
        papers=[make_paper("p1", topic="RAG", score=0.8), make_paper("p2", topic="RAG", score=0.75)],
        chunks=[make_chunk("c1", paper_id="p1", topic="RAG", score=0.82)],
    )

    agreement, signals = score_topic_agreement(response)
    assessment = assess_confidence(response)

    assert agreement == 1.0
    assert any("hybrid_agreement=1.00" in signal for signal in signals)
    assert assessment.decision == "sufficient_evidence"


def test_hybrid_both_disagreement_lowers_confidence():
    response = make_response(
        route="hybrid_both",
        papers=[make_paper("p1", topic="RAG", score=0.7), make_paper("p2", topic="RAG", score=0.68)],
        chunks=[make_chunk("c1", paper_id="p3", topic="Fine-tuning (LoRA / PEFT)", score=0.7)],
    )

    components = confidence_components(response)

    assert components["agreement_score"] == 0.45
    assert any("do not overlap" in signal for signal in components["agreement_signals"])


def test_off_topic_results_are_insufficient_even_with_high_scores():
    response = make_response(
        query="quantum cryptography hardware",
        papers=[make_paper("p1", score=0.95), make_paper("p2", score=0.9)],
    )

    assessment = assess_confidence(response)

    assert assessment.decision == "insufficient_evidence"
    assert any("query_support_score=0.00" in signal for signal in assessment.signals)


def test_underspecified_query_asks_for_clarification():
    response = make_response(
        query="Tell me about it.",
        papers=[make_paper("p1", score=0.95), make_paper("p2", score=0.9)],
    )

    assessment = assess_confidence(response)

    assert assessment.decision == "ask_clarifying_question"
    assert any("query_specificity=0" in signal for signal in assessment.signals)


def test_load_response_reads_unified_response_json(tmp_path):
    response = make_response(papers=[make_paper("p1")])
    path = tmp_path / "response.json"
    path.write_text(response.model_dump_json())

    loaded = load_response(path)

    assert loaded.paper_results[0].paper_id == "p1"


def test_confidence_cli_accepts_input_file(tmp_path):
    response = make_response(papers=[make_paper("p1"), make_paper("p2")])
    path = tmp_path / "response.json"
    path.write_text(response.model_dump_json())

    completed = subprocess.run(
        [sys.executable, "-m", "retrieval.confidence", "--input", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["decision"] in {"sufficient_evidence", "broaden_search"}
    assert payload["result_count"] == 2
