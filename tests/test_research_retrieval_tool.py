from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.schemas import RetrievalRequest, RetrievalResponse
from tools.research_retrieval import (
    RetrievalToolError,
    build_retrieval_response,
    candidate_to_retrieved_paper,
    run_research_retrieval,
)


def fake_retriever(query, **kwargs):
    assert query == "What reduces hallucinations?"
    assert kwargs["collection_name"] == "research_papers"
    assert kwargs["dense_top_k"] == 4
    assert kwargs["sparse_top_k"] == 5
    assert kwargs["final_top_k"] == 2
    return [
        {
            "paper_id": "paper-1",
            "title": "Hallucination Detection for LLMs",
            "topic": "LLM Evaluation & Hallucination Detection",
            "year": 2024,
            "citation_count": 100,
            "authors": ["Ada Lovelace"],
            "abstract": "A paper about hallucination detection.",
            "main_contribution": "Detects hallucinations.",
            "methodology": "Benchmark evaluation.",
            "dataset_used": "QA benchmark",
            "key_result": "Improves detection quality.",
            "limitations": "not stated in abstract",
            "dense_score": 0.8,
            "sparse_score": 3.0,
            "hybrid_score": 0.95,
            "matched_by": ["dense", "sparse"],
        }
    ]


def test_retrieval_request_strips_and_validates_query():
    request = RetrievalRequest(query="  hallucination detection  ", top_k=3)

    assert request.query == "hallucination detection"
    assert request.top_k == 3


def test_retrieval_request_rejects_blank_query():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="   ")


def test_candidate_to_retrieved_paper_uses_schema_defaults():
    paper = candidate_to_retrieved_paper({"title": "A Paper", "topic": "RAG"})

    assert paper.title == "A Paper"
    assert paper.topic == "RAG"
    assert paper.citation_count == 0
    assert paper.matched_by == []


def test_build_retrieval_response_returns_schema_compliant_output():
    request = RetrievalRequest(query="hallucination", top_k=1)
    response = build_retrieval_response(
        request,
        [{"title": "A Paper", "topic": "LLM Evaluation", "hybrid_score": 0.7}],
    )

    assert isinstance(response, RetrievalResponse)
    assert response.query == "hallucination"
    assert response.result_count == 1
    assert response.results[0].hybrid_score == 0.7


def test_run_research_retrieval_uses_injected_retriever_without_live_clients():
    response = run_research_retrieval(
        "What reduces hallucinations?",
        top_k=2,
        dense_top_k=4,
        sparse_top_k=5,
        retriever=fake_retriever,
        openai_client=object(),
        qdrant_client=object(),
        bm25_artifact={},
    )

    assert response.result_count == 1
    assert response.results[0].title == "Hallucination Detection for LLMs"
    assert response.results[0].matched_by == ["dense", "sparse"]


def test_run_research_retrieval_reports_validation_errors_cleanly():
    with pytest.raises(RetrievalToolError, match="query"):
        run_research_retrieval(
            " ",
            retriever=fake_retriever,
            openai_client=object(),
            qdrant_client=object(),
            bm25_artifact={},
        )


def test_run_research_retrieval_checks_missing_bm25_index_before_live_call(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    missing_index = tmp_path / "missing.pkl"

    with pytest.raises(RetrievalToolError, match="BM25 index not found"):
        run_research_retrieval("hallucination", bm25_index=missing_index, env_file=Path("missing.env"))
