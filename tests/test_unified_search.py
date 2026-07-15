import json
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from retrieval.unified_search import (
    UnifiedSearchError,
    metadata_filter_papers,
    qdrant_chunk_point_to_candidate,
    retrieve_chunks,
    run_unified_search,
)
from shared.schemas import QueryRoute, UnifiedSearchRequest


def route(name):
    return QueryRoute(
        query="test query",
        route=name,
        reason=f"route to {name}",
        confidence=0.9,
        matched_signals=[name],
    )


def fake_router(name):
    def _router(query):
        decision = route(name)
        return decision.model_copy(update={"query": query})

    return _router


def fake_paper_retriever(query, **kwargs):
    assert query
    assert kwargs["collection_name"] == "research_papers"
    assert kwargs["dense_top_k"] == 4
    assert kwargs["sparse_top_k"] == 5
    assert kwargs["final_top_k"] == 2
    return [
        {
            "paper_id": "paper-1",
            "title": "Hallucination Survey",
            "topic": "LLM Evaluation & Hallucination Detection",
            "citation_count": 100,
            "abstract": "Survey of hallucination reduction methods.",
            "hybrid_score": 0.9,
            "matched_by": ["dense", "sparse"],
        },
        {
            "paper_id": "paper-2",
            "title": "Grounded Generation",
            "topic": "Retrieval-Augmented Generation (RAG)",
            "citation_count": 50,
            "abstract": "Uses retrieval for grounding.",
            "hybrid_score": 0.7,
            "matched_by": ["dense"],
        },
    ]


def fake_chunk_retriever(query, **kwargs):
    assert query
    assert kwargs["collection_name"] == "research_paper_chunks"
    assert kwargs["top_k"] == 2
    return [
        {
            "chunk_id": "chunk-1",
            "paper_id": "paper-3",
            "title": "Hallucination Benchmarks",
            "topic": "LLM Evaluation & Hallucination Detection",
            "citation_count": 75,
            "section_hint": "experiments",
            "text": "TruthfulQA and HaluEval are used as benchmarks.",
            "dense_score": 0.82,
            "matched_by": ["chunk_dense"],
        }
    ]


def fake_reranker(query, candidates, top_k):
    enriched = []
    for index, candidate in enumerate(candidates):
        score = round(1.0 - (index * 0.1), 6)
        enriched.append(
            {
                **candidate,
                "rerank_raw_score": score,
                "rerank_score": score,
                "citation_score": 0.5,
                "blended_score": round((0.75 * score) + 0.125, 6),
                "score_breakdown": {"rerank_weight": 0.75, "citation_weight": 0.25},
            }
        )
    return enriched[:top_k]


def make_bm25_artifact():
    return {
        "papers": [
            {
                "paper_id": "p1",
                "title": "Older LoRA Paper",
                "topic": "Fine-tuning (LoRA / PEFT)",
                "year": 2020,
                "citation_count": 500,
                "metadata": {"abstract": "Old LoRA work."},
            },
            {
                "paper_id": "p2",
                "title": "Recent LoRA Paper",
                "topic": "Fine-tuning (LoRA / PEFT)",
                "year": 2023,
                "citation_count": 300,
                "metadata": {"abstract": "Recent LoRA work."},
            },
            {
                "paper_id": "p3",
                "title": "Recent RAG Paper",
                "topic": "Retrieval-Augmented Generation (RAG)",
                "year": 2024,
                "citation_count": 999,
                "metadata": {"abstract": "Recent RAG work."},
            },
        ]
    }


def test_unified_search_request_strips_query():
    request = UnifiedSearchRequest(query="  hallucination datasets  ")

    assert request.query == "hallucination datasets"


def test_qdrant_chunk_point_to_candidate_flattens_payload():
    point = SimpleNamespace(
        score=0.8,
        payload={
            "chunk_id": "chunk-1",
            "paper_id": "paper-1",
            "title": "A Paper",
            "topic": "RAG",
            "text": "Chunk text",
        },
    )

    candidate = qdrant_chunk_point_to_candidate(point)

    assert candidate["chunk_id"] == "chunk-1"
    assert candidate["text"] == "Chunk text"
    assert candidate["dense_score"] == 0.8
    assert candidate["matched_by"] == ["chunk_dense"]


def test_retrieve_chunks_embeds_query_and_searches_chunk_collection():
    class FakeEmbeddings:
        def create(self, model, input):
            assert model == "text-embedding-3-large"
            assert input == "hallucination benchmarks"
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 3072)])

    class FakeOpenAI:
        embeddings = FakeEmbeddings()

    class FakeQdrant:
        def query_points(self, collection_name, query, limit, with_payload):
            assert collection_name == "research_paper_chunks"
            assert len(query) == 1024
            assert limit == 1
            assert with_payload is True
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        score=0.7,
                        payload={
                            "chunk_id": "chunk-1",
                            "title": "Benchmark Paper",
                            "topic": "LLM Evaluation",
                            "text": "Benchmark text",
                        },
                    )
                ]
            )

    results = retrieve_chunks(
        "hallucination benchmarks",
        openai_client=FakeOpenAI(),
        qdrant_client=FakeQdrant(),
        top_k=1,
    )

    assert results[0]["chunk_id"] == "chunk-1"


def test_run_unified_search_paper_level_returns_only_papers():
    response = run_unified_search(
        "What are the main approaches?",
        top_k=2,
        dense_top_k=4,
        sparse_top_k=5,
        router=fake_router("paper_level"),
        paper_retriever=fake_paper_retriever,
        chunk_retriever=fake_chunk_retriever,
        reranker=fake_reranker,
        openai_client=object(),
        qdrant_client=object(),
        bm25_artifact={},
    )

    assert response.route.route == "paper_level"
    assert response.paper_result_count == 2
    assert response.chunk_result_count == 0
    assert response.paper_results[0].blended_score is not None


def test_run_unified_search_chunk_level_returns_only_chunks():
    response = run_unified_search(
        "Which datasets and metrics are used?",
        top_k=2,
        router=fake_router("chunk_level"),
        paper_retriever=fake_paper_retriever,
        chunk_retriever=fake_chunk_retriever,
        reranker=fake_reranker,
        openai_client=object(),
        qdrant_client=object(),
        bm25_artifact={},
    )

    assert response.route.route == "chunk_level"
    assert response.paper_result_count == 0
    assert response.chunk_result_count == 1
    assert response.chunk_results[0].text.startswith("TruthfulQA")
    assert response.chunk_results[0].blended_score is not None


def test_run_unified_search_hybrid_both_keeps_result_sets_separate():
    response = run_unified_search(
        "Compare RAG and self-verification methods.",
        top_k=2,
        dense_top_k=4,
        sparse_top_k=5,
        router=fake_router("hybrid_both"),
        paper_retriever=fake_paper_retriever,
        chunk_retriever=fake_chunk_retriever,
        reranker=fake_reranker,
        openai_client=object(),
        qdrant_client=object(),
        bm25_artifact={},
    )

    assert response.route.route == "hybrid_both"
    assert response.paper_result_count == 2
    assert response.chunk_result_count == 1
    assert response.paper_results[0].paper_id == "paper-1"
    assert response.chunk_results[0].chunk_id == "chunk-1"


def test_metadata_filter_papers_filters_topic_and_year_then_sorts_by_citations():
    results = metadata_filter_papers("Show top-cited LoRA papers after 2020", make_bm25_artifact(), top_k=5)

    assert [paper["paper_id"] for paper in results] == ["p2"]
    assert results[0]["matched_by"] == ["metadata_filter"]


def test_run_unified_search_metadata_filter_does_not_require_vector_clients():
    response = run_unified_search(
        "Show top-cited LoRA papers after 2020",
        top_k=5,
        router=fake_router("metadata_filter"),
        bm25_artifact=make_bm25_artifact(),
        apply_reranking=False,
    )

    assert response.route.route == "metadata_filter"
    assert response.paper_result_count == 1
    assert response.paper_results[0].paper_id == "p2"
    assert response.chunk_result_count == 0


def test_run_unified_search_rejects_blank_query():
    with pytest.raises(UnifiedSearchError, match="query"):
        run_unified_search(" ", router=fake_router("paper_level"), openai_client=object(), qdrant_client=object(), bm25_artifact={})


def test_unified_search_cli_metadata_filter_outputs_json(tmp_path):
    artifact_path = tmp_path / "bm25.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(make_bm25_artifact(), handle)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "retrieval.unified_search",
            "Show top-cited LoRA papers after 2020",
            "--bm25-index",
            str(artifact_path),
            "--no-rerank",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["route"]["route"] == "metadata_filter"
    assert payload["paper_result_count"] == 1
    assert payload["paper_results"][0]["paper_id"] == "p2"
