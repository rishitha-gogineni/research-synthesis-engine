import json
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from retrieval.unified_search import (
    UnifiedSearchError,
    expand_query_for_retrieval,
    maybe_promote,
    metadata_filter_papers,
    qdrant_chunk_point_to_candidate,
    rerank_candidate_pool_size,
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
    assert kwargs["final_top_k"] == 4
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
    assert kwargs["top_k"] == 4
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
        def create(self, model, input, dimensions):
            assert model == "text-embedding-3-large"
            assert input == "hallucination benchmarks"
            assert dimensions == 1024
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1024)])

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


def test_run_unified_search_forwards_fusion_method_to_paper_retriever():
    seen_fusion_methods = []

    def capturing_paper_retriever(query, **kwargs):
        seen_fusion_methods.append(kwargs.get("fusion_method"))
        return fake_paper_retriever(query, **kwargs)

    response = run_unified_search(
        "What are the main approaches?",
        top_k=2,
        dense_top_k=4,
        sparse_top_k=5,
        fusion_method="rrf",
        router=fake_router("paper_level"),
        paper_retriever=capturing_paper_retriever,
        chunk_retriever=fake_chunk_retriever,
        reranker=fake_reranker,
        openai_client=object(),
        qdrant_client=object(),
        bm25_artifact={},
    )

    assert seen_fusion_methods == ["rrf"]
    assert response.route.route == "paper_level"


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

def test_expand_query_for_retrieval_maps_colloquial_hallucination_language():
    expanded = expand_query_for_retrieval("How can I stop a chatbot from making things up?")

    assert "making things up" in expanded
    assert "hallucination detection mitigation" in expanded


def test_run_unified_search_uses_expanded_query_for_retrievers_but_keeps_visible_query():
    seen = []

    def paper_retriever(query, **kwargs):
        seen.append(query)
        return []

    def chunk_retriever(query, **kwargs):
        seen.append(query)
        return []

    response = run_unified_search(
        "How can I stop a chatbot from making things up?",
        top_k=2,
        paper_retriever=paper_retriever,
        chunk_retriever=chunk_retriever,
        router=fake_router("hybrid_both"),
        openai_client=object(),
        qdrant_client=object(),
        bm25_artifact=make_bm25_artifact(),
        apply_reranking=False,
    )

    assert response.query == "How can I stop a chatbot from making things up?"
    assert seen
    assert all("hallucination detection mitigation" in query for query in seen)


def test_extended_expansions_are_opt_in():
    query = "Which datasets are used to evaluate hallucination detection methods?"

    assert expand_query_for_retrieval(query) == query
    assert "dataset benchmark corpus" in expand_query_for_retrieval(query, extended=True)


def test_extended_expansions_do_not_replace_the_base_table():
    query = "How can I stop a chatbot from making things up?"
    expanded = expand_query_for_retrieval(query, extended=True)

    assert "hallucination detection mitigation" in expanded


def test_rerank_candidate_pool_size_defaults_to_no_oversampling():
    assert rerank_candidate_pool_size(10, True) == 10
    assert rerank_candidate_pool_size(10, False) == 10
    assert rerank_candidate_pool_size(10, True, multiplier=3) == 30

    with pytest.raises(ValueError):
        rerank_candidate_pool_size(10, True, multiplier=0)


def test_maybe_promote_disabled_is_plain_truncation():
    candidates = [{"chunk_id": f"c{index}", "paper_id": f"p{index}"} for index in range(1, 6)]
    reduced = maybe_promote("q", candidates, enabled=False, top_k=3, level="chunk")

    assert [candidate["chunk_id"] for candidate in reduced] == ["c1", "c2", "c3"]
    assert "promotion_score" not in reduced[0]


def test_maybe_promote_enabled_annotates_and_reduces():
    candidates = [
        {"chunk_id": f"c{index}", "paper_id": f"p{index}", "section_hint": "results"}
        for index in range(1, 6)
    ]
    reduced = maybe_promote("What results are reported?", candidates, enabled=True, top_k=3, level="chunk")

    assert len(reduced) == 3
    assert "promotion_score" in reduced[0]


def test_run_unified_search_promotion_is_on_by_default():
    requested = {}

    def chunk_retriever(query, **kwargs):
        requested["top_k"] = kwargs["top_k"]
        return [
            {"chunk_id": f"c{index}", "paper_id": f"p{index}", "section_hint": "limitations"}
            for index in range(1, 7)
        ]

    response = run_unified_search(
        "What are the limitations of retrieval augmented generation?",
        top_k=3,
        paper_retriever=lambda query, **kwargs: [],
        chunk_retriever=chunk_retriever,
        router=fake_router("chunk_level"),
        openai_client=object(),
        qdrant_client=object(),
        bm25_artifact=make_bm25_artifact(),
        apply_reranking=False,
    )

    assert requested["top_k"] == 6
    assert len(response.chunk_results) == 3
    assert [chunk.chunk_id for chunk in response.chunk_results] == ["c1", "c2", "c3"]


def test_run_unified_search_pool_multiplier_widens_the_internal_request():
    requested = {}

    def chunk_retriever(query, **kwargs):
        requested["top_k"] = kwargs["top_k"]
        return [{"chunk_id": f"c{index}", "paper_id": f"p{index}"} for index in range(1, 21)]

    response = run_unified_search(
        "What are the limitations of retrieval augmented generation?",
        top_k=5,
        paper_retriever=lambda query, **kwargs: [],
        chunk_retriever=chunk_retriever,
        router=fake_router("chunk_level"),
        openai_client=object(),
        qdrant_client=object(),
        bm25_artifact=make_bm25_artifact(),
        apply_reranking=False,
        pool_multiplier=3,
        apply_promotion=True,
    )

    assert requested["top_k"] == 15
    # The wider pool is internal only; the caller still sees top_k results.
    assert len(response.chunk_results) == 5


def test_run_unified_search_promotion_uses_the_original_query_not_the_expansion():
    """Expansion text contains intent words the user never typed."""

    def chunk_retriever(query, **kwargs):
        return [
            {"chunk_id": "c1", "paper_id": "p1", "section_hint": "introduction"},
            {"chunk_id": "c2", "paper_id": "p2", "section_hint": "experiments"},
        ]

    response = run_unified_search(
        "How can I stop a chatbot from making things up?",
        top_k=2,
        paper_retriever=lambda query, **kwargs: [],
        chunk_retriever=chunk_retriever,
        router=fake_router("chunk_level"),
        openai_client=object(),
        qdrant_client=object(),
        bm25_artifact=make_bm25_artifact(),
        apply_reranking=False,
        apply_promotion=True,
        extended_expansions=True,
    )

    # The user asked nothing about datasets or experiments, so the experiments
    # chunk must not be promoted over the rank-1 result.
    assert [chunk.chunk_id for chunk in response.chunk_results] == ["c1", "c2"]


def test_run_unified_search_metadata_route_ignores_pool_widening():
    """The metadata/ranked-list path must stay on its existing top_k behavior."""

    response = run_unified_search(
        "Most cited papers after 2020",
        top_k=1,
        router=fake_router("metadata_filter"),
        bm25_artifact=make_bm25_artifact(),
        apply_reranking=False,
        pool_multiplier=5,
        apply_promotion=True,
    )

    assert response.paper_result_count == 1
    assert "promotion_score" not in response.paper_results[0].model_dump()
