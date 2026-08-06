from types import SimpleNamespace

import pytest

from retrieval.hybrid_search import (
    bm25_result_to_candidate,
    candidate_key,
    merge_candidates,
    merge_candidates_rrf,
    retrieve_papers,
)


class FakeEmbeddings:
    def create(self, model, input, dimensions):
        assert model == "text-embedding-3-large"
        assert input == "hallucination detection"
        assert dimensions == 1024
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1024)])


class FakeOpenAIClient:
    embeddings = FakeEmbeddings()


class FakeQdrantClient:
    def query_points(self, collection_name, query, limit, with_payload):
        assert collection_name == "research_papers"
        assert len(query) == 1024
        assert limit == 2
        assert with_payload is True
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.8,
                    payload={
                        "paper_id": "paper-1",
                        "title": "Hallucination Detection for LLMs",
                        "topic": "LLM Evaluation & Hallucination Detection",
                        "year": 2024,
                        "citation_count": 40,
                        "abstract": "Detects hallucinations in generated answers.",
                        "main_contribution": "Introduces a hallucination detector.",
                    },
                )
            ]
        )


class FakeBm25:
    def get_scores(self, query_tokens):
        assert "hallucination" in query_tokens
        return [3.0, 1.0]


def make_bm25_artifact():
    return {
        "bm25": FakeBm25(),
        "papers": [
            {
                "paper_id": "paper-1",
                "title": "Hallucination Detection for LLMs",
                "topic": "LLM Evaluation & Hallucination Detection",
                "year": 2024,
                "citation_count": 40,
                "metadata": {"abstract": "Detects hallucinations in generated answers."},
            },
            {
                "paper_id": "paper-2",
                "title": "Retrieval Grounding for Language Models",
                "topic": "Retrieval-Augmented Generation (RAG)",
                "year": 2023,
                "citation_count": 20,
                "metadata": {"abstract": "Uses retrieval to ground answers."},
            },
        ],
    }


def test_bm25_result_to_candidate_flattens_metadata():
    candidate = bm25_result_to_candidate(
        {
            "paper_id": "paper-1",
            "title": "A paper",
            "topic": "RAG",
            "score": 2.5,
            "metadata": {"abstract": "Useful abstract", "main_contribution": "Introduces a method."},
        }
    )

    assert candidate["title"] == "A paper"
    assert candidate["abstract"] == "Useful abstract"
    assert candidate["main_contribution"] == "Introduces a method."
    assert candidate["sparse_score"] == 2.5


def test_candidate_key_prefers_paper_id_and_falls_back_to_normalized_title():
    assert candidate_key({"paper_id": "abc", "title": "A Paper"}) == "id:abc"
    assert candidate_key({"paper_id": None, "title": "A Paper!"}) == "title:a paper"


def test_merge_candidates_combines_dense_and_sparse_matches():
    dense = [
        {"paper_id": "paper-1", "title": "A", "dense_score": 0.8, "sparse_score": None, "citation_count": 10},
        {"paper_id": "paper-2", "title": "B", "dense_score": 0.4, "sparse_score": None, "citation_count": 5},
    ]
    sparse = [
        {"paper_id": "paper-1", "title": "A", "dense_score": None, "sparse_score": 3.0, "citation_count": 10},
        {"paper_id": "paper-3", "title": "C", "dense_score": None, "sparse_score": 2.0, "citation_count": 20},
    ]

    results = merge_candidates(dense, sparse, final_top_k=3)

    assert results[0]["paper_id"] == "paper-1"
    assert results[0]["matched_by"] == ["dense", "sparse"]
    assert results[0]["hybrid_score"] == 1.0
    assert len(results) == 3


def test_merge_candidates_defaults_to_weighted_fusion():
    dense = [{"paper_id": "paper-1", "title": "A", "dense_score": 0.8, "sparse_score": None, "citation_count": 10}]
    sparse = [{"paper_id": "paper-1", "title": "A", "dense_score": None, "sparse_score": 3.0, "citation_count": 10}]

    results = merge_candidates(dense, sparse, final_top_k=1)

    assert results[0]["fusion_method"] == "weighted"
    assert results[0]["hybrid_score"] == 1.0


def test_merge_candidates_rrf_rewards_top_ranked_agreement():
    # paper-1 is rank 1 in both lists; paper-2 is rank 1 dense only; paper-3 is rank 2 sparse only.
    dense = [
        {"paper_id": "paper-1", "title": "A", "dense_score": 0.9, "citation_count": 1},
        {"paper_id": "paper-2", "title": "B", "dense_score": 0.85, "citation_count": 1},
    ]
    sparse = [
        {"paper_id": "paper-1", "title": "A", "sparse_score": 5.0, "citation_count": 1},
        {"paper_id": "paper-3", "title": "C", "sparse_score": 4.0, "citation_count": 1},
    ]

    results = merge_candidates_rrf(dense, sparse, final_top_k=3, rrf_k=60)

    assert results[0]["paper_id"] == "paper-1"
    assert results[0]["fusion_method"] == "rrf"
    # Rank-1-in-both should score exactly 1/61 + 1/61.
    assert results[0]["hybrid_score"] == round(1 / 61 + 1 / 61, 6)
    assert {result["paper_id"] for result in results} == {"paper-1", "paper-2", "paper-3"}


def test_merge_candidates_rrf_is_insensitive_to_score_scale():
    # Dense scores (~0-1) and sparse scores (BM25, unbounded) live on very
    # different scales; RRF should not let a huge BM25 score dominate purely
    # because of scale, only because of rank.
    dense = [{"paper_id": "paper-1", "title": "A", "dense_score": 0.99, "citation_count": 1}]
    sparse = [{"paper_id": "paper-2", "title": "B", "sparse_score": 500.0, "citation_count": 1}]

    results = merge_candidates_rrf(dense, sparse, final_top_k=2)

    # Both are rank-1 in their own list, so they should tie under RRF.
    assert results[0]["hybrid_score"] == results[1]["hybrid_score"]


def test_merge_candidates_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError, match="rrf_k"):
        merge_candidates_rrf([], [], rrf_k=0)


def test_retrieve_papers_supports_rrf_fusion_method():
    results = retrieve_papers(
        "hallucination detection",
        openai_client=FakeOpenAIClient(),
        qdrant_client=FakeQdrantClient(),
        bm25_artifact=make_bm25_artifact(),
        dense_top_k=2,
        sparse_top_k=2,
        final_top_k=2,
        fusion_method="rrf",
    )

    assert results[0]["paper_id"] == "paper-1"
    assert results[0]["fusion_method"] == "rrf"


def test_retrieve_papers_runs_dense_and_sparse_paths_without_real_api():
    results = retrieve_papers(
        "hallucination detection",
        openai_client=FakeOpenAIClient(),
        qdrant_client=FakeQdrantClient(),
        bm25_artifact=make_bm25_artifact(),
        dense_top_k=2,
        sparse_top_k=2,
        final_top_k=2,
    )

    assert results[0]["paper_id"] == "paper-1"
    assert results[0]["matched_by"] == ["dense", "sparse"]
    assert results[0]["dense_score"] == 0.8
    assert results[0]["sparse_score"] == 3.0


def test_retrieve_papers_rejects_empty_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        retrieve_papers(
            " ",
            openai_client=FakeOpenAIClient(),
            qdrant_client=FakeQdrantClient(),
            bm25_artifact=make_bm25_artifact(),
        )
