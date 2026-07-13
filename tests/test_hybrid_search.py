from types import SimpleNamespace

import pytest

from retrieval.hybrid_search import (
    bm25_result_to_candidate,
    candidate_key,
    merge_candidates,
    retrieve_papers,
)


class FakeEmbeddings:
    def create(self, model, input):
        assert model == "text-embedding-3-large"
        assert input == "hallucination detection"
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 3072)])


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
