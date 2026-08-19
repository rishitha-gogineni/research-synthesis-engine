from types import SimpleNamespace

from retrieval.filters import RetrievalFilters
from retrieval.hybrid_search import search_dense
from retrieval.unified_search import search_chunks


def test_retrieval_filters_match_topic_and_year():
    filters = RetrievalFilters.from_values(topics=["RAG"], year_min=2020, year_max=2024)

    assert filters.matches({"topic": "RAG", "year": 2022})
    assert not filters.matches({"topic": "RAG", "year": 2019})
    assert not filters.matches({"topic": "Agents", "year": 2022})


def test_qdrant_search_forwards_query_filter_only_when_active():
    class FakeQdrant:
        def __init__(self):
            self.kwargs = None

        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(points=[])

    client = FakeQdrant()
    filters = RetrievalFilters.from_values(year_min=2020)
    search_dense(client, "research_papers", [0.1], top_k=2, retrieval_filters=filters)

    assert "query_filter" in client.kwargs
    assert client.kwargs["query_filter"].must[0].key == "year"


def test_chunk_search_forwards_query_filter():
    class FakeQdrant:
        def __init__(self):
            self.kwargs = None

        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(points=[])

    client = FakeQdrant()
    filters = RetrievalFilters.from_values(topics=["RAG"])
    search_chunks(client, "research_paper_chunks", [0.1], top_k=2, retrieval_filters=filters)

    assert "query_filter" in client.kwargs


def test_factual_chunk_queries_use_rank_fusion_to_keep_sparse_exact_matches():
    from retrieval.chunk_bm25 import chunk_query_prefers_sparse, merge_chunk_candidates

    dense = [
        {"chunk_id": "dense-1", "paper_id": "p1", "dense_score": 0.9, "matched_by": ["chunk_dense"]},
        {"chunk_id": "dense-2", "paper_id": "p2", "dense_score": 0.8, "matched_by": ["chunk_dense"]},
    ]
    sparse = [
        {"chunk_id": "exact", "paper_id": "p3", "sparse_score": 12.0, "matched_by": ["chunk_sparse"]},
    ]

    assert chunk_query_prefers_sparse("What accuracy does PRAG achieve?")
    results = merge_chunk_candidates(dense, sparse, top_k=2, fusion_method="rrf")

    assert "exact" in [item["chunk_id"] for item in results]
