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
