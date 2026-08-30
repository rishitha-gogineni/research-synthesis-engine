"""Tests for agentic/dispatch.py — external search dispatch and dedup."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic.dispatch import (
    ExternalSearchResponse,
    deduplicate_papers,
    run_external_search,
)
from agentic.external import ExternalPaper, ExternalSearchError


def _paper(title="Test", url="http://example.com", source="arxiv", abstract="abs"):
    return ExternalPaper(
        title=title, abstract=abstract, url=url, source=source,
        authors=[], published_date="2024", citation_count=0,
        paper_id="", relevance_score=0.5,
    )


class TestDeduplicatePapers:
    def test_removes_duplicates_by_url(self):
        papers = [_paper(title="A", url="http://a.com"), _paper(title="A copy", url="http://a.com")]
        result = deduplicate_papers(papers)
        assert len(result) == 1

    def test_keeps_distinct(self):
        papers = [_paper(title="A", url="http://a.com"), _paper(title="B", url="http://b.com")]
        result = deduplicate_papers(papers)
        assert len(result) == 2

    def test_merges_sources(self):
        p1 = _paper(title="A", url="http://a.com", source="arxiv")
        p2 = _paper(title="A", url="http://a.com", source="semantic_scholar")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1
        assert "arxiv" in result[0]["source"]
        assert "semantic_scholar" in result[0]["source"]


class TestRunExternalSearch:
    def test_dispatches_to_sources(self):
        client = MagicMock()
        client.search_arxiv.return_value = [_paper(source="arxiv")]
        client.search_semantic_scholar.return_value = [_paper(title="B", url="http://b.com", source="s2")]

        resp = run_external_search("test query", sources=("arxiv", "semantic_scholar"), client=client)
        assert isinstance(resp, ExternalSearchResponse)
        assert "arxiv" in resp.sources
        assert "semantic_scholar" in resp.sources
        assert len(resp.results) == 2
        assert resp.warnings == []

    def test_handles_source_failure(self):
        client = MagicMock()
        client.search_arxiv.side_effect = ExternalSearchError("timeout")
        client.search_semantic_scholar.return_value = [_paper(source="s2")]

        resp = run_external_search("test", sources=("arxiv", "semantic_scholar"), client=client)
        assert "arxiv" not in resp.sources
        assert "semantic_scholar" in resp.sources
        assert any("arxiv" in w for w in resp.warnings)

    def test_unsupported_source(self):
        client = MagicMock(spec=[])
        resp = run_external_search("test", sources=("fake_source",), client=client)
        assert resp.sources == []
        assert any("unsupported" in w for w in resp.warnings)

    def test_empty_query_raises(self):
        import pytest
        with pytest.raises(ValueError, match="empty"):
            run_external_search("", client=MagicMock())
