import pytest
from agentic.dispatch import deduplicate_papers, run_external_search
from agentic.external import ExternalPaper, ExternalSearchClient, ExternalSearchError
class FakeResponse:
    def __init__(self, status_code=200, *, text="", data=None):
        self.status_code = status_code
        self.text = text
        self._data = data
    def json(self):
        if isinstance(self._data, Exception): raise self._data
        return self._data
class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)
def test_arxiv_xml_is_normalized():
    xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/1234.1</id><title> A paper </title><summary> An abstract. </summary><published>2025-01-01</published><author><name>Ada</name></author></entry></feed>'
    client = ExternalSearchClient(FakeSession([FakeResponse(text=xml)]), max_retries=0)
    paper = client.search_arxiv("attention")[0]
    assert paper.source == "arxiv"
    assert paper.paper_id == "1234.1"
    assert paper.title == "A paper"
    assert paper.authors == ("Ada",)
def test_semantic_scholar_uses_api_key_and_normalizes_json():
    session = FakeSession([FakeResponse(data={"data": [{"paperId": "s1", "title": "Paper", "authors": [{"name": "A"}], "citationCount": 4}]})])
    client = ExternalSearchClient(session, max_retries=0, semantic_scholar_api_key="secret")
    paper = client.search_semantic_scholar("rag")[0]
    assert paper.paper_id == "s1"
    assert paper.citation_count == 4
    assert session.calls[0][2]["headers"]["x-api-key"] == "secret"
def test_tavily_requires_key():
    client = ExternalSearchClient(FakeSession([]), max_retries=0, tavily_api_key="")
    with pytest.raises(ExternalSearchError, match="TAVILY_API_KEY"):
        client.search_tavily("rag")
def test_retry_then_success():
    session = FakeSession([FakeResponse(status_code=429), FakeResponse(data={"data": []})])
    client = ExternalSearchClient(session, max_retries=1, backoff_seconds=0)
    assert client.search_semantic_scholar("rag") == []
    assert len(session.calls) == 2
def test_dispatch_is_fail_soft_and_deduplicates():
    class FakeClient:
        def search_arxiv(self, query, max_results): return [ExternalPaper("arxiv", "1", "Same", url="https://x")]
        def search_semantic_scholar(self, query, max_results): raise ExternalSearchError("down")
        def search_tavily(self, query, max_results): return [ExternalPaper("tavily", "https://x", "Same", url="https://x", abstract="longer")]
    response = run_external_search("rag", client=FakeClient())
    assert len(response.results) == 1
    assert "arxiv" in response.results[0]["source"] and "tavily" in response.results[0]["source"]
    assert response.warnings and "semantic_scholar" in response.warnings[0]
