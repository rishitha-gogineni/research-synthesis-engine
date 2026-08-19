from types import SimpleNamespace
import pytest
from agentic.api import AgenticResearchRequest, agentic_research
def test_agentic_request_normalizes_question():
    request = AgenticResearchRequest(question="  latest papers  ", top_k=4)
    assert request.question == "latest papers"
def test_agentic_endpoint_returns_trace(monkeypatch):
    fake_state = {"query": "latest papers", "status": "completed", "route": "live", "route_reason": "current", "route_confidence": 0.9, "planned_tools": ["search_arxiv"], "tool_calls": [{"tool": "search_arxiv"}], "evidence": [{"kind": "external", "title": "Paper"}], "warnings": []}
    monkeypatch.setattr("agentic.api.run_agentic_research", lambda query, top_k: fake_state)
    request = SimpleNamespace(state=SimpleNamespace(request_id="request-123"))
    response = agentic_research(AgenticResearchRequest(question="latest papers"), request)
    assert response.request_id == "request-123"
    assert response.route == "live"
    assert response.tool_calls[0]["tool"] == "search_arxiv"
def test_agentic_endpoint_is_registered():
    from api.main import app
    assert "/agentic/research" in app.openapi()["paths"]
