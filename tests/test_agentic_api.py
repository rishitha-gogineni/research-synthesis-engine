from types import SimpleNamespace
from agentic.api import AgenticResearchRequest, agentic_research
from agentic.llm import LLMResult
def test_agentic_request_normalizes_question():
    request = AgenticResearchRequest(question="  latest papers  ", top_k=4)
    assert request.question == "latest papers"
def test_agentic_endpoint_returns_trace(monkeypatch):
    fake_state = {"query": "latest papers", "status": "completed", "route": "live", "route_reason": "current", "route_confidence": 0.9, "planned_tools": ["search_arxiv"], "tool_calls": [{"tool": "search_arxiv"}], "evidence": [{"kind": "external", "title": "Paper"}], "warnings": []}
    monkeypatch.setattr("agentic.api.run_agentic_research", lambda query, top_k: fake_state)
    monkeypatch.setattr("agentic.api.run_grounded_answer", lambda query, evidence, max_tool_calls: LLMResult("Grounded answer [source_1]", ["source_1"], [{"tool": "search_arxiv", "status": "completed"}], [], {"total_tokens": 12}, 4.0))
    request = SimpleNamespace(state=SimpleNamespace(request_id="request-123"))
    response = agentic_research(AgenticResearchRequest(question="latest papers"), request)
    assert response.request_id == "request-123"
    assert response.route == "live"
    assert response.answer == "Grounded answer [source_1]"
    assert response.citations == ["source_1"]
    assert response.llm_usage["total_tokens"] == 12
def test_agentic_endpoint_skips_low_confidence_synthesis(monkeypatch):
    fake_state = {"query": "question", "status": "completed", "route": "corpus", "confidence_decision": "insufficient_evidence", "warnings": []}
    monkeypatch.setattr("agentic.api.run_agentic_research", lambda query, top_k: fake_state)
    monkeypatch.setattr("agentic.api.run_grounded_answer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not synthesize")))
    request = SimpleNamespace(state=SimpleNamespace(request_id="request-456"))
    response = agentic_research(AgenticResearchRequest(question="question"), request)
    assert response.answer is None
    assert any("skipped" in warning for warning in response.warnings)

def test_agentic_endpoint_is_registered():
    from api.main import app
    assert "/agentic/research" in app.openapi()["paths"]
