from types import SimpleNamespace
from agentic.graph import run_agentic_research
def fake_response():
    chunk = SimpleNamespace(chunk_id="c1", text="evidence", page_start=2, page_end=3)
    paper = SimpleNamespace(paper_id="p1", title="Paper", chunks=[chunk])
    return SimpleNamespace(papers=[paper], model_dump=lambda mode="json": {"papers": [{"paper_id": "p1"}]})
def test_local_path_completes():
    state = run_agentic_research("Explain the method", searcher=lambda query, top_k: fake_response())
    assert state["status"] == "completed"
    assert state["route"] == "corpus"
    assert state["tool_calls"][0]["tool"] == "search_local_corpus"
    assert state["evidence"][1]["page_start"] == 2
def test_live_path_is_explicitly_pending():
    state = run_agentic_research("What are the latest papers?")
    assert state["status"] == "pending_external_tools"
    assert "external" in state["warnings"][0].lower()
def test_oversized_query_is_blocked():
    state = run_agentic_research("x" * 2001)
    assert state["status"] == "blocked"
    assert "2000" in state["error"]
