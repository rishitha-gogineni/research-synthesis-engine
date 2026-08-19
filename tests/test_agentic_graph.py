from types import SimpleNamespace
from agentic.graph import run_agentic_research
def fake_response():
    chunk = SimpleNamespace(chunk_id="c1", text="evidence", page_start=2, page_end=3, paper_id="p1")
    paper = SimpleNamespace(paper_id="p1", title="Paper", chunks=[chunk])
    return SimpleNamespace(papers=[paper], model_dump=lambda mode="json": {"papers": [{"paper_id": "p1"}]})
def test_local_path_completes():
    state = run_agentic_research("Explain the method", searcher=lambda query, top_k: fake_response())
    assert state["status"] == "completed"
    assert state["route"] == "corpus"
    assert state["tool_calls"][0]["tool"] == "search_local_corpus"
    assert state["evidence"][1]["page_start"] == 2
def test_live_path_uses_external_dispatch():
    fake = lambda query, sources, max_results: {"results": [{"source": "arxiv", "title": "Paper"}], "sources": list(sources), "warnings": []}
    state = run_agentic_research("What are the latest papers?", external_searcher=fake)
    assert state["status"] == "completed"
    assert state["external_response"]["results"][0]["source"] == "arxiv"
    assert state["tool_calls"][0]["tool"].startswith("search_")
def test_oversized_query_is_blocked():
    state = run_agentic_research("x" * 2001)
    assert state["status"] == "blocked"
    assert "2000" in state["error"]


def test_hybrid_path_combines_local_and_external_evidence():
    from agentic.planner import RoutePlan

    def planner(query):
        return RoutePlan(
            "hybrid",
            ("search_local_corpus", "search_arxiv"),
            "test hybrid",
            0.9,
        )

    fake_external = lambda query, sources, max_results: {
        "results": [{"source": "arxiv", "title": "External paper"}],
        "sources": list(sources),
        "warnings": [],
    }
    state = run_agentic_research(
        "Compare the indexed and latest papers.",
        planner=planner,
        searcher=lambda query, top_k: fake_response(),
        external_searcher=fake_external,
    )

    assert state["status"] == "completed"
    assert [item["kind"] for item in state["evidence"]] == ["paper", "chunk", "external"]
    assert {item["tool"] for item in state["tool_calls"]} == {"search_local_corpus", "search_arxiv"}


def test_external_failure_is_reported_without_raising():
    def failing_external(query, sources, max_results):
        raise RuntimeError("provider timeout")

    state = run_agentic_research(
        "What are the latest papers?",
        external_searcher=failing_external,
    )

    assert state["status"] == "pending_external_tools"
    assert any("provider timeout" in warning for warning in state["warnings"])
