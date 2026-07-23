import pytest

from agent.query_rewriter import ChatTurn, QueryRewriteResult
from agent.research_graph import initial_state, run_research_agent
from shared.schemas import BriefTheme, ConfidenceAssessment, EvidenceSource, QueryRoute, ResearchBrief, RetrievedPaper, UnifiedSearchResponse


def make_response(query: str, *, score: float = 0.9) -> UnifiedSearchResponse:
    paper = RetrievedPaper(
        paper_id=f"paper-{len(query)}",
        title="Grounded Agent Survey",
        topic="AI Agents & Tool Use",
        year=2024,
        citation_count=100,
        abstract="Agents use planning and tools to execute tasks.",
        hybrid_score=score,
        rerank_score=score,
        blended_score=score,
        matched_by=["test"],
    )
    return UnifiedSearchResponse(
        query=query,
        route=QueryRoute(query=query, route="hybrid_both", reason="test route", confidence=0.9),
        paper_result_count=1,
        chunk_result_count=0,
        paper_results=[paper],
        chunk_results=[],
    )


def make_confidence(response: UnifiedSearchResponse, decision: str = "sufficient_evidence") -> ConfidenceAssessment:
    return ConfidenceAssessment(
        query=response.query,
        route=response.route.route,
        confidence_score=0.9 if decision == "sufficient_evidence" else 0.4,
        decision=decision,
        reason="test confidence reason",
        recommended_action="test recommended action",
        signals=["test_signal"],
        result_count=response.paper_result_count + response.chunk_result_count,
        top_score=0.9,
        route_confidence=0.9,
    )


def make_brief(response: UnifiedSearchResponse, confidence: ConfidenceAssessment) -> ResearchBrief:
    return ResearchBrief(
        query=response.query,
        status="generated" if confidence.decision == "sufficient_evidence" else "skipped_low_confidence",
        confidence_decision=confidence.decision,
        direct_answer=f"Answer for {response.query}",
        themes=[BriefTheme(theme="Agents", summary="Agents plan and act.", supporting_source_ids=["paper:test"])],
        evidence_bullets=[],
        limitations=[],
        open_problems=[],
        sources=[
            EvidenceSource(
                source_id="paper:test",
                title="Grounded Agent Survey",
                topic="AI Agents & Tool Use",
                paper_id="paper-test",
                citation_count=100,
                evidence_text="Agents use planning and tools.",
                score=0.9,
            )
        ],
    )


def test_initial_state_preserves_query_and_history():
    history = [ChatTurn(role="user", content="Explain RAG.")]

    state = initial_state(" What are its limitations? ", history)

    assert state["original_query"] == "What are its limitations?"
    assert state["standalone_query"] == "What are its limitations?"
    assert state["chat_history"] == history
    assert state["retry_count"] == 0
    assert state["retrieved_papers"] == []


def test_research_agent_high_confidence_path_runs_once():
    calls = {"search": [], "confidence": 0, "synthesis": 0}

    def rewriter(query, chat_history):
        return QueryRewriteResult(original_query=query, standalone_query="standalone agent query", rewrite_used=True, method="heuristic", reason="test")

    def searcher(query):
        calls["search"].append(query)
        return make_response(query)

    def confidence_checker(response):
        calls["confidence"] += 1
        return make_confidence(response, "sufficient_evidence")

    def synthesizer(response, confidence):
        calls["synthesis"] += 1
        return make_brief(response, confidence)

    state = run_research_agent(
        "What are its limitations?",
        chat_history=[ChatTurn(role="user", content="Explain AI agents.")],
        rewriter=rewriter,
        searcher=searcher,
        confidence_checker=confidence_checker,
        synthesizer=synthesizer,
    )

    assert calls == {"search": ["standalone agent query"], "confidence": 1, "synthesis": 1}
    assert state["standalone_query"] == "standalone agent query"
    assert state["confidence_decision"] == "sufficient_evidence"
    assert state["brief"].status == "generated"
    assert state["attempted_queries"] == ["standalone agent query"]
    assert state["warnings"] == ["Query rewritten with heuristic: test"]


def test_research_agent_retries_after_low_confidence_then_succeeds():
    searched = []

    def searcher(query):
        searched.append(query)
        return make_response(query)

    def confidence_checker(response):
        decision = "broaden_search" if len(searched) == 1 else "sufficient_evidence"
        return make_confidence(response, decision)

    def reflection_rewriter(state):
        return f"expanded {state['standalone_query']}"

    state = run_research_agent(
        "agent task execution",
        searcher=searcher,
        confidence_checker=confidence_checker,
        synthesizer=make_brief,
        reflection_rewriter=reflection_rewriter,
    )

    assert searched == ["agent task execution", "expanded agent task execution"]
    assert state["retry_count"] == 1
    assert state["confidence_decision"] == "sufficient_evidence"
    assert state["brief"].status == "generated"
    assert state["attempted_queries"] == ["agent task execution", "expanded agent task execution"]
    assert any("Retry 1" in warning for warning in state["warnings"])


def test_research_agent_stops_after_retry_limit_and_returns_guarded_brief():
    searched = []

    def searcher(query):
        searched.append(query)
        return make_response(query)

    def confidence_checker(response):
        return make_confidence(response, "broaden_search")

    state = run_research_agent(
        "out of corpus question",
        max_retries=1,
        searcher=searcher,
        confidence_checker=confidence_checker,
        synthesizer=make_brief,
        reflection_rewriter=lambda state: f"broader {state['standalone_query']}",
    )

    assert searched == ["out of corpus question", "broader out of corpus question"]
    assert state["retry_count"] == 1
    assert state["confidence_decision"] == "broaden_search"
    assert state["brief"].status == "skipped_low_confidence"


def test_research_agent_rejects_bad_retry_limit():
    with pytest.raises(ValueError, match="max_retries"):
        run_research_agent("valid query", max_retries=-1)
