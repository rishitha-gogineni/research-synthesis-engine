"""Lightweight state graph for multi-turn research guidance."""

from __future__ import annotations

from typing import Callable, NotRequired, TypedDict

from agent.query_rewriter import ChatTurn, QueryRewriteResult, rewrite_query
from agent.synthesis import build_research_brief
from retrieval.confidence import assess_confidence
from retrieval.unified_search import run_unified_search
from shared.schemas import ConfidenceAssessment, ResearchBrief, RetrievedChunk, RetrievedPaper, UnifiedSearchResponse


DEFAULT_MAX_RETRIES = 2


class ResearchAgentState(TypedDict):
    """State carried across the research-agent loop."""

    original_query: str
    chat_history: list[ChatTurn]
    standalone_query: str
    retrieved_papers: list[RetrievedPaper]
    retrieved_chunks: list[RetrievedChunk]
    confidence_decision: str | None
    retry_count: int
    retrieval_response: NotRequired[UnifiedSearchResponse]
    confidence: NotRequired[ConfidenceAssessment]
    brief: NotRequired[ResearchBrief]
    attempted_queries: NotRequired[list[str]]
    warnings: NotRequired[list[str]]


RewriteFn = Callable[[str, list[ChatTurn]], QueryRewriteResult]
SearchFn = Callable[[str], UnifiedSearchResponse]
ConfidenceFn = Callable[[UnifiedSearchResponse], ConfidenceAssessment]
SynthesisFn = Callable[[UnifiedSearchResponse, ConfidenceAssessment], ResearchBrief]
ReflectionFn = Callable[[ResearchAgentState], str]


def _clean_query(query: str) -> str:
    cleaned = " ".join(str(query or "").split())
    if not cleaned:
        raise ValueError("query must not be empty")
    return cleaned


def initial_state(query: str, chat_history: list[ChatTurn] | None = None) -> ResearchAgentState:
    original = _clean_query(query)
    return ResearchAgentState(
        original_query=original,
        chat_history=list(chat_history or []),
        standalone_query=original,
        retrieved_papers=[],
        retrieved_chunks=[],
        confidence_decision=None,
        retry_count=0,
        attempted_queries=[],
        warnings=[],
    )


def context_rewrite_node(state: ResearchAgentState, *, rewriter: RewriteFn = rewrite_query) -> ResearchAgentState:
    result = rewriter(state["original_query"], state["chat_history"])
    state["standalone_query"] = result.standalone_query
    state.setdefault("attempted_queries", []).append(result.standalone_query)
    if result.rewrite_used:
        state.setdefault("warnings", []).append(f"Query rewritten with {result.method}: {result.reason}")
    return state


def unified_search_node(state: ResearchAgentState, *, searcher: SearchFn) -> ResearchAgentState:
    response = searcher(state["standalone_query"])
    state["retrieval_response"] = response
    state["retrieved_papers"] = list(response.paper_results)
    state["retrieved_chunks"] = list(response.chunk_results)
    return state


def confidence_node(state: ResearchAgentState, *, confidence_checker: ConfidenceFn = assess_confidence) -> ResearchAgentState:
    response = state.get("retrieval_response")
    if response is None:
        raise ValueError("retrieval_response is required before confidence assessment")
    confidence = confidence_checker(response)
    state["confidence"] = confidence
    state["confidence_decision"] = confidence.decision
    return state


def synthesis_node(state: ResearchAgentState, *, synthesizer: SynthesisFn) -> ResearchAgentState:
    response = state.get("retrieval_response")
    confidence = state.get("confidence")
    if response is None or confidence is None:
        raise ValueError("retrieval_response and confidence are required before synthesis")
    state["brief"] = synthesizer(response, confidence)
    return state


def should_retry(state: ResearchAgentState, *, max_retries: int = DEFAULT_MAX_RETRIES) -> bool:
    return state.get("confidence_decision") != "sufficient_evidence" and state["retry_count"] < max_retries


def default_reflection_rewrite(state: ResearchAgentState) -> str:
    confidence = state.get("confidence")
    reason = confidence.reason if confidence else "retrieved evidence was weak"
    query = state["standalone_query"]
    return (
        f"{query} broader evidence survey methods datasets limitations. "
        f"Resolve weak retrieval because: {reason}"
    )


def reflection_rewrite_node(
    state: ResearchAgentState,
    *,
    reflection_rewriter: ReflectionFn = default_reflection_rewrite,
) -> ResearchAgentState:
    state["retry_count"] += 1
    next_query = _clean_query(reflection_rewriter(state))
    state["standalone_query"] = next_query
    state.setdefault("attempted_queries", []).append(next_query)
    state.setdefault("warnings", []).append(f"Retry {state['retry_count']} used expanded query after low confidence.")
    return state


def default_searcher(query: str) -> UnifiedSearchResponse:
    return run_unified_search(query, top_k=8)


def default_synthesizer(response: UnifiedSearchResponse, confidence: ConfidenceAssessment) -> ResearchBrief:
    return build_research_brief(response, confidence=confidence)


def run_research_agent(
    query: str,
    *,
    chat_history: list[ChatTurn] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    rewriter: RewriteFn = rewrite_query,
    searcher: SearchFn = default_searcher,
    confidence_checker: ConfidenceFn = assess_confidence,
    synthesizer: SynthesisFn = default_synthesizer,
    reflection_rewriter: ReflectionFn = default_reflection_rewrite,
) -> ResearchAgentState:
    """Run rewrite -> retrieval -> confidence -> synthesis with bounded low-confidence retries."""

    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    state = initial_state(query, chat_history)
    state = context_rewrite_node(state, rewriter=rewriter)

    while True:
        state = unified_search_node(state, searcher=searcher)
        state = confidence_node(state, confidence_checker=confidence_checker)
        if not should_retry(state, max_retries=max_retries):
            break
        state = reflection_rewrite_node(state, reflection_rewriter=reflection_rewriter)

    state = synthesis_node(state, synthesizer=synthesizer)
    return state
