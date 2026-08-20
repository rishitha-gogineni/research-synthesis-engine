"""LangGraph orchestration with corpus, live, and coverage-aware routes."""
from typing import Any, Callable

from agentic.dispatch import run_external_search
from agentic.planner import RoutePlan, plan_query
from agentic.state import AgenticState, initial_state
from agentic.tools import run_local_corpus_search
from retrieval.confidence import assess_confidence

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    END = START = StateGraph = None

EXTERNAL_TOOLS = ("search_arxiv", "search_semantic_scholar", "search_tavily")
LOW_CONFIDENCE_DECISIONS = {"insufficient_evidence", "ask_clarifying_question", "broaden_search"}


class AgenticDependencyError(RuntimeError):
    """Raised when the LangGraph runtime is requested but not installed."""


def guardrail_node(state: AgenticState) -> AgenticState:
    if len(state["query"]) > 2000:
        state["status"] = "blocked"
        state["error"] = "query exceeds the 2000-character safety limit"
    return state


def planner_node(
    state: AgenticState,
    planner: Callable[[str], RoutePlan] = plan_query,
) -> AgenticState:
    plan = planner(state["query"])
    state.update(
        route=plan.route,
        route_reason=plan.reason,
        route_confidence=plan.confidence,
        fallback_external=plan.fallback_external,
        planned_tools=list(plan.tools),
    )
    return state


def local_corpus_node(
    state: AgenticState,
    searcher: Callable[..., Any] = run_local_corpus_search,
) -> AgenticState:
    try:
        response = searcher(state["query"], state["top_k"])
        if hasattr(response, "model_dump"):
            payload = response.model_dump(mode="json")
        elif hasattr(response, "dict"):
            payload = response.dict()
        else:
            payload = response
        state["retrieval_response"] = payload
        if hasattr(response, "route"):
            try:
                state["confidence_decision"] = assess_confidence(response).decision
            except Exception:
                pass

        evidence = []
        papers = list(getattr(response, "paper_results", None) or getattr(response, "papers", []) or [])
        chunks = list(getattr(response, "chunk_results", None) or [])
        if not chunks:
            for paper in papers:
                chunks.extend(getattr(paper, "chunks", []) or [])
        for paper in papers:
            evidence.append({"kind": "paper", "paper_id": paper.paper_id, "title": paper.title})
        for chunk in chunks:
            evidence.append(
                {
                    "kind": "chunk",
                    "paper_id": getattr(chunk, "paper_id", None),
                    "chunk_id": getattr(chunk, "chunk_id", None),
                    "text": chunk.text,
                    "page_start": getattr(chunk, "page_start", None),
                    "page_end": getattr(chunk, "page_end", None),
                }
            )
        state["evidence"] = evidence
        if state.get("route") == "corpus" and not evidence and not state.get("confidence_decision"):
            state["confidence_decision"] = "insufficient_evidence"
        state["tool_calls"] = [
            *state["tool_calls"],
            {"tool": "search_local_corpus", "query": state["query"], "top_k": state["top_k"], "status": "completed"},
        ]
        state["status"] = "completed"
    except Exception as exc:
        state["status"] = "blocked"
        state["error"] = str(exc)
        state["tool_calls"] = [
            *state["tool_calls"],
            {"tool": "search_local_corpus", "query": state["query"], "top_k": state["top_k"], "status": "failed"},
        ]
    return state


def after_local(state: AgenticState) -> str:
    if state.get("route") == "hybrid":
        return "external"
    decision = state.get("confidence_decision")
    should_fallback = (
        state.get("route") == "corpus"
        and state.get("fallback_external", False)
        and (
            state.get("status") == "blocked"
            or decision in LOW_CONFIDENCE_DECISIONS
            or not state.get("evidence")
        )
    )
    return "fallback" if should_fallback else "done"


def prepare_external_fallback_node(state: AgenticState) -> AgenticState:
    state["route"] = "hybrid"
    state["route_reason"] = (
        "Local evidence did not meet the confidence threshold; "
        "external research is being used as a coverage fallback."
    )
    state["planned_tools"] = list(
        dict.fromkeys([*state.get("planned_tools", []), *EXTERNAL_TOOLS])
    )
    state.pop("error", None)
    state["status"] = "ready"
    return state


def external_node(
    state: AgenticState,
    external_searcher: Callable[..., Any] = run_external_search,
) -> AgenticState:
    sources = tuple(
        name.removeprefix("search_")
        for name in state.get("planned_tools", [])
        if name in EXTERNAL_TOOLS
    )
    try:
        response = external_searcher(state["query"], sources=sources, max_results=state["top_k"])
        payload = response.as_dict() if hasattr(response, "as_dict") else dict(response)
        state["external_response"] = payload
        results = payload.get("results") or []
        state["evidence"] = [
            *state.get("evidence", []),
            *[{"kind": "external", **item} for item in results],
        ]
        state["warnings"] = [*state["warnings"], *payload.get("warnings", [])]
        for source in payload.get("sources", []):
            state["tool_calls"] = [
                *state["tool_calls"],
                {
                    "tool": f"search_{source}",
                    "query": state["query"],
                    "top_k": state["top_k"],
                    "status": "completed",
                },
            ]
        if results:
            state["confidence_decision"] = "sufficient_evidence"
            state["status"] = "completed"
        elif payload.get("warnings"):
            state["confidence_decision"] = "insufficient_evidence"
            state["status"] = "pending_external_tools"
        else:
            state["confidence_decision"] = "insufficient_evidence"
            state["status"] = "completed"
    except Exception as exc:
        state["status"] = "pending_external_tools"
        state["confidence_decision"] = "insufficient_evidence"
        state["warnings"] = [*state["warnings"], f"External discovery unavailable: {exc}"]
    return state


def after_guardrail(state: AgenticState) -> str:
    return "halt" if state["status"] == "blocked" else "plan"


def after_plan(state: AgenticState) -> str:
    return "local" if state.get("route") in {"corpus", "hybrid"} else "external"


def build_agent_graph(
    planner: Callable[[str], RoutePlan] = plan_query,
    searcher: Callable[..., Any] = run_local_corpus_search,
    external_searcher: Callable[..., Any] = run_external_search,
):
    if StateGraph is None:
        raise AgenticDependencyError(
            "LangGraph is not installed. Install the project dependencies to use the compiled graph."
        )
    graph = StateGraph(AgenticState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("planner", lambda state: planner_node(state, planner))
    graph.add_node("local_corpus", lambda state: local_corpus_node(state, searcher))
    graph.add_node("coverage_fallback", prepare_external_fallback_node)
    graph.add_node("external", lambda state: external_node(state, external_searcher))
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", after_guardrail, {"plan": "planner", "halt": END})
    graph.add_conditional_edges("planner", after_plan, {"local": "local_corpus", "external": "external"})
    graph.add_conditional_edges(
        "local_corpus",
        after_local,
        {"external": "external", "fallback": "coverage_fallback", "done": END},
    )
    graph.add_edge("coverage_fallback", "external")
    graph.add_edge("external", END)
    return graph.compile()


def run_agentic_research(
    query: str,
    top_k: int = 8,
    *,
    planner: Callable[[str], RoutePlan] = plan_query,
    searcher: Callable[..., Any] = run_local_corpus_search,
    external_searcher: Callable[..., Any] = run_external_search,
) -> AgenticState:
    state = initial_state(query, top_k)
    if StateGraph is not None:
        return build_agent_graph(planner, searcher, external_searcher).invoke(state)
    state = guardrail_node(state)
    if state["status"] == "blocked":
        return state
    state = planner_node(state, planner)
    if after_plan(state) == "local":
        state = local_corpus_node(state, searcher)
        if state["status"] == "blocked" and not state.get("fallback_external", False):
            return state
        local_decision = after_local(state)
        if local_decision == "done":
            return state
        if local_decision == "fallback":
            state = prepare_external_fallback_node(state)
    return external_node(state, external_searcher)
