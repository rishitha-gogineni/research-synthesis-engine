"""LangGraph StateGraph for the multi-agent research pipeline.

Implements the orchestrator-worker pattern as a proper LangGraph graph:
  plan → spawn_subagents → synthesize → [loop] → cite → judge → END
"""

from __future__ import annotations

import json
import sys
from typing import Any

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    END = START = StateGraph = None

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired
    from typing import TypedDict

from multi_agent.config import classify_effort, EffortLevel, DEFAULT_MODEL, SUBAGENT_MODEL


class MultiAgentState(TypedDict):
    """State flowing through the multi-agent LangGraph."""
    query: str
    effort: str
    max_subagents: int
    max_tool_calls: int
    max_iterations: int
    iteration: int
    plan: dict
    subtasks: list
    findings: list
    synthesis: dict
    cited_report: dict
    judge_scores: dict
    needs_more_research: bool
    status: str
    error: str
    trace_events: list


def initial_state(query: str) -> MultiAgentState:
    effort = classify_effort(query)
    return MultiAgentState(
        query=query,
        effort=effort.name,
        max_subagents=effort.max_subagents,
        max_tool_calls=effort.max_tool_calls_per_agent,
        max_iterations=effort.max_iterations,
        iteration=0,
        plan={},
        subtasks=[],
        findings=[],
        synthesis={},
        cited_report={},
        judge_scores={},
        needs_more_research=False,
        status="ready",
        error="",
        trace_events=[],
    )


def guardrail_node(state: MultiAgentState) -> MultiAgentState:
    """Block queries that exceed safety limits."""
    if len(state["query"]) > 2000:
        state["status"] = "blocked"
        state["error"] = "Query exceeds 2000-character safety limit"
    return state


def plan_node(state: MultiAgentState) -> MultiAgentState:
    """Lead agent decomposes query into subtasks."""
    from multi_agent.lead import create_plan
    from multi_agent.trace import Tracer
    from openai import OpenAI

    tracer = Tracer()
    client = OpenAI()
    effort = classify_effort(state["query"])

    try:
        plan = create_plan(state["query"], tracer, client=client, effort=effort)
        state["plan"] = plan
        state["subtasks"] = plan.get("subtasks", [])
        state["trace_events"].append({"node": "plan", "subtask_count": len(state["subtasks"])})
    except Exception as exc:
        state["status"] = "blocked"
        state["error"] = f"Planning failed: {exc}"

    return state


def subagents_node(state: MultiAgentState) -> MultiAgentState:
    """Spawn parallel subagents to execute subtasks."""
    from multi_agent.orchestrator import _run_subagents_parallel
    from multi_agent.findings_store import FindingsStore
    from multi_agent.trace import Tracer
    from agentic.external import DEFAULT_EXTERNAL_CLIENT
    from openai import OpenAI

    store = FindingsStore()
    tracer = Tracer()
    client = OpenAI()

    _run_subagents_parallel(
        state["subtasks"],
        store,
        tracer,
        client,
        DEFAULT_EXTERNAL_CLIENT,
        state["max_tool_calls"],
    )

    # Collect findings from store
    all_findings = []
    for result in store.get_completed():
        for f in result.findings:
            all_findings.append({
                "source": f.source,
                "title": f.title,
                "content": f.content,
                "url": f.url,
                "relevance_score": f.relevance_score,
            })

    state["findings"] = [*state.get("findings", []), *all_findings]
    state["iteration"] = state.get("iteration", 0) + 1
    state["trace_events"].append({
        "node": "subagents",
        "iteration": state["iteration"],
        "findings_count": len(all_findings),
        "agents_used": len(store.get_all()),
    })

    return state


def synthesize_node(state: MultiAgentState) -> MultiAgentState:
    """Lead agent synthesizes all findings."""
    from multi_agent.lead import synthesize_findings
    from multi_agent.findings_store import FindingsStore, Finding, SubagentResult
    from multi_agent.trace import Tracer
    from openai import OpenAI

    # Rebuild a store from accumulated findings
    store = FindingsStore()
    findings = [
        Finding(
            source=f.get("source", ""),
            title=f.get("title", ""),
            content=f.get("content", ""),
            url=f.get("url", ""),
            relevance_score=f.get("relevance_score", 0.0),
        )
        for f in state.get("findings", [])
    ]
    store.store(SubagentResult(
        agent_id="aggregated",
        agent_type="all",
        subtask="all subtasks",
        status="complete",
        findings=findings,
    ))

    tracer = Tracer()
    client = OpenAI()
    synthesis = synthesize_findings(state["query"], store, tracer, client=client)

    state["synthesis"] = synthesis
    state["needs_more_research"] = synthesis.get("needs_more_research", False)
    state["trace_events"].append({
        "node": "synthesize",
        "confidence": synthesis.get("confidence", "unknown"),
        "needs_more": state["needs_more_research"],
    })

    return state


def cite_node(state: MultiAgentState) -> MultiAgentState:
    """Citation agent attributes sources."""
    from multi_agent.citation import add_citations
    from multi_agent.findings_store import FindingsStore, Finding, SubagentResult
    from multi_agent.trace import Tracer
    from openai import OpenAI

    store = FindingsStore()
    findings = [
        Finding(
            source=f.get("source", ""),
            title=f.get("title", ""),
            content=f.get("content", ""),
            url=f.get("url", ""),
        )
        for f in state.get("findings", [])
    ]
    store.store(SubagentResult(
        agent_id="aggregated", agent_type="all", subtask="all",
        status="complete", findings=findings,
    ))

    tracer = Tracer()
    client = OpenAI()
    cited = add_citations(state["synthesis"], store, tracer, client=client)

    state["cited_report"] = cited
    state["trace_events"].append({"node": "cite", "references": len(cited.get("references", []))})
    return state


def judge_node(state: MultiAgentState) -> MultiAgentState:
    """LLM-as-judge scores the output."""
    from multi_agent.judge import evaluate_output
    from multi_agent.trace import Tracer
    from openai import OpenAI

    tracer = Tracer()
    client = OpenAI()
    store_summary = {
        "total_agents": state.get("iteration", 1) * state.get("max_subagents", 3),
        "total_findings": len(state.get("findings", [])),
        "elapsed_seconds": 0,
    }

    scores = evaluate_output(state["query"], state["cited_report"], store_summary, tracer, client=client)
    state["judge_scores"] = scores
    state["status"] = "completed"
    state["trace_events"].append({"node": "judge", "overall": scores.get("overall", 0)})
    return state


def after_guardrail(state: MultiAgentState) -> str:
    return "halt" if state["status"] == "blocked" else "plan"


def after_synthesis(state: MultiAgentState) -> str:
    """Decide whether to loop for more research."""
    if state.get("needs_more_research", False) and state.get("iteration", 0) < state.get("max_iterations", 2):
        return "more_research"
    return "cite"


def build_multi_agent_graph():
    """Build and compile the multi-agent LangGraph StateGraph."""
    if StateGraph is None:
        raise RuntimeError("LangGraph is not installed.")

    graph = StateGraph(MultiAgentState)

    # Add nodes
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("plan", plan_node)
    graph.add_node("subagents", subagents_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("cite", cite_node)
    graph.add_node("judge", judge_node)

    # Add edges
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", after_guardrail, {"plan": "plan", "halt": END})
    graph.add_edge("plan", "subagents")
    graph.add_edge("subagents", "synthesize")
    graph.add_conditional_edges("synthesize", after_synthesis, {"more_research": "plan", "cite": "cite"})
    graph.add_edge("cite", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


def run_multi_agent_graph(query: str) -> MultiAgentState:
    """Execute the multi-agent research pipeline via LangGraph."""
    state = initial_state(query)

    if StateGraph is not None:
        graph = build_multi_agent_graph()
        return graph.invoke(state)

    # Fallback without LangGraph
    state = guardrail_node(state)
    if state["status"] == "blocked":
        return state
    state = plan_node(state)
    if state["status"] == "blocked":
        return state
    state = subagents_node(state)
    state = synthesize_node(state)

    iteration = 1
    while state.get("needs_more_research") and iteration < state.get("max_iterations", 2):
        state = plan_node(state)
        state = subagents_node(state)
        state = synthesize_node(state)
        iteration += 1

    state = cite_node(state)
    state = judge_node(state)
    return state
