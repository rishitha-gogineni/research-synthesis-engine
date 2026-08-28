"""FastAPI router for the multi-agent research endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from multi_agent.orchestrator import run_research
from multi_agent.config import DEFAULT_MODEL, classify_effort

router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    model: str = DEFAULT_MODEL


class ResearchResponse(BaseModel):
    query: str
    effort_level: str
    synthesis: dict
    cited_report: dict
    judge_scores: dict
    store_summary: dict
    trace_summary: dict


@router.post("/research", response_model=ResearchResponse)
def multi_agent_research(request: ResearchRequest) -> ResearchResponse:
    """Run the full multi-agent research pipeline.

    Executes: plan → parallel subagents → synthesize → cite → judge.
    """
    try:
        result = run_research(request.query, model=request.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research failed: {exc}")

    return ResearchResponse(
        query=request.query,
        effort_level=result["effort_level"],
        synthesis=result["synthesis"],
        cited_report=result["cited_report"],
        judge_scores=result["judge_scores"],
        store_summary=result["store_summary"],
        trace_summary=result["trace"],
    )


@router.post("/plan")
def preview_plan(request: ResearchRequest) -> dict:
    """Preview the research plan without executing subagents."""
    from multi_agent.lead import create_plan
    from multi_agent.trace import Tracer
    from openai import OpenAI

    tracer = Tracer()
    effort = classify_effort(request.query)
    plan = create_plan(
        request.query, tracer, client=OpenAI(), effort=effort, model=request.model
    )
    return {
        "query": request.query,
        "effort_level": effort.name,
        "plan": plan,
    }


@router.post("/research/light")
def multi_agent_research_light(request: ResearchRequest) -> dict:
    """Run multi-agent research without citation and judge (faster, cheaper)."""
    from multi_agent.lead import create_plan, synthesize_findings
    from multi_agent.orchestrator import _run_subagents_parallel
    from multi_agent.findings_store import FindingsStore
    from multi_agent.trace import Tracer
    from agentic.external import DEFAULT_EXTERNAL_CLIENT
    from openai import OpenAI

    client = OpenAI()
    store = FindingsStore()
    tracer = Tracer()
    effort = classify_effort(request.query)

    plan = create_plan(
        request.query, tracer, client=client, effort=effort, model=request.model
    )

    _run_subagents_parallel(
        plan.get("subtasks", []),
        store,
        tracer,
        client,
        DEFAULT_EXTERNAL_CLIENT,
        effort.max_tool_calls_per_agent,
    )

    synthesis = synthesize_findings(
        request.query, store, tracer, client=client, model=request.model
    )

    return {
        "query": request.query,
        "effort_level": effort.name,
        "synthesis": synthesis,
        "store_summary": store.summary(),
        "trace_summary": tracer.summary(),
    }


@router.post("/research/graph")
def multi_agent_research_graph(request: ResearchRequest) -> dict:
    """Run the multi-agent pipeline via LangGraph StateGraph.

    Uses the compiled LangGraph with proper nodes and conditional edges:
    guardrail → plan → subagents → synthesize → [loop] → cite → judge → END
    """
    from multi_agent.graph import run_multi_agent_graph

    try:
        result = run_multi_agent_graph(request.query)
        return {
            "query": result["query"],
            "effort_level": result["effort"],
            "synthesis": result["synthesis"],
            "cited_report": result["cited_report"],
            "judge_scores": result["judge_scores"],
            "iterations": result["iteration"],
            "findings_count": len(result.get("findings", [])),
            "trace_events": result.get("trace_events", []),
            "status": result["status"],
        }
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}")
