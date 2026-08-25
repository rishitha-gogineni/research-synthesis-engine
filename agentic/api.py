"""FastAPI router for the optional multi-agent research workflow."""
from __future__ import annotations
import logging
import time
from typing import Any
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator
from agentic.graph import run_agentic_research
from agentic.llm import LLMResult, run_grounded_answer

LOGGER = logging.getLogger("research_synthesis_engine.agentic")
router = APIRouter(prefix="/agentic", tags=["Agentic"])

class AgenticResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    max_tool_calls: int = Field(default=3, ge=0, le=6)
    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value: raise ValueError("question must not be empty")
        return value

class AgenticResearchResponse(BaseModel):
    request_id: str | None = None
    query: str
    status: str
    route: str | None = None
    route_reason: str | None = None
    route_confidence: float | None = None
    confidence_decision: str | None = None
    planned_tools: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    llm_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    llm_usage: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    latency_ms: float
    llm_latency_ms: float | None = None

@router.post("/research", response_model=AgenticResearchResponse, summary="Run corpus, live, or hybrid agentic research")
def agentic_research(request: AgenticResearchRequest, http_request: Request) -> AgenticResearchResponse:
    started = time.perf_counter()
    state = run_agentic_research(request.question, request.top_k)
    request_id = getattr(http_request.state, "request_id", None)
    warnings = list(state.get("warnings", []))
    answer = None
    citations: list[str] = []
    llm_tool_calls: list[dict[str, Any]] = []
    llm_usage: dict[str, int] = {}
    llm_latency_ms = None
    decision = state.get("confidence_decision")
    if state["status"] == "completed" and decision not in {"ask_clarifying_question", "insufficient_evidence"}:
        try:
            route = state.get("route")
            route_tools = {
                "corpus": ("search_local_corpus",),
                "live": ("search_arxiv", "search_semantic_scholar", "search_tavily"),
                "hybrid": ("search_local_corpus", "search_arxiv", "search_semantic_scholar", "search_tavily"),
            }.get(route)
            route_budget = min(
                request.max_tool_calls,
                {"corpus": 1, "live": 3, "hybrid": 3}.get(route, request.max_tool_calls),
            )
            result: LLMResult = run_grounded_answer(
                request.question,
                state.get("evidence", []),
                max_tool_calls=route_budget,
                allowed_tools=route_tools,
            )
            answer = result.answer
            citations = result.citations
            llm_tool_calls = result.tool_calls
            llm_usage = result.usage
            warnings.extend(result.warnings)
            llm_latency_ms = result.latency_ms
        except Exception as exc:
            warnings.append(f"Grounded LLM synthesis unavailable: {exc}")
            LOGGER.warning("agentic_synthesis_unavailable", extra={"request_id": request_id, "error": str(exc)})
    if decision in {"ask_clarifying_question", "insufficient_evidence"}:
        warnings.append(f"LLM synthesis skipped because confidence decision is {decision}.")
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    LOGGER.info("agentic_workflow", extra={"request_id": request_id, "route": state.get("route"), "status": state["status"], "tool_calls": [*state.get("tool_calls", []), *llm_tool_calls], "latency_ms": latency_ms, "llm_latency_ms": llm_latency_ms})
    return AgenticResearchResponse(
        request_id=request_id,
        query=state["query"],
        status=state["status"],
        route=state.get("route"),
        route_reason=state.get("route_reason"),
        route_confidence=state.get("route_confidence"),
        confidence_decision=decision,
        planned_tools=state.get("planned_tools", []),
        tool_calls=state.get("tool_calls", []),
        evidence=state.get("evidence", []),
        answer=answer,
        citations=citations,
        llm_tool_calls=llm_tool_calls,
        llm_usage=llm_usage,
        warnings=warnings,
        error=state.get("error"),
        latency_ms=latency_ms,
        llm_latency_ms=llm_latency_ms,
    )
