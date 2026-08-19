"""FastAPI router for the optional multi-agent research workflow."""
from __future__ import annotations
import logging
import time
from typing import Any
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator
from agentic.graph import run_agentic_research

LOGGER = logging.getLogger("research_synthesis_engine.agentic")
router = APIRouter(prefix="/agentic", tags=["Agentic"])

class AgenticResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
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
    planned_tools: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    latency_ms: float

@router.post("/research", response_model=AgenticResearchResponse, summary="Run corpus, live, or hybrid agentic research")
def agentic_research(request: AgenticResearchRequest, http_request: Request) -> AgenticResearchResponse:
    started = time.perf_counter()
    state = run_agentic_research(request.question, request.top_k)
    request_id = getattr(http_request.state, "request_id", None)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    LOGGER.info("agentic_workflow", extra={"request_id": request_id, "route": state.get("route"), "status": state["status"], "tool_calls": state.get("tool_calls", []), "latency_ms": latency_ms})
    return AgenticResearchResponse(
        request_id=request_id,
        query=state["query"],
        status=state["status"],
        route=state.get("route"),
        route_reason=state.get("route_reason"),
        route_confidence=state.get("route_confidence"),
        planned_tools=state.get("planned_tools", []),
        tool_calls=state.get("tool_calls", []),
        evidence=state.get("evidence", []),
        warnings=state.get("warnings", []),
        error=state.get("error"),
        latency_ms=latency_ms,
    )
