"""Shared state types for the RSE multi-agent workflow."""
from typing import Any, Literal, NotRequired, TypedDict

AgentRoute = Literal["corpus", "live", "hybrid"]
AgentStatus = Literal["ready", "blocked", "completed", "pending_external_tools"]


class AgenticState(TypedDict):
    query: str
    top_k: int
    status: AgentStatus
    warnings: list[str]
    tool_calls: list[dict[str, Any]]
    route: NotRequired[AgentRoute]
    route_reason: NotRequired[str]
    route_confidence: NotRequired[float]
    fallback_external: NotRequired[bool]
    confidence_decision: NotRequired[str]
    planned_tools: NotRequired[list[str]]
    retrieval_response: NotRequired[dict[str, Any]]
    external_response: NotRequired[dict[str, Any]]
    evidence: NotRequired[list[dict[str, Any]]]
    error: NotRequired[str]


def initial_state(query: str, top_k: int = 8) -> AgenticState:
    cleaned = " ".join(query.split())
    if not cleaned:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return {
        "query": cleaned,
        "top_k": min(top_k, 50),
        "status": "ready",
        "warnings": [],
        "tool_calls": [],
    }
