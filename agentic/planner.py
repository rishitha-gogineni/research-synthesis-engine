"""Deterministic query planner for corpus and live research routes."""
from dataclasses import dataclass
from typing import Literal
from retrieval.router import route_query

Route = Literal["corpus", "live", "hybrid"]
LOCAL_HINTS = ("in the corpus", "in our papers", "in the indexed papers", "from the collection", "from the uploaded papers", "according to the papers")
LIVE_HINTS = ("latest", "recent", "current", "today", "new papers", "search the web", "on arxiv", "on semantic scholar")
COMPARISON_HINTS = ("compare", "comparison", "versus", " vs ", "difference between")

@dataclass(frozen=True)
class RoutePlan:
    route: Route
    tools: tuple[str, ...]
    reason: str
    confidence: float

def plan_query(query: str) -> RoutePlan:
    normalized = " ".join(query.lower().split())
    if not normalized:
        raise ValueError("query must not be empty")
    has_local = any(x in normalized for x in LOCAL_HINTS)
    has_live = any(x in normalized for x in LIVE_HINTS)
    has_comparison = any(x in normalized for x in COMPARISON_HINTS)
    if has_local and not has_live:
        return RoutePlan("corpus", ("search_local_corpus",), "The query explicitly asks about the indexed corpus.", 0.98)
    if has_live and has_comparison:
        return RoutePlan("hybrid", ("search_local_corpus", "search_arxiv", "search_semantic_scholar", "search_tavily"), "The query asks for current information and a comparison with the corpus.", 0.94)
    if has_live:
        return RoutePlan("live", ("search_arxiv", "search_semantic_scholar", "search_tavily"), "The query asks for current or external research.", 0.94)
    routed = route_query(query)
    return RoutePlan("corpus", ("search_local_corpus",), f"Planner delegated to the existing RSE router ({routed.route}).", max(0.82, float(routed.confidence)))
