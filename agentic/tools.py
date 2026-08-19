"""Tool contracts and adapters for the RSE agent layer."""
from dataclasses import dataclass
from typing import Any, Callable
from retrieval.unified_search import run_unified_search
class AgentToolError(RuntimeError):
    """Raised when an agent tool cannot validate or execute a request."""
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    def as_openai_tool(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.input_schema}}
LOCAL_CORPUS_TOOL = ToolSpec("search_local_corpus", "Search the canonical RSE Qdrant and BM25 indexes for grounded evidence.", {"type": "object", "properties": {"query": {"type": "string", "description": "Research question to search."}, "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8}}, "required": ["query"], "additionalProperties": False})
def run_local_corpus_search(query: str, top_k: int = 8) -> Any:
    try:
        return run_unified_search(query, top_k=min(max(top_k, 1), 50))
    except Exception as exc:
        raise AgentToolError(f"Local corpus search failed: {exc}") from exc
def search_local_corpus(query: str, top_k: int = 8) -> dict[str, Any]:
    response = run_local_corpus_search(query, top_k)
    if hasattr(response, "model_dump"): payload = response.model_dump(mode="json")
    elif hasattr(response, "dict"): payload = response.dict()
    elif isinstance(response, dict): payload = dict(response)
    else: raise AgentToolError("Local search returned an unsupported response type.")
    payload["source_kind"] = "local_corpus"
    return payload
def tool_definitions() -> list[dict[str, Any]]:
    return [LOCAL_CORPUS_TOOL.as_openai_tool()]
def execute_tool(name: str, arguments: dict[str, Any], *, searcher: Callable[..., dict[str, Any]] = search_local_corpus) -> dict[str, Any]:
    if name != LOCAL_CORPUS_TOOL.name: raise AgentToolError(f"Unknown tool: {name}")
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip(): raise AgentToolError("search_local_corpus requires a non-empty query.")
    top_k = arguments.get("top_k", 8)
    if isinstance(top_k, bool) or not isinstance(top_k, int): raise AgentToolError("top_k must be an integer.")
    return searcher(query, min(max(top_k, 1), 50))
