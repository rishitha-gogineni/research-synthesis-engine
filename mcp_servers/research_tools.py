"""MCP server exposing the RSE corpus and external discovery tools."""
from __future__ import annotations
from typing import Any
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None
from agentic.dispatch import run_external_search
from agentic.tools import search_local_corpus as _search_local_corpus

MCP_AVAILABLE = FastMCP is not None
mcp = FastMCP("rse-research-tools") if MCP_AVAILABLE else None

def _external(source: str, query: str, top_k: int = 5) -> dict[str, Any]:
    response = run_external_search(query, sources=(source,), max_results=top_k)
    return response.as_dict()

if mcp is not None:
    @mcp.tool()
    def search_local_corpus(query: str, top_k: int = 8) -> dict[str, Any]:
        """Search the canonical RSE Qdrant and BM25 indexes."""
        return _search_local_corpus(query, top_k)

    @mcp.tool()
    def search_arxiv(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search Arxiv for current research papers."""
        return _external("arxiv", query, top_k)

    @mcp.tool()
    def search_semantic_scholar(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search Semantic Scholar for paper metadata."""
        return _external("semantic_scholar", query, top_k)

    @mcp.tool()
    def search_tavily(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search the web for current research sources using Tavily."""
        return _external("tavily", query, top_k)
else:
    def search_local_corpus(query: str, top_k: int = 8) -> dict[str, Any]:
        return _search_local_corpus(query, top_k)
    def search_arxiv(query: str, top_k: int = 5) -> dict[str, Any]:
        return _external("arxiv", query, top_k)
    def search_semantic_scholar(query: str, top_k: int = 5) -> dict[str, Any]:
        return _external("semantic_scholar", query, top_k)
    def search_tavily(query: str, top_k: int = 5) -> dict[str, Any]:
        return _external("tavily", query, top_k)

def tool_names() -> list[str]:
    return ["search_local_corpus", "search_arxiv", "search_semantic_scholar", "search_tavily"]

def main() -> None:
    if mcp is None:
        raise RuntimeError("The MCP package is not installed.")
    mcp.run()

if __name__ == "__main__":
    main()
