"""MCP server exposing research tools for the multi-agent system.

Provides both corpus search tools and multi-agent orchestration tools.
Tool descriptions are written for agent consumption following best practices:
- Clear purpose statement
- Input/output format
- When to use vs when NOT to use
- Example queries
"""
from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None

from agentic.dispatch import run_external_search
from agentic.tools import search_local_corpus as _search_local_corpus

MCP_AVAILABLE = FastMCP is not None
mcp = FastMCP("multi-agent-research-tools") if MCP_AVAILABLE else None


def _external(source: str, query: str, top_k: int = 5) -> dict[str, Any]:
    response = run_external_search(query, sources=(source,), max_results=top_k)
    return response.as_dict()


if mcp is not None:

    @mcp.tool()
    def search_local_corpus(query: str, top_k: int = 8) -> dict[str, Any]:
        """Search the indexed corpus of 250 AI research papers using hybrid dense+BM25 retrieval.

        USE THIS TOOL WHEN: The question is about AI/ML topics covered in the corpus
        (RAG, transformers, hallucination, agents, fine-tuning).

        DO NOT USE WHEN: The question requires very recent information (after 2024),
        non-AI topics, or specific company/product information.

        INPUT: A search query (2-5 words works best). Avoid long, overly specific queries.
        OUTPUT: Ranked list of papers/passages with title, abstract, relevance score, and metadata.

        EXAMPLES:
        - Good: "RAG retrieval methods"
        - Good: "hallucination detection"
        - Bad: "what are the latest advances in retrieval augmented generation for multi-hop reasoning in 2025"
        """
        return _search_local_corpus(query, top_k)

    @mcp.tool()
    def search_arxiv(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search arXiv for academic research papers. Returns titles, abstracts, and URLs.

        USE THIS TOOL WHEN: You need recent academic papers, preprints, or papers
        NOT in the local corpus. Good for cutting-edge research.

        DO NOT USE WHEN: The local corpus already has sufficient information on the topic.
        Prefer search_local_corpus first, use this as a supplement.

        INPUT: Short academic search query (2-4 words). arXiv responds best to concise queries.
        OUTPUT: List of papers with title, abstract, authors, publication date, and arXiv URL.

        EXAMPLES:
        - Good: "multi-agent LLM systems"
        - Good: "LoRA efficient fine-tuning"
        - Bad: "papers about how to reduce hallucinations in large language models using retrieval"
        """
        return _external("arxiv", query, top_k)

    @mcp.tool()
    def search_semantic_scholar(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search Semantic Scholar for papers with citation counts and metadata.

        USE THIS TOOL WHEN: You need citation counts, want to find highly-cited papers,
        or need to understand a paper's impact. Good for surveys and literature reviews.

        DO NOT USE WHEN: You just need paper content (use arxiv or local corpus instead).

        INPUT: Short query (2-4 words). Returns papers ranked by relevance.
        OUTPUT: Papers with title, abstract, authors, citation count, year, and S2 URL.

        EXAMPLES:
        - Good: "attention mechanism transformer"
        - Good: "RLHF language model"
        """
        return _external("semantic_scholar", query, top_k)

    @mcp.tool()
    def search_web(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search the web using Tavily for non-academic sources (blogs, docs, news).

        USE THIS TOOL WHEN: You need information from non-academic sources — blog posts,
        documentation, news articles, company announcements, tutorials.

        DO NOT USE WHEN: Academic papers would be a better source. Prefer arxiv/semantic_scholar
        for academic content and local corpus for indexed papers.

        INPUT: Natural language query. Can be longer than academic search queries.
        OUTPUT: Web results with title, content snippet, URL, and relevance score.

        EXAMPLES:
        - Good: "LangGraph multi-agent tutorial 2025"
        - Good: "OpenAI function calling best practices"
        """
        return _external("tavily", query, top_k)

    @mcp.tool()
    def run_multi_agent_research(query: str) -> dict[str, Any]:
        """Run the full multi-agent research pipeline on a complex query.

        This spawns a lead agent that decomposes the query into subtasks,
        delegates to parallel subagents (local corpus, arxiv, semantic scholar, web),
        synthesizes findings, adds citations, and scores quality.

        USE THIS TOOL WHEN: The query is complex, requires multiple perspectives,
        or needs information from multiple sources. Good for comprehensive research questions.

        DO NOT USE WHEN: The query is simple and can be answered with a single search.
        This tool is expensive (multiple LLM calls + API calls).

        INPUT: A research question (can be complex and multi-faceted).
        OUTPUT: Synthesis, cited report, quality scores, and execution trace.
        """
        from multi_agent.orchestrator import run_research
        return run_research(query)

    @mcp.tool()
    def preview_research_plan(query: str) -> dict[str, Any]:
        """Preview how the multi-agent system would decompose a research query.

        Returns the lead agent's plan (subtasks, sources, queries) without executing.
        Use this to understand the system's approach before committing to a full run.

        INPUT: A research question.
        OUTPUT: Plan with subtasks, assigned sources, and suggested queries per subtask.
        """
        from multi_agent.lead import create_plan
        from multi_agent.trace import Tracer
        from multi_agent.config import classify_effort
        from openai import OpenAI

        tracer = Tracer()
        effort = classify_effort(query)
        plan = create_plan(query, tracer, client=OpenAI(), effort=effort)
        return {"effort_level": effort.name, "plan": plan}

else:
    # Fallback definitions when MCP is not installed
    def search_local_corpus(query: str, top_k: int = 8) -> dict[str, Any]:
        return _search_local_corpus(query, top_k)

    def search_arxiv(query: str, top_k: int = 5) -> dict[str, Any]:
        return _external("arxiv", query, top_k)

    def search_semantic_scholar(query: str, top_k: int = 5) -> dict[str, Any]:
        return _external("semantic_scholar", query, top_k)

    def search_web(query: str, top_k: int = 5) -> dict[str, Any]:
        return _external("tavily", query, top_k)

    def run_multi_agent_research(query: str) -> dict[str, Any]:
        from multi_agent.orchestrator import run_research
        return run_research(query)

    def preview_research_plan(query: str) -> dict[str, Any]:
        from multi_agent.lead import create_plan
        from multi_agent.trace import Tracer
        from multi_agent.config import classify_effort
        from openai import OpenAI

        tracer = Tracer()
        effort = classify_effort(query)
        plan = create_plan(query, tracer, client=OpenAI(), effort=effort)
        return {"effort_level": effort.name, "plan": plan}


def tool_names() -> list[str]:
    return [
        "search_local_corpus",
        "search_arxiv",
        "search_semantic_scholar",
        "search_web",
        "run_multi_agent_research",
        "preview_research_plan",
    ]


def main() -> None:
    if mcp is None:
        raise RuntimeError("The MCP package is not installed. Run: pip install mcp")
    mcp.run()


if __name__ == "__main__":
    main()
