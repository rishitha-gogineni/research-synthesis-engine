"""Autonomous subagent that searches, evaluates, and refines iteratively."""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from agentic.external import ExternalSearchClient, ExternalPaper
from multi_agent.config import SUBAGENT_MODEL, SUBAGENT_TIMEOUT_SECONDS
from multi_agent.findings_store import Finding, SubagentResult, FindingsStore
from multi_agent.prompts import SUBAGENT_SYSTEM_PROMPT, SUBAGENT_EVALUATE_PROMPT
from multi_agent.trace import Tracer


class SubagentError(RuntimeError):
    pass


SOURCE_FALLBACKS = {
    "arxiv": ["semantic_scholar", "web"],
    "semantic_scholar": ["arxiv", "web"],
    "web": ["semantic_scholar"],
    "local_corpus": ["arxiv", "semantic_scholar"],
}


def _papers_to_findings(papers: list[ExternalPaper], source: str) -> list[Finding]:
    return [
        Finding(
            source=source,
            title=p.title,
            content=p.abstract,
            url=p.url,
            relevance_score=p.relevance_score or 0.0,
            metadata={
                "authors": list(p.authors),
                "published_date": p.published_date,
                "citation_count": p.citation_count,
                "paper_id": p.paper_id,
            },
        )
        for p in papers
        if p.title
    ]


def _search_local_corpus(query: str, top_k: int = 10) -> list[Finding]:
    """Search local Qdrant corpus using unified paper+chunk search."""
    try:
        from retrieval.unified_search import run_unified_search
        from multi_agent.corpus_relevance import best_score

        response = run_unified_search(query, top_k=top_k)
        findings: list[Finding] = []

        # Prefer chunks (full text) over papers (abstract-only) for depth
        for chunk in response.chunk_results[:top_k]:
            findings.append(
                Finding(
                    source="local_corpus",
                    title=chunk.title or "",
                    content=chunk.text or "",
                    url=chunk.pdf_url or "",
                    relevance_score=best_score(chunk),
                    metadata={
                        "paper_id": chunk.paper_id,
                        "section": chunk.section_hint,
                        "chunk_index": chunk.chunk_index,
                        "year": chunk.year,
                        "citation_count": chunk.citation_count,
                        "level": "chunk",
                    },
                )
            )

        # Add paper-level results only if we haven't already found the paper
        seen_paper_ids = {f.metadata.get("paper_id") for f in findings if f.metadata.get("paper_id")}
        for paper in response.paper_results[:top_k]:
            if paper.paper_id in seen_paper_ids:
                continue
            findings.append(
                Finding(
                    source="local_corpus",
                    title=paper.title or "",
                    content=paper.abstract or "",
                    url=paper.pdf_url or "",
                    relevance_score=best_score(paper),
                    metadata={
                        "paper_id": paper.paper_id,
                        "year": paper.year,
                        "citation_count": paper.citation_count,
                        "level": "paper",
                    },
                )
            )

        return findings[:top_k]
    except Exception as exc:
        raise SubagentError(f"Local corpus search failed: {exc}") from exc


def _search_source(
    source: str,
    query: str,
    client: ExternalSearchClient,
    max_results: int = 5,
) -> list[Finding]:
    """Dispatch search to the appropriate source."""
    if source == "local_corpus":
        return _search_local_corpus(query, top_k=max_results)
    elif source == "arxiv":
        papers = client.search_arxiv(query, max_results=max_results)
        return _papers_to_findings(papers, "arxiv")
    elif source == "semantic_scholar":
        papers = client.search_semantic_scholar(query, max_results=max_results)
        return _papers_to_findings(papers, "semantic_scholar")
    elif source == "web":
        papers = client.search_tavily(query, max_results=max_results)
        return _papers_to_findings(papers, "web")
    else:
        raise SubagentError(f"Unknown source: {source}")


def _evaluate_results(
    openai_client: OpenAI,
    query: str,
    result_count: int,
    objective: str,
    model: str = SUBAGENT_MODEL,
) -> dict[str, Any]:
    """Ask the LLM whether current results are sufficient."""
    prompt = SUBAGENT_EVALUATE_PROMPT.format(
        query=query,
        result_count=result_count,
        objective=objective,
    )
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"sufficient": True, "reasoning": "Failed to parse evaluation"}


def run_subagent(
    subtask: dict[str, Any],
    store: FindingsStore,
    tracer: Tracer,
    *,
    openai_client: OpenAI | None = None,
    external_client: ExternalSearchClient | None = None,
    max_tool_calls: int = 10,
    model: str = SUBAGENT_MODEL,
) -> SubagentResult:
    """Run a single subagent autonomously: search → evaluate → refine → complete."""
    from agentic.external import DEFAULT_EXTERNAL_CLIENT

    if openai_client is None:
        openai_client = OpenAI()
    if external_client is None:
        external_client = DEFAULT_EXTERNAL_CLIENT

    source = subtask["source"]
    objective = subtask["objective"]
    queries = subtask.get("queries", [])
    agent_id = store.create_agent_id(source)

    tracer.log(agent_id, "start", objective=objective, source=source)
    start_time = time.time()
    all_findings: list[Finding] = []
    queries_used: list[str] = []
    tool_calls = 0

    # Agent-to-agent awareness: check what other agents have already found
    existing_findings = store.get_all_findings()
    existing_titles = {f.title.lower().strip() for f in existing_findings}
    if existing_titles:
        tracer.log(agent_id, "aware_of_peers", existing_findings_count=len(existing_titles))

    for query in queries[:max_tool_calls]:
        elapsed = time.time() - start_time
        if elapsed > SUBAGENT_TIMEOUT_SECONDS:
            tracer.log(agent_id, "timeout")
            break
        if tool_calls >= max_tool_calls:
            tracer.log(agent_id, "max_tool_calls_reached")
            break

        tracer.log(agent_id, "search", query=query, source=source)
        try:
            findings = _search_source(source, query, external_client)
            all_findings.extend(findings)
            queries_used.append(query)
            tool_calls += 1
        except (SubagentError, Exception) as exc:
            tracer.log(agent_id, "error", error=str(exc), query=query, source=source)
            tool_calls += 1
            # Error recovery: try fallback sources
            fallbacks = SOURCE_FALLBACKS.get(source, [])
            for fallback in fallbacks:
                try:
                    tracer.log(agent_id, "fallback", original_source=source, fallback_source=fallback, query=query)
                    findings = _search_source(fallback, query, external_client)
                    all_findings.extend(findings)
                    queries_used.append(query)
                    tool_calls += 1
                    tracer.log(agent_id, "fallback_success", fallback_source=fallback, count=len(findings))
                    break
                except Exception:
                    continue
            else:
                continue

        evaluation = _evaluate_results(
            openai_client, query, len(findings), objective, model
        )
        tracer.log(agent_id, "evaluate", result=evaluation)
        tool_calls += 1

        if evaluation.get("sufficient", False):
            break

        next_query = evaluation.get("next_query")
        if next_query and next_query not in queries:
            queries.append(next_query)

    # Deduplicate findings by title (including across other agents)
    seen_titles: set[str] = set(existing_titles)
    unique_findings: list[Finding] = []
    for f in all_findings:
        key = f.title.lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            unique_findings.append(f)

    elapsed_seconds = time.time() - start_time
    summary_parts = [f.title for f in unique_findings[:5]]
    summary = f"Found {len(unique_findings)} results: " + "; ".join(summary_parts)

    result = SubagentResult(
        agent_id=agent_id,
        agent_type=source,
        subtask=objective,
        status="complete",
        findings=unique_findings,
        summary=summary,
        queries_used=queries_used,
        tool_calls_count=tool_calls,
        tokens_used=0,
        elapsed_seconds=elapsed_seconds,
    )

    store.store(result)
    tracer.log(agent_id, "complete", findings_count=len(unique_findings))
    return result
