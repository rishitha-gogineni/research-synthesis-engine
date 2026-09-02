"""Shared test harness for handoff-contract and fault-injection tests.

Centralizes the two things every test in that family needs: a mock OpenAI
client that returns a scripted sequence of JSON responses (one per call, so
multi-call flows like run_subagent's search->evaluate loop can be exercised),
and prebuilt FindingsStore fixtures modeling handoffs that are realistic but
imperfect (the kinds of things a live external API or a race between
subagents actually produces). Keeping these here instead of copy-pasted
across test files means a change to the mock response shape is a one-line
fix instead of an N-file find-and-replace.
"""

from __future__ import annotations

import json
from typing import Any, Iterable
from unittest.mock import MagicMock

from multi_agent.findings_store import Finding, FindingsStore, SubagentResult


def make_llm_client(*responses: dict[str, Any] | str) -> MagicMock:
    """Build a mock OpenAI client returning each response in order across
    successive chat.completions.create() calls (JSON-encoded if given a dict).

    Calling this with zero responses configured is a deliberate way to assert
    an LLM call must NOT happen (the mock's side_effect list is empty, so any
    call raises StopIteration).
    """
    client = MagicMock()
    mock_responses = []
    for r in responses:
        content = json.dumps(r) if isinstance(r, dict) else r
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.usage = None
        mock_responses.append(resp)
    client.chat.completions.create.side_effect = mock_responses
    return client


def corrupted_store(scenario: str) -> FindingsStore:
    """Build a FindingsStore with a specific real-world handoff imperfection.

    Scenarios:
    - "empty_content": a completed finding with empty title/content (a source
      returned a hit with no extractable text).
    - "missing_metadata": a completed finding with an empty metadata dict
      (fields like citation_count/published_date absent, not just falsy).
    - "partial_status": one agent completed with a finding, one failed
      outright — the common real shape of a partially-successful run.
    - "duplicate_titles": two different sources surface the "same" paper with
      slightly different title casing/whitespace, each with distinct content.
    """
    store = FindingsStore()
    if scenario == "empty_content":
        store.store(SubagentResult(
            agent_id="arxiv_1", agent_type="arxiv", subtask="t",
            status="complete",
            findings=[Finding(source="arxiv", title="", content="")],
        ))
    elif scenario == "missing_metadata":
        store.store(SubagentResult(
            agent_id="arxiv_1", agent_type="arxiv", subtask="t",
            status="complete",
            findings=[Finding(source="arxiv", title="Paper", content="Abstract", metadata={})],
        ))
    elif scenario == "partial_status":
        store.store(SubagentResult(
            agent_id="arxiv_1", agent_type="arxiv", subtask="t1", status="complete",
            findings=[Finding(source="arxiv", title="Paper A", content="Content A")],
        ))
        store.store(SubagentResult(
            agent_id="web_1", agent_type="web", subtask="t2", status="failed",
            findings=[], error="timeout",
        ))
    elif scenario == "duplicate_titles":
        store.store(SubagentResult(
            agent_id="arxiv_1", agent_type="arxiv", subtask="t1", status="complete",
            findings=[Finding(source="arxiv", title="Attention Is All You Need", content="C1")],
        ))
        store.store(SubagentResult(
            agent_id="semantic_scholar_1", agent_type="semantic_scholar", subtask="t2", status="complete",
            findings=[Finding(source="semantic_scholar", title="attention is all you need ", content="C2")],
        ))
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return store


class FailingExternalClient:
    """Stand-in for ExternalSearchClient whose search_* methods fail for
    configured sources and return canned papers for the rest. Used to test
    that SOURCE_FALLBACKS (multi_agent/subagent.py) is actually exercised
    end-to-end, not just present in a dict.
    """

    def __init__(
        self,
        fail_sources: Iterable[str],
        papers_by_source: dict[str, list] | None = None,
    ) -> None:
        self._fail = set(fail_sources)
        self._papers = papers_by_source or {}

    def _maybe_fail(self, source: str) -> None:
        if source in self._fail:
            raise RuntimeError(f"simulated failure for {source}")

    def search_arxiv(self, query: str, max_results: int = 5) -> list:
        self._maybe_fail("arxiv")
        return self._papers.get("arxiv", [])

    def search_semantic_scholar(self, query: str, max_results: int = 5) -> list:
        self._maybe_fail("semantic_scholar")
        return self._papers.get("semantic_scholar", [])

    def search_tavily(self, query: str, max_results: int = 5) -> list:
        self._maybe_fail("web")
        return self._papers.get("web", [])
