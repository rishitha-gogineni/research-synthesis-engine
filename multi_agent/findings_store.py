"""Shared storage for subagent findings.

Subagents write findings here instead of passing through the lead agent's
context, following the 'filesystem output' pattern to minimize the
'game of telephone' effect.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Finding:
    source: str
    title: str
    content: str
    url: str | None = None
    relevance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentResult:
    agent_id: str
    agent_type: str
    subtask: str
    status: str  # "complete", "partial", "failed"
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    queries_used: list[str] = field(default_factory=list)
    tool_calls_count: int = 0
    tokens_used: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


class FindingsStore:
    """In-memory store for subagent results."""

    def __init__(self) -> None:
        self._results: dict[str, SubagentResult] = {}
        self._created_at = time.time()

    def create_agent_id(self, agent_type: str) -> str:
        short_id = uuid.uuid4().hex[:8]
        return f"{agent_type}_{short_id}"

    def store(self, result: SubagentResult) -> None:
        self._results[result.agent_id] = result

    def get(self, agent_id: str) -> SubagentResult | None:
        return self._results.get(agent_id)

    def get_all(self) -> list[SubagentResult]:
        return list(self._results.values())

    def get_completed(self) -> list[SubagentResult]:
        return [r for r in self._results.values() if r.status == "complete"]

    def get_all_findings(self) -> list[Finding]:
        findings = []
        for result in self.get_completed():
            findings.extend(result.findings)
        return findings

    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self._results.values())

    def summary(self) -> dict[str, Any]:
        results = self.get_all()
        return {
            "total_agents": len(results),
            "completed": sum(1 for r in results if r.status == "complete"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "total_findings": len(self.get_all_findings()),
            "total_tokens": self.total_tokens(),
            "elapsed_seconds": time.time() - self._created_at,
        }
