"""Structured logging and tracing for multi-agent research system."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class TraceEvent:
    timestamp: float
    agent_id: str
    event_type: str  # "plan", "search", "evaluate", "complete", "error"
    data: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects structured trace events for observability and debugging."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._start_time = time.time()

    def log(
        self,
        agent_id: str,
        event_type: str,
        **data: Any,
    ) -> None:
        event = TraceEvent(
            timestamp=time.time() - self._start_time,
            agent_id=agent_id,
            event_type=event_type,
            data=data,
        )
        self._events.append(event)

    def get_events(self, agent_id: str | None = None) -> list[TraceEvent]:
        if agent_id is None:
            return list(self._events)
        return [e for e in self._events if e.agent_id == agent_id]

    def summary(self) -> dict[str, Any]:
        agents = set(e.agent_id for e in self._events)
        return {
            "total_events": len(self._events),
            "agents_involved": list(agents),
            "elapsed_seconds": time.time() - self._start_time,
            "events_by_type": self._count_by_type(),
            "total_tokens": self._sum_tokens(),
        }

    def _sum_tokens(self) -> int:
        return sum(
            e.data.get("total_tokens", 0)
            for e in self._events
            if e.event_type == "llm_usage"
        )

    def _count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts

    def to_json(self) -> str:
        return json.dumps(
            [asdict(e) for e in self._events],
            indent=2,
            default=str,
        )
