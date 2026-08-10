"""Shared metadata filters for paper and full-text retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalFilters:
    """Query-time filters applied before ranking whenever the backend supports them."""

    topics: tuple[str, ...] = ()
    year_min: int | None = None
    year_max: int | None = None
    full_text_only: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        topics: list[str] | tuple[str, ...] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        full_text_only: bool = False,
    ) -> "RetrievalFilters":
        return cls(
            topics=tuple(dict.fromkeys(item.strip() for item in (topics or ()) if item and item.strip())),
            year_min=year_min,
            year_max=year_max,
            full_text_only=full_text_only,
        )

    @property
    def active(self) -> bool:
        return bool(self.topics or self.year_min is not None or self.year_max is not None or self.full_text_only)

    def matches(self, payload: dict[str, Any]) -> bool:
        topic = payload.get("topic")
        if self.topics and topic not in self.topics:
            return False
        year = payload.get("year")
        try:
            year_value = int(year) if year is not None else None
        except (TypeError, ValueError):
            year_value = None
        if self.year_min is not None and (year_value is None or year_value < self.year_min):
            return False
        if self.year_max is not None and (year_value is None or year_value > self.year_max):
            return False
        return True

    def to_qdrant_filter(self) -> Any | None:
        if not self.active or (self.full_text_only and not self.topics and self.year_min is None and self.year_max is None):
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

        conditions = []
        if self.topics:
            topic_match = MatchValue(value=self.topics[0]) if len(self.topics) == 1 else MatchAny(any=list(self.topics))
            conditions.append(FieldCondition(key="topic", match=topic_match))
        if self.year_min is not None or self.year_max is not None:
            conditions.append(
                FieldCondition(
                    key="year",
                    range=Range(gte=self.year_min, lte=self.year_max),
                )
            )
        return Filter(must=conditions) if conditions else None
