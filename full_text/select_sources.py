"""Select a topic-balanced, high-citation full-text subset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/full_text_sources.json")
DEFAULT_OUTPUT = Path("data/full_text_selected.json")
DEFAULT_PER_TOPIC = 25
SOURCE_TYPE_PRIORITY = {
    "arxiv": 0,
    "openalex_oa": 1,
}


def load_sources(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_selected(path: Path, selected: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")


def source_key(source: dict[str, Any]) -> str:
    paper_id = source.get("paper_id")
    if paper_id:
        return f"id:{paper_id}"
    return f"title:{(source.get('title') or '').strip().lower()}"


def rank_source(source: dict[str, Any]) -> tuple[int, int, str]:
    citation_count = int(source.get("citation_count") or 0)
    source_priority = SOURCE_TYPE_PRIORITY.get(source.get("source_type"), 99)
    title = source.get("title") or ""
    return (-citation_count, source_priority, title.lower())


def available_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    available: list[dict[str, Any]] = []
    for source in sources:
        if not source.get("full_text_available") or not source.get("pdf_url"):
            continue
        key = source_key(source)
        if key in seen:
            continue
        seen.add(key)
        available.append(source)
    return available


def group_by_topic(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        grouped[source.get("topic") or "Unknown"].append(source)
    return dict(grouped)


def select_full_text_sources(
    sources: list[dict[str, Any]],
    *,
    per_topic: int = DEFAULT_PER_TOPIC,
    max_total: int | None = None,
) -> list[dict[str, Any]]:
    """Select high-citation full-text sources while preserving topic balance."""

    if per_topic <= 0:
        raise ValueError("per_topic must be greater than 0")
    if max_total is not None and max_total <= 0:
        raise ValueError("max_total must be greater than 0")

    grouped = group_by_topic(available_sources(sources))
    selected: list[dict[str, Any]] = []
    for topic in sorted(grouped):
        ranked = sorted(grouped[topic], key=rank_source)
        for topic_rank, source in enumerate(ranked[:per_topic], start=1):
            selected.append(
                {
                    **source,
                    "selected_for_full_text": True,
                    "topic_selection_rank": topic_rank,
                }
            )

    selected = sorted(selected, key=lambda item: (item.get("topic") or "", item["topic_selection_rank"]))
    if max_total is not None:
        selected = sorted(selected, key=rank_source)[:max_total]
        selected = sorted(selected, key=lambda item: (item.get("topic") or "", item.get("topic_selection_rank") or 0))

    for global_rank, source in enumerate(sorted(selected, key=rank_source), start=1):
        source["global_selection_rank"] = global_rank

    return selected


def summarize_selection(selected: list[dict[str, Any]]) -> dict[str, Any]:
    by_topic: dict[str, int] = {}
    by_source_type: dict[str, int] = {}
    citation_counts = [int(source.get("citation_count") or 0) for source in selected]

    for source in selected:
        topic = source.get("topic") or "Unknown"
        by_topic[topic] = by_topic.get(topic, 0) + 1
        source_type = source.get("source_type") or "unknown"
        by_source_type[source_type] = by_source_type.get(source_type, 0) + 1

    return {
        "selected": len(selected),
        "by_topic": by_topic,
        "by_source_type": by_source_type,
        "max_citation_count": max(citation_counts) if citation_counts else 0,
        "min_citation_count": min(citation_counts) if citation_counts else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-topic", type=int, default=DEFAULT_PER_TOPIC)
    parser.add_argument("--max-total", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = load_sources(args.input)
    selected = select_full_text_sources(sources, per_topic=args.per_topic, max_total=args.max_total)
    write_selected(args.output, selected)
    print(json.dumps(summarize_selection(selected), indent=2, ensure_ascii=False))
    print(f"Wrote {len(selected)} selected full-text sources to {args.output}")


if __name__ == "__main__":
    main()
