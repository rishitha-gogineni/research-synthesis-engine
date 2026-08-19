"""External-tool dispatch, normalization, deduplication, and fail-soft handling."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any, Iterable
from agentic.external import DEFAULT_EXTERNAL_CLIENT, ExternalPaper, ExternalSearchClient, ExternalSearchError

@dataclass(frozen=True)
class ExternalSearchResponse:
    query: str
    results: list[dict[str, Any]]
    sources: list[str]
    warnings: list[str]
    def as_dict(self) -> dict[str, Any]:
        return {"query": self.query, "results": self.results, "sources": self.sources, "warnings": self.warnings}

def _dedupe_key(paper: ExternalPaper) -> str:
    value = paper.url or paper.paper_id or paper.title
    return re.sub(r"[^a-z0-9]+", "", value.lower())

def deduplicate_papers(papers: Iterable[ExternalPaper]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for paper in papers:
        item = paper.as_dict()
        key = _dedupe_key(paper)
        if not key: continue
        if key not in merged:
            merged[key] = item
            continue
        current = merged[key]
        current["source"] = ",".join(sorted(set((current.get("source") or "").split(",") + [paper.source])))
        if len(item.get("abstract") or "") > len(current.get("abstract") or ""):
            current["abstract"] = item["abstract"]
        if not current.get("url") and item.get("url"): current["url"] = item["url"]
    return list(merged.values())

def run_external_search(query: str, *, sources: Iterable[str] = ("arxiv", "semantic_scholar", "tavily"), max_results: int = 5, client: ExternalSearchClient | None = None) -> ExternalSearchResponse:
    query = " ".join(query.split())
    if not query: raise ValueError("query must not be empty")
    client = client or DEFAULT_EXTERNAL_CLIENT
    all_papers: list[ExternalPaper] = []
    used: list[str] = []
    warnings: list[str] = []
    for source in sources:
        method = getattr(client, f"search_{source}", None)
        if method is None:
            warnings.append(f"{source}: unsupported external source")
            continue
        try:
            all_papers.extend(method(query, max_results))
            used.append(source)
        except (ExternalSearchError, ValueError) as exc:
            warnings.append(f"{source}: {exc}")
    return ExternalSearchResponse(query, deduplicate_papers(all_papers), used, warnings)
