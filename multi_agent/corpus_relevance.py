"""Corpus relevance pre-check — routes queries based on what's actually in the corpus.

Runs before the lead agent plans. Uses vector similarity to check if:
- A full-text (chunk) match exists → 1 subagent (local_corpus only)
- Only paper-level (abstract) match exists → 2 subagents (local_corpus + arxiv)
- No match at all → skip local_corpus, use external sources

This gives the lead agent grounded routing info instead of relying on
topical description alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


RELEVANCE_SCORE_THRESHOLD = 0.35
TOP_K_CHECK = 5


RelevanceState = Literal["full_text_match", "abstract_only", "no_match"]


@dataclass
class CorpusRelevance:
    state: RelevanceState
    matching_papers: list[dict[str, Any]] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    reason: str = ""


SearchFn = Callable[[str, int], Any]

_SCORE_FIELDS = ("blended_score", "rerank_score", "hybrid_score", "dense_score")


def best_score(item: Any) -> float:
    """Pick the most authoritative 0-1 relevance score available on a result.

    Prefers the post-rerank blended score, falling back through rerank, hybrid,
    and raw dense cosine. RetrievedPaper/RetrievedChunk have no single `.score`.
    """
    for field_name in _SCORE_FIELDS:
        value = getattr(item, field_name, None)
        if value is not None:
            return float(value)
    return 0.0


def check_corpus_relevance(
    query: str,
    *,
    searcher: SearchFn | None = None,
    threshold: float = RELEVANCE_SCORE_THRESHOLD,
) -> CorpusRelevance:
    """Determine what tier of corpus match exists for the query."""
    if searcher is None:
        from retrieval.unified_search import run_unified_search
        def _default_searcher(q: str, top_k: int) -> Any:
            return run_unified_search(q, top_k=top_k)
        searcher = _default_searcher

    try:
        response = searcher(query, TOP_K_CHECK)
    except Exception as exc:
        return CorpusRelevance(
            state="no_match",
            reason=f"Corpus check failed: {exc}",
        )

    chunk_results = getattr(response, "chunk_results", []) or []
    paper_results = getattr(response, "paper_results", []) or []

    strong_chunks = [c for c in chunk_results if best_score(c) >= threshold]
    strong_papers = [p for p in paper_results if best_score(p) >= threshold]

    if strong_chunks:
        matching = []
        topics: set[str] = set()
        seen: set[str] = set()
        for c in strong_chunks[:3]:
            title = getattr(c, "title", "") or ""
            if title in seen:
                continue
            seen.add(title)
            topic = getattr(c, "topic", "") or ""
            matching.append({
                "title": title,
                "year": getattr(c, "year", None),
                "score": round(best_score(c), 3),
                "level": "chunk",
            })
            if topic:
                topics.add(topic)
        return CorpusRelevance(
            state="full_text_match",
            matching_papers=matching,
            topics=sorted(topics),
            reason=f"Found {len(strong_chunks)} full-text chunks above threshold {threshold}.",
        )

    if strong_papers:
        matching = []
        topics: set[str] = set()
        seen: set[str] = set()
        for p in strong_papers[:3]:
            title = getattr(p, "title", "") or ""
            if title in seen:
                continue
            seen.add(title)
            topic = getattr(p, "topic", "") or ""
            matching.append({
                "title": title,
                "year": getattr(p, "year", None),
                "score": round(best_score(p), 3),
                "level": "paper",
            })
            if topic:
                topics.add(topic)
        return CorpusRelevance(
            state="abstract_only",
            matching_papers=matching,
            topics=sorted(topics),
            reason=f"Found {len(strong_papers)} papers at abstract level but no strong full-text chunks.",
        )

    top_chunk_score = max((best_score(c) for c in chunk_results), default=0.0)
    top_paper_score = max((best_score(p) for p in paper_results), default=0.0)
    top_score = max(top_chunk_score, top_paper_score)

    return CorpusRelevance(
        state="no_match",
        reason=f"No matches above threshold {threshold} (top score: {top_score:.2f}).",
    )


def format_relevance_context(relevance: CorpusRelevance) -> str:
    """Format the relevance result as context for the lead agent's prompt."""
    if relevance.state == "no_match":
        return (
            f"CORPUS PRE-CHECK: no_match. {relevance.reason} "
            f"→ SKIP local_corpus. Spawn arxiv/semantic_scholar/web subagents in parallel."
        )

    paper_lines = "\n".join(
        f"    - {p['title']} ({p.get('year', '?')}, score: {p['score']}, level: {p['level']})"
        for p in relevance.matching_papers
    )

    if relevance.state == "full_text_match":
        return (
            f"CORPUS PRE-CHECK: full_text_match. {relevance.reason}\n"
            f"Matching full-text papers:\n{paper_lines}\n"
            f"→ Spawn ONLY local_corpus subagent. Do NOT add external sources — "
            f"the corpus has enough depth to answer this."
        )

    # abstract_only
    return (
        f"CORPUS PRE-CHECK: abstract_only. {relevance.reason}\n"
        f"Matching papers (abstract only):\n{paper_lines}\n"
        f"→ Spawn local_corpus subagent AND arxiv subagent in parallel. "
        f"The corpus has the paper's abstract but not full text — arxiv can fetch the full paper."
    )
