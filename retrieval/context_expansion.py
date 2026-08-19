"""Context expansion and paper evidence aggregation for route-aware RAG."""

from __future__ import annotations

from typing import Any

from retrieval.corpus_index import CorpusIndex


DEFAULT_PARENT_CONTEXT_WINDOW = 1
DEFAULT_PARENT_CONTEXT_TOP_N = 3
DEFAULT_PARENT_CONTEXT_MAX_WORDS = 900


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _candidate_score(candidate: dict[str, Any]) -> float:
    """Return one comparable evidence score without changing existing scores."""

    for field in ("blended_score", "rerank_score", "hybrid_score", "dense_score", "sparse_score"):
        value = candidate.get(field)
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                continue
    return 0.0


def _bounded_context(anchor: str, before: list[str], after: list[str], max_words: int) -> str:
    anchor_words = (anchor or "").split()
    if max_words <= 0 or len(anchor_words) <= max_words:
        anchor_words = anchor_words[:max_words]
    else:
        return " ".join(anchor_words[:max_words])

    remaining = max(0, max_words - len(anchor_words))
    before_budget = remaining // 2
    before_words = before[-before_budget:] if before_budget else []
    after_words = after[: remaining - len(before_words)]
    return " ".join([*before_words, *anchor_words, *after_words]).strip()


def expand_chunk_context(
    candidates: list[dict[str, Any]],
    corpus_index: CorpusIndex | None,
    *,
    top_n: int = DEFAULT_PARENT_CONTEXT_TOP_N,
    window: int = DEFAULT_PARENT_CONTEXT_WINDOW,
    max_words: int = DEFAULT_PARENT_CONTEXT_MAX_WORDS,
) -> list[dict[str, Any]]:
    """Attach neighboring chunks to the strongest retrieved anchors.

    Chunk IDs remain the original anchor IDs, so retrieval evaluation and
    citations stay stable. Only the text supplied to synthesis is expanded.
    """

    if not candidates or corpus_index is None or top_n <= 0 or window < 0:
        return candidates

    expanded = [dict(candidate) for candidate in candidates]
    for candidate in expanded[:top_n]:
        paper_id = str(candidate.get("paper_id") or "").strip()
        if not paper_id:
            continue
        rows = corpus_index.chunks_by_paper_id.get(paper_id, [])
        if not rows:
            continue

        anchor_id = str(candidate.get("chunk_id") or "").strip()
        anchor_index = _safe_int(candidate.get("chunk_index"))
        if anchor_index is None and anchor_id:
            for row in rows:
                if str(row.get("chunk_id") or "").strip() == anchor_id:
                    anchor_index = _safe_int(row.get("chunk_index"))
                    break
        if anchor_index is None:
            continue

        selected = [
            row for row in rows
            if (_safe_int(row.get("chunk_index")) is not None
                and abs(_safe_int(row.get("chunk_index")) - anchor_index) <= window)
        ]
        if len(selected) <= 1:
            continue
        selected.sort(key=lambda row: _safe_int(row.get("chunk_index")) or 0)

        context_ids = [str(row.get("chunk_id") or "") for row in selected if row.get("chunk_id")]
        anchor_text = str(candidate.get("text") or "")
        before: list[str] = []
        after: list[str] = []
        for row in selected:
            row_index = _safe_int(row.get("chunk_index"))
            row_text = str(row.get("text") or "")
            if row_index is None or not row_text:
                continue
            if row_index < anchor_index:
                before.extend(row_text.split())
            elif row_index > anchor_index:
                after.extend(row_text.split())

        candidate["text"] = _bounded_context(anchor_text, before, after, max_words)
        candidate["matched_by"] = sorted(set(candidate.get("matched_by", []) + ["parent_context"]))
        breakdown = dict(candidate.get("score_breakdown") or {})
        breakdown["parent_context"] = {
            "anchor_chunk_id": anchor_id,
            "context_chunk_ids": context_ids,
            "window": window,
        }
        candidate["score_breakdown"] = breakdown
    return expanded


def aggregate_paper_evidence(
    paper_candidates: list[dict[str, Any]],
    chunk_candidates: list[dict[str, Any]],
    corpus_index: CorpusIndex | None = None,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Use chunk evidence to strengthen and, when possible, backfill papers.

    Existing paper order remains the dominant signal. Chunk coverage provides a
    small, fixed-scale bonus so one noisy chunk cannot overwhelm the retriever.
    """

    evidence: dict[str, dict[str, Any]] = {}
    for chunk in chunk_candidates:
        paper_id = str(chunk.get("paper_id") or "").strip()
        if not paper_id:
            continue
        row = evidence.setdefault(paper_id, {"count": 0, "score": 0.0, "chunk_ids": []})
        row["count"] += 1
        row["score"] = max(row["score"], _candidate_score(chunk))
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in row["chunk_ids"]:
            row["chunk_ids"].append(chunk_id)

    enriched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, paper in enumerate(paper_candidates):
        candidate = dict(paper)
        paper_id = str(candidate.get("paper_id") or "").strip()
        if paper_id:
            seen.add(paper_id)
        info = evidence.get(paper_id, {"count": 0, "score": 0.0, "chunk_ids": []})
        candidate["_evidence_rank"] = (1.0 / (rank + 1)) + min(0.08, 0.04 * info["count"]) + (0.04 * min(1.0, info["score"]))
        breakdown = dict(candidate.get("score_breakdown") or {})
        breakdown["chunk_evidence"] = {
            "chunk_count": info["count"],
            "chunk_ids": info["chunk_ids"],
            "max_score": round(info["score"], 6),
        }
        candidate["score_breakdown"] = breakdown
        if info["count"]:
            candidate["matched_by"] = sorted(set(candidate.get("matched_by", []) + ["chunk_evidence"]))
        enriched.append(candidate)

    if corpus_index is not None:
        for paper_id, info in evidence.items():
            if paper_id in seen:
                continue
            candidate = corpus_index.paper_candidate(paper_id, ["chunk_evidence_backfill", "corpus_index"])
            if candidate is None:
                continue
            candidate["_evidence_rank"] = min(0.12, 0.04 * info["count"]) + (0.04 * min(1.0, info["score"]))
            candidate["score_breakdown"] = {
                "chunk_evidence": {
                    "chunk_count": info["count"],
                    "chunk_ids": info["chunk_ids"],
                    "max_score": round(info["score"], 6),
                }
            }
            enriched.append(candidate)

    enriched.sort(key=lambda item: item.get("_evidence_rank", 0.0), reverse=True)
    for candidate in enriched:
        candidate.pop("_evidence_rank", None)
    return enriched[:top_k] if top_k is not None else enriched
