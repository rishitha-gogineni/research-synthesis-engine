"""Optional hybrid BM25 retrieval over full-text chunks."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from retrieval.build_bm25 import tokenize

DEFAULT_CHUNK_INPUT = Path("data/full_text_chunks.json")
DEFAULT_CHUNK_BM25_PATH = Path("data/chunk_bm25_index.pkl")


def chunk_text(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {}) or {}
    return "\n".join(
        str(value or "")
        for value in (
            record.get("title"),
            record.get("topic"),
            record.get("section_hint"),
            metadata.get("text") or record.get("text"),
        )
    )


def build_chunk_bm25_artifact(records: list[dict[str, Any]]) -> dict[str, Any]:
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(chunk_text(record)) for record in records]
    return {
        "bm25": BM25Okapi(tokenized_corpus),
        "tokenized_corpus": tokenized_corpus,
        "chunks": records,
    }


def save_chunk_bm25_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)


def load_chunk_bm25_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def chunk_candidate(record: dict[str, Any], score: float) -> dict[str, Any]:
    metadata = record.get("metadata", {}) or {}
    return {
        "chunk_id": record.get("chunk_id"),
        "paper_id": record.get("paper_id"),
        "title": record.get("title") or "Untitled paper",
        "topic": record.get("topic") or "Unknown topic",
        "year": record.get("year"),
        "citation_count": record.get("citation_count", 0),
        "chunk_index": record.get("chunk_index"),
        "total_chunks": record.get("total_chunks"),
        "section_hint": record.get("section_hint"),
        "word_count": record.get("word_count"),
        "text": metadata.get("text") or record.get("text") or "",
        "pdf_url": metadata.get("pdf_url"),
        "source_type": metadata.get("source_type"),
        "page_count": metadata.get("page_count"),
        "dense_score": None,
        "sparse_score": float(score),
        "matched_by": ["chunk_sparse"],
    }


def search_chunk_bm25(
    artifact: dict[str, Any],
    query: str,
    top_k: int,
    *,
    retrieval_filters: Any | None = None,
) -> list[dict[str, Any]]:
    scores = artifact["bm25"].get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    results = []
    for index in ranked_indices:
        record = artifact["chunks"][index]
        if retrieval_filters is not None and not retrieval_filters.matches(record):
            continue
        results.append(chunk_candidate(record, scores[index]))
        if len(results) >= top_k:
            break
    return results



def chunk_query_prefers_sparse(query: str) -> bool:
    """Use rank fusion when exact factual or named-entity matching matters."""

    if re.search(
        r"\b(?:accuracy|precision|recall|f1|bleu|rouge|"
        r"percentage)\b|\bhow much\b|\bhow fast\b",
        query,
        re.IGNORECASE,
    ):
        return True
    generic_acronyms = {"AI", "LLM", "RAG", "QA"}
    named_tokens = re.findall(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b", query)
    return any(token not in generic_acronyms for token in named_tokens)


def merge_chunk_candidates(
    dense_candidates: list[dict[str, Any]],
    sparse_candidates: list[dict[str, Any]],
    top_k: int,
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
    fusion_method: str = "weighted",
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if fusion_method not in {"weighted", "rrf"}:
        raise ValueError(f"unknown fusion_method: {fusion_method!r}")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")

    merged: dict[str, dict[str, Any]] = {}
    for candidate in dense_candidates + sparse_candidates:
        key = str(candidate.get("chunk_id") or candidate.get("paper_id") or candidate.get("title"))
        current = merged.setdefault(key, dict(candidate))
        for field, value in candidate.items():
            if current.get(field) is None and value is not None:
                current[field] = value
        current["matched_by"] = sorted(set(current.get("matched_by", []) + candidate.get("matched_by", [])))

    if fusion_method == "rrf":
        scores: dict[str, float] = {key: 0.0 for key in merged}
        for rank, candidate in enumerate(dense_candidates, start=1):
            key = str(candidate.get("chunk_id") or candidate.get("paper_id") or candidate.get("title"))
            scores[key] += 1.0 / (rrf_k + rank)
        for rank, candidate in enumerate(sparse_candidates, start=1):
            key = str(candidate.get("chunk_id") or candidate.get("paper_id") or candidate.get("title"))
            scores[key] += 1.0 / (rrf_k + rank)
        for key, candidate in merged.items():
            candidate["hybrid_score"] = round(scores[key], 6)
            candidate["fusion_method"] = "rrf"
    else:
        max_dense = max((float(item.get("dense_score") or 0.0) for item in merged.values()), default=0.0)
        max_sparse = max((float(item.get("sparse_score") or 0.0) for item in merged.values()), default=0.0)
        for candidate in merged.values():
            dense = float(candidate.get("dense_score") or 0.0) / max_dense if max_dense else 0.0
            sparse = float(candidate.get("sparse_score") or 0.0) / max_sparse if max_sparse else 0.0
            candidate["hybrid_score"] = round(dense_weight * dense + sparse_weight * sparse, 6)
            candidate["fusion_method"] = "weighted"

    return sorted(
        merged.values(),
        key=lambda item: (item["hybrid_score"], item.get("dense_score") or 0.0, item.get("sparse_score") or 0.0),
        reverse=True,
    )[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CHUNK_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHUNK_BM25_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    artifact = build_chunk_bm25_artifact([record for record in records if isinstance(record, dict)])
    save_chunk_bm25_artifact(args.output, artifact)
    print(f"Chunk BM25 indexing complete: {len(artifact['chunks'])} chunks -> {args.output}")


if __name__ == "__main__":
    main()
