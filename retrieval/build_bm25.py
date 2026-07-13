"""Build a local BM25 sparse index over embedded paper text."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def normalize_title(title: str | None) -> str:
    return " ".join(tokenize(title or ""))


def load_embedded_records(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_corpus_text(record: dict[str, Any]) -> str:
    metadata = record["metadata"]
    return "\n".join(
        [
            record.get("embedding_text") or "",
            metadata.get("title") or record.get("title") or "",
            metadata.get("abstract") or "",
            metadata.get("main_contribution") or "",
            metadata.get("methodology") or "",
            metadata.get("dataset_used") or "",
            metadata.get("key_result") or "",
            metadata.get("limitations") or "",
        ]
    )


def build_bm25_artifact(records: list[dict[str, Any]]) -> dict[str, Any]:
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(build_corpus_text(record)) for record in records]
    bm25 = BM25Okapi(tokenized_corpus)
    papers = [
        {
            "paper_id": record.get("paper_id"),
            "title": record.get("title"),
            "topic": record.get("topic"),
            "year": record.get("year"),
            "citation_count": record.get("citation_count", 0),
            "metadata": record.get("metadata", {}),
        }
        for record in records
    ]
    return {
        "bm25": bm25,
        "tokenized_corpus": tokenized_corpus,
        "papers": papers,
    }


def save_bm25_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)


def load_bm25_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def search_bm25(
    artifact: dict[str, Any],
    query: str,
    top_k: int = 5,
    dedupe_titles: bool = True,
) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    scores = artifact["bm25"].get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    results: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for index in ranked_indices:
        paper = artifact["papers"][index]
        title_key = normalize_title(paper.get("title"))
        if dedupe_titles and title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        results.append({**paper, "score": float(scores[index])})
        if len(results) >= top_k:
            break

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/embedded_papers.json"))
    parser.add_argument("--output", type=Path, default=Path("data/bm25_index.pkl"))
    parser.add_argument("--query", default=None, help="Optional sanity-check query after building.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_embedded_records(args.input)
    artifact = build_bm25_artifact(records)
    save_bm25_artifact(args.output, artifact)
    print(f"BM25 indexing complete: {len(records)} papers -> {args.output}")

    if args.query:
        for result in search_bm25(artifact, args.query, top_k=args.top_k):
            print(f"{result['score']:.3f} | {result['title']} | {result['topic']}")


if __name__ == "__main__":
    main()
