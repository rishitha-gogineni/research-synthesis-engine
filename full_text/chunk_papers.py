"""Chunk extracted full-text papers for chunk-level retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/full_text_papers.json")
DEFAULT_OUTPUT = Path("data/full_text_chunks.json")
DEFAULT_MAX_WORDS = 450
DEFAULT_OVERLAP_WORDS = 75
WORD_PATTERN = re.compile(r"\S+")

SECTION_PATTERNS = [
    ("limitations", re.compile(r"\b(limitations?|future work|threats to validity)\b", re.IGNORECASE)),
    ("results", re.compile(r"\b(results?|findings?|performance|we find|we show|achieves?)\b", re.IGNORECASE)),
    ("experiments", re.compile(r"\b(experiments?|evaluation|benchmark|metrics?|dataset)\b", re.IGNORECASE)),
    ("methodology", re.compile(r"\b(methods?|methodology|approach|architecture|framework|algorithm|training)\b", re.IGNORECASE)),
    ("related_work", re.compile(r"\b(related work|background|prior work)\b", re.IGNORECASE)),
    ("introduction", re.compile(r"\b(introduction|abstract)\b", re.IGNORECASE)),
    ("conclusion", re.compile(r"\b(conclusion|concluding)\b", re.IGNORECASE)),
]


def load_full_text_papers(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_json_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [clean_json_text(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json_text(item) for key, item in value.items()}
    return value


def write_chunks(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json_text(chunks), indent=2, ensure_ascii=False), encoding="utf-8")


def paper_key(paper: dict[str, Any]) -> str:
    return str(paper.get("paper_id") or paper.get("title") or paper.get("pdf_url") or "paper")


def stable_chunk_id(paper: dict[str, Any], chunk_index: int) -> str:
    digest = hashlib.sha1(f"{paper_key(paper)}::{chunk_index}".encode("utf-8")).hexdigest()[:16]
    return f"chunk-{digest}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_words(text: str) -> list[str]:
    return [match.group(0) for match in WORD_PATTERN.finditer(normalize_text(text))]


def detect_section_hint(text: str) -> str:
    sample = text[:2500]
    for section, pattern in SECTION_PATTERNS:
        if pattern.search(sample):
            return section
    return "unknown"


def word_windows(words: list[str], max_words: int, overlap_words: int) -> list[list[str]]:
    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    if not words:
        return []

    windows: list[list[str]] = []
    step = max_words - overlap_words
    start = 0
    while start < len(words):
        window = words[start : start + max_words]
        if window:
            windows.append(window)
        if start + max_words >= len(words):
            break
        start += step
    return windows


def build_chunk(paper: dict[str, Any], chunk_index: int, total_chunks: int, words: list[str]) -> dict[str, Any]:
    text = " ".join(words).strip()
    return {
        "chunk_id": stable_chunk_id(paper, chunk_index),
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "topic": paper.get("topic"),
        "year": paper.get("year"),
        "citation_count": paper.get("citation_count", 0),
        "source_type": paper.get("source_type"),
        "pdf_url": paper.get("pdf_url"),
        "page_count": paper.get("page_count", 0),
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "word_count": len(words),
        "section_hint": detect_section_hint(text),
        "text": text,
    }


def chunk_paper(paper: dict[str, Any], *, max_words: int = DEFAULT_MAX_WORDS, overlap_words: int = DEFAULT_OVERLAP_WORDS) -> list[dict[str, Any]]:
    if paper.get("extraction_status") != "success":
        return []
    words = split_words(paper.get("text") or "")
    windows = word_windows(words, max_words=max_words, overlap_words=overlap_words)
    return [build_chunk(paper, index, len(windows), window) for index, window in enumerate(windows)]


def chunk_papers(
    papers: list[dict[str, Any]],
    *,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for paper in papers:
        chunks.extend(chunk_paper(paper, max_words=max_words, overlap_words=overlap_words))
    return chunks


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    by_topic: dict[str, int] = {}
    by_section: dict[str, int] = {}
    paper_ids = set()
    for chunk in chunks:
        by_topic[chunk.get("topic") or "Unknown"] = by_topic.get(chunk.get("topic") or "Unknown", 0) + 1
        by_section[chunk.get("section_hint") or "unknown"] = by_section.get(chunk.get("section_hint") or "unknown", 0) + 1
        paper_ids.add(chunk.get("paper_id") or chunk.get("title"))
    return {
        "chunks": len(chunks),
        "papers": len(paper_ids),
        "by_topic": by_topic,
        "by_section_hint": by_section,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    papers = load_full_text_papers(args.input)
    chunks = chunk_papers(papers, max_words=args.max_words, overlap_words=args.overlap_words)
    write_chunks(args.output, chunks)
    print(json.dumps(summarize_chunks(chunks), indent=2, ensure_ascii=False))
    print(f"Wrote {len(chunks)} full-text chunks to {args.output}")


if __name__ == "__main__":
    main()
