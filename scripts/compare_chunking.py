"""Compare chunking strategies on a single sample of text.

Usage:
    python -m scripts.compare_chunking

Reads scripts/sample_text.txt and shows how 6 different chunking strategies
carve it up. Meant as an eyeball comparison, not an eval - pick the strategy
that best preserves the kind of semantic boundaries your queries care about.
"""

from __future__ import annotations

import re
from pathlib import Path


SAMPLE_PATH = Path(__file__).parent / "sample_text.txt"


def read_sample() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8").strip()


def word_count(text: str) -> int:
    return len(text.split())


def preview(text: str, max_chars: int = 200) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def chunk_fixed_words(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    """Current RSE strategy - fixed word window with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
    return chunks


def chunk_fixed_chars(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """Fixed character count - respects nothing, just counts."""
    if not text:
        return []
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + size]
        if not chunk:
            break
        chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def chunk_by_sentence(text: str, sentences_per_chunk: int = 4) -> list[str]:
    """Split on sentence boundaries, group N sentences together."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks


def chunk_recursive(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """Recursive: split on paragraphs first, then sentences, then words.

    Mimics LangChain's RecursiveCharacterTextSplitter without the dependency.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, seps: list[str]) -> list[str]:
        if len(text) <= size:
            return [text]
        if not seps:
            return [text[i:i + size] for i in range(0, len(text), size - overlap)]
        sep = seps[0]
        parts = text.split(sep)
        chunks = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}{sep}{part}" if buffer else part
            if len(candidate) <= size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                if len(part) > size:
                    chunks.extend(_split(part, seps[1:]))
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            chunks.append(buffer)
        return chunks

    return _split(text, separators)


def chunk_sliding_window(text: str, size: int = 100, overlap: int = 50) -> list[str]:
    """Heavy overlap - for high context preservation."""
    return chunk_fixed_words(text, size=size, overlap=overlap)


def chunk_paragraph(text: str) -> list[str]:
    """Split on paragraph boundaries only - variable size, respects meaning."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paragraphs:
        return paragraphs
    # Fallback: single sentences act as pseudo-paragraphs if no blank lines
    return re.split(r"(?<=[.!?])\s+", text)


def summarize(name: str, chunks: list[str], config: str) -> dict:
    if not chunks:
        return {"name": name, "config": config, "count": 0}
    sizes = [word_count(c) for c in chunks]
    return {
        "name": name,
        "config": config,
        "count": len(chunks),
        "avg_words": sum(sizes) // len(sizes),
        "min_words": min(sizes),
        "max_words": max(sizes),
    }


def main() -> None:
    text = read_sample()
    total_words = word_count(text)
    total_chars = len(text)
    print(f"INPUT: {total_words} words, {total_chars} chars\n")
    print(f"Preview: {preview(text, 300)}\n")
    print("=" * 80)

    strategies = [
        ("Fixed word (RSE current)", "size=100, overlap=20", chunk_fixed_words(text, 100, 20)),
        ("Fixed char", "size=600, overlap=100", chunk_fixed_chars(text, 600, 100)),
        ("Sentence-based", "4 sentences per chunk", chunk_by_sentence(text, 4)),
        ("Recursive", "size=600, overlap=100", chunk_recursive(text, 600, 100)),
        ("Sliding window", "size=100, overlap=50", chunk_sliding_window(text, 100, 50)),
        ("Paragraph", "natural boundaries", chunk_paragraph(text)),
    ]

    summaries = []
    for name, config, chunks in strategies:
        print(f"\n=== {name} ({config}) ===")
        print(f"Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, start=1):
            wc = word_count(chunk)
            print(f"\n  Chunk {i} [{wc} words]")
            print(f"  {preview(chunk, 250)}")
        summaries.append(summarize(name, chunks, config))
        print()
        print("-" * 80)

    print("\n\n=== COMPARISON TABLE ===\n")
    print(f"{'Strategy':<28} {'Config':<28} {'#chunks':>8} {'avg':>6} {'min':>5} {'max':>5}")
    print("-" * 80)
    for s in summaries:
        if s["count"] == 0:
            continue
        print(
            f"{s['name']:<28} {s['config']:<28} {s['count']:>8} "
            f"{s['avg_words']:>6} {s['min_words']:>5} {s['max_words']:>5}"
        )


if __name__ == "__main__":
    main()
