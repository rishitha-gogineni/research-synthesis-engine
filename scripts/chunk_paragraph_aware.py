"""Task 5: Paragraph + section-aware chunking with size guardrails.

Reads data/full_text_papers_v2.json (PyMuPDF-extracted, section-tagged blocks)
and produces data/full_text_chunks_v2.json with paragraph-boundary-respecting
chunks. Each chunk carries a section_hint compatible with v1's eval fixture.

Chunking strategy:
- Group blocks by detected section
- Within each section, greedily pack paragraphs into chunks:
    - Target 300 words, max 450 (v1 default was 450 fixed)
    - Merge paragraphs < 50 words into the next chunk
    - Never split a paragraph mid-way (respect natural boundaries)
- Skip references, acknowledgments (noise for retrieval)
- Map v2 section names to v1 section_hint values for eval compatibility

This is drop-in compatible with the RSE eval fixture — chunks match on
paper_id since v1's eval accepts both chunk_id and paper_id for hits.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


INPUT = Path("data/full_text_papers_v2.json")
OUTPUT = Path("data/full_text_chunks_v2.json")

TARGET_WORDS = 300
MAX_WORDS = 450
MIN_WORDS = 50
SKIP_SECTIONS = {"references", "acknowledgments", "appendix"}

V2_TO_V1_SECTION = {
    "abstract": "introduction",
    "introduction": "introduction",
    "related_work": "related_work",
    "methods": "methodology",
    "experiments": "experiments",
    "results": "results",
    "discussion": "results",
    "limitations": "limitations",
    "conclusion": "conclusion",
    "front_matter": "introduction",
}


def word_count(text: str) -> int:
    return len(text.split())


def stable_chunk_id(paper_id: str, chunk_index: int) -> str:
    digest = hashlib.sha1(f"{paper_id}::{chunk_index}".encode("utf-8")).hexdigest()[:16]
    return f"chunk-{digest}"


def content_fingerprint(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text).lower().strip().encode("utf-8")).hexdigest()


def pack_paragraphs(paragraphs: list[str], target: int, max_w: int, min_w: int) -> list[str]:
    """Greedy paragraph packer.

    Fills a buffer until it hits target words, then closes it and starts a new
    one. If a single paragraph exceeds max_words, it becomes its own chunk
    (respecting the boundary — we do NOT split it further).
    """
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = word_count(para)

        if para_words >= max_w:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffer_words = [], 0
            chunks.append(para)
            continue

        if buffer_words + para_words > max_w and buffer_words >= min_w:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_words = [para], para_words
            continue

        buffer.append(para)
        buffer_words += para_words

        if buffer_words >= target:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_words = [], 0

    if buffer:
        text = "\n\n".join(buffer)
        if word_count(text) < min_w and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + text
        else:
            chunks.append(text)

    return chunks


def chunk_paper(paper: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = paper.get("blocks") or []
    paper_id = paper["paper_id"]

    grouped: dict[str, list[str]] = {}
    section_order: list[str] = []
    for block in blocks:
        section = block.get("section") or "front_matter"
        if section in SKIP_SECTIONS:
            continue
        if section not in grouped:
            grouped[section] = []
            section_order.append(section)
        grouped[section].append(block["text"])

    all_chunks: list[dict[str, Any]] = []
    chunk_idx = 0
    for section in section_order:
        paragraphs = grouped[section]
        v1_section = V2_TO_V1_SECTION.get(section, "unknown")
        packed = pack_paragraphs(paragraphs, TARGET_WORDS, MAX_WORDS, MIN_WORDS)
        for chunk_text in packed:
            all_chunks.append({
                "chunk_id": stable_chunk_id(paper_id, chunk_idx),
                "paper_id": paper_id,
                "title": paper.get("title"),
                "topic": paper.get("topic"),
                "year": paper.get("year"),
                "citation_count": paper.get("citation_count", 0),
                "arxiv_id": paper.get("arxiv_id"),
                "chunk_index": chunk_idx,
                "word_count": word_count(chunk_text),
                "section_hint": v1_section,
                "section_v2_raw": section,
                "text": chunk_text,
            })
            chunk_idx += 1

    for chunk in all_chunks:
        chunk["total_chunks"] = len(all_chunks)

    return all_chunks


def deduplicate(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    dropped = 0
    for c in chunks:
        fp = content_fingerprint(c["text"])
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        out.append(c)
    return out, dropped


def main() -> None:
    papers = json.loads(INPUT.read_text())
    print(f"Chunking {len(papers)} papers...")

    all_chunks: list[dict[str, Any]] = []
    for paper in papers:
        all_chunks.extend(chunk_paper(paper))

    deduped, dropped = deduplicate(all_chunks)
    OUTPUT.write_text(json.dumps(deduped, indent=2, ensure_ascii=False))

    print(f"\n=== CHUNKING SUMMARY ===")
    print(f"Total chunks (raw)    : {len(all_chunks):,}")
    print(f"After dedup           : {len(deduped):,} (dropped {dropped})")
    print(f"Papers with chunks    : {len({c['paper_id'] for c in deduped})}")

    word_counts = [c["word_count"] for c in deduped]
    if word_counts:
        print(f"Avg words/chunk       : {sum(word_counts) / len(word_counts):.0f}")
        print(f"Min/Max words         : {min(word_counts)} / {max(word_counts)}")

    section_counts: dict[str, int] = {}
    for c in deduped:
        section_counts[c["section_hint"]] = section_counts.get(c["section_hint"], 0) + 1
    print(f"\n=== SECTION DISTRIBUTION ===")
    for sec, cnt in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {sec:<20} {cnt:>5} ({cnt/len(deduped)*100:.0f}%)")

    print(f"\nWritten to: {OUTPUT}")


if __name__ == "__main__":
    main()
