"""Semantic chunker (v3) — sentence-level topic-shift detection.

Reads data/full_text_papers_v2.json (already extracted, section-tagged)
and produces data/full_text_chunks_v3.json with chunks that respect:
1. Section boundaries (chunk within a section, never across)
2. Topic-shift boundaries within a section (cosine drop between adjacent
   sentence embeddings falls below the 20th percentile)
3. Size guardrails (min 50, target 300, max 500 words)
4. Overlap: last 2 sentences of chunk N carry into start of chunk N+1

Differences from v2 (paragraph):
- Uses OpenAI text-embedding-3-small to detect semantic boundaries
- Adds 2-sentence overlap (v2 had zero overlap)
- Keeps appendix content (v2 skipped it)
- Deduplicates papers at the paper level (v2 dedup'd at chunk level,
  which caused 4 papers to end up with zero chunks)

Costs ~$0.35 to run (embedding ~100k sentences with -small model).
Runtime: ~25-30 min due to OpenAI rate limits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

INPUT = Path("data/full_text_papers_v2.json")
OUTPUT = Path("data/full_text_chunks_v3.json")

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 128
TARGET_WORDS = 300
MAX_WORDS = 500
MIN_WORDS = 50
OVERLAP_SENTENCES = 2
BOUNDARY_PERCENTILE = 20  # split where similarity falls below the 20th pct

# Only skip references + acknowledgments; keep appendix (v2 was too aggressive)
SKIP_SECTIONS = {"references", "acknowledgments"}

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
    "appendix": "methodology",  # appendix usually has methods details
}

# Sentence splitter — dodge Python's fixed-width lookbehind restriction by
# temporarily masking abbreviation dots, splitting, then unmasking.
_ABBREVIATIONS = (
    "e.g.", "i.e.", "cf.", "vs.", "etc.", "Fig.", "Tab.", "Eq.", "Ref.",
    "al.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sec.", "Ch.", "St.",
    "Jr.", "Sr.", "No.", "Vol.", "pp.",
)
_DOT_PLACEHOLDER = "\x00DOT\x00"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")


def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting common academic abbreviations."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    masked = text
    for abbr in _ABBREVIATIONS:
        masked = masked.replace(abbr, abbr.replace(".", _DOT_PLACEHOLDER))
    parts = SENTENCE_SPLIT_RE.split(masked)
    return [p.replace(_DOT_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


def stable_chunk_id(paper_id: str, chunk_index: int) -> str:
    digest = hashlib.sha1(f"{paper_id}::{chunk_index}::v3".encode("utf-8")).hexdigest()[:16]
    return f"chunk-{digest}"


def content_fingerprint(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text).lower().strip().encode("utf-8")).hexdigest()


def dedupe_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only one copy of each paper by title, preferring more content."""
    by_title: dict[str, dict[str, Any]] = {}
    for p in papers:
        key = re.sub(r"\s+", " ", (p.get("title") or "").lower().strip())[:80]
        if not key:
            continue
        existing = by_title.get(key)
        if existing is None or p.get("extraction_char_count", 0) > existing.get("extraction_char_count", 0):
            by_title[key] = p
    return list(by_title.values())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def embed_batch(client: OpenAI, texts: list[str], retries: int = 3) -> np.ndarray:
    """Embed a batch of texts with retry on transient failures."""
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return np.array([d.embedding for d in resp.data])
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  embed error ({exc}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def find_boundaries(similarities: list[float], target_boundaries: int) -> set[int]:
    """Return sentence indices AFTER which a chunk break should be inserted.

    Uses the percentile of similarity drops — the lowest-similarity transitions
    become the break points. Guarantees at least `target_boundaries` breaks.
    """
    if not similarities:
        return set()
    if target_boundaries <= 0:
        return set()
    # Similarities align with the gap AFTER sentence i (similarities[i] is between
    # sentence i and sentence i+1). Lower similarity → bigger topic shift → good break.
    threshold = float(np.percentile(similarities, BOUNDARY_PERCENTILE))
    # Get all indices with similarity <= threshold, ranked lowest-first
    candidates = sorted(range(len(similarities)), key=lambda i: similarities[i])
    return set(candidates[:target_boundaries])


def chunk_section(
    client: OpenAI,
    text: str,
    section_hint: str,
) -> list[str]:
    """Break a single section into semantic chunks with overlap.

    Returns list of chunk texts.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    # Embed all sentences
    embeddings_list: list[np.ndarray] = []
    for i in range(0, len(sentences), EMBED_BATCH):
        batch = sentences[i:i + EMBED_BATCH]
        embeddings_list.append(embed_batch(client, batch))
    embeddings = np.vstack(embeddings_list)

    # Similarity between adjacent sentences
    similarities = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(sentences) - 1)
    ]

    # Estimate how many boundaries we need (target ~300 words per chunk)
    total_words = sum(word_count(s) for s in sentences)
    target_chunk_count = max(1, total_words // TARGET_WORDS)
    boundaries = find_boundaries(similarities, max(0, target_chunk_count - 1))

    # Walk sentences, close chunk at boundaries or when hitting max_words
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    prev_tail: list[str] = []  # overlap from previous chunk

    def close_buffer() -> None:
        nonlocal buffer, buffer_words, prev_tail
        if not buffer:
            return
        text = " ".join(buffer)
        if word_count(text) < MIN_WORDS and chunks:
            # Too small — merge into previous chunk
            chunks[-1] = chunks[-1] + " " + text
        else:
            chunks.append(text)
        # Prepare overlap for next chunk
        prev_tail = buffer[-OVERLAP_SENTENCES:] if len(buffer) >= OVERLAP_SENTENCES else buffer[:]
        buffer = []
        buffer_words = 0

    for i, sentence in enumerate(sentences):
        sw = word_count(sentence)

        # Start a new buffer with overlap
        if not buffer and prev_tail:
            buffer.extend(prev_tail)
            buffer_words += sum(word_count(s) for s in prev_tail)
            prev_tail = []

        buffer.append(sentence)
        buffer_words += sw

        # Close if: at a semantic boundary OR at max size
        is_boundary = i in boundaries and buffer_words >= MIN_WORDS
        is_full = buffer_words >= MAX_WORDS
        if is_boundary or is_full:
            close_buffer()

    close_buffer()
    return chunks


def build_chunks_for_paper(client: OpenAI, paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Semantic-chunk a paper, one section at a time."""
    blocks = paper.get("blocks") or []
    paper_id = paper["paper_id"]

    # Group blocks by section
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
        section_text = "\n\n".join(grouped[section])
        v1_section = V2_TO_V1_SECTION.get(section, "unknown")
        chunk_texts = chunk_section(client, section_text, v1_section)
        for text in chunk_texts:
            all_chunks.append({
                "chunk_id": stable_chunk_id(paper_id, chunk_idx),
                "paper_id": paper_id,
                "title": paper.get("title"),
                "topic": paper.get("topic"),
                "year": paper.get("year"),
                "citation_count": paper.get("citation_count", 0),
                "arxiv_id": paper.get("arxiv_id"),
                "chunk_index": chunk_idx,
                "word_count": word_count(text),
                "section_hint": v1_section,
                "section_v2_raw": section,
                "text": text,
            })
            chunk_idx += 1

    for c in all_chunks:
        c["total_chunks"] = len(all_chunks)

    return all_chunks


def main() -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"Loading papers from {INPUT}...")
    papers = json.loads(INPUT.read_text())
    print(f"Loaded {len(papers)} papers")

    deduped = dedupe_papers(papers)
    print(f"After paper-level dedup: {len(deduped)} papers")

    all_chunks: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    total_sentences = 0
    for i, paper in enumerate(deduped, start=1):
        title = (paper.get("title") or "")[:50]
        try:
            chunks = build_chunks_for_paper(client, paper)
            all_chunks.extend(chunks)
            total_sentences += sum(len(split_sentences(b["text"])) for b in paper.get("blocks", []))
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (len(deduped) - i)
            print(f"  [{i}/{len(deduped)}] {title:<50} → {len(chunks)} chunks  ({elapsed:.0f}s elapsed, {eta:.0f}s ETA)")
        except Exception as exc:
            print(f"  [{i}/{len(deduped)}] {title} FAILED: {exc}")

    # Fingerprint-based chunk dedup (defensive)
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    dropped = 0
    for c in all_chunks:
        fp = content_fingerprint(c["text"])
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        unique.append(c)

    OUTPUT.write_text(json.dumps(unique, indent=2, ensure_ascii=False))

    print(f"\n=== SEMANTIC CHUNKING SUMMARY ===")
    print(f"Papers processed      : {len(deduped)}")
    print(f"Sentences embedded    : ~{total_sentences:,}")
    print(f"Total chunks (raw)    : {len(all_chunks):,}")
    print(f"After dedup           : {len(unique):,} (dropped {dropped})")
    if unique:
        wcs = [c["word_count"] for c in unique]
        print(f"Avg words/chunk       : {sum(wcs) / len(wcs):.0f}")
        print(f"Min/Max words         : {min(wcs)} / {max(wcs)}")

    section_counts: dict[str, int] = {}
    for c in unique:
        section_counts[c["section_hint"]] = section_counts.get(c["section_hint"], 0) + 1
    print(f"\n=== SECTION DISTRIBUTION ===")
    for sec, cnt in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {sec:<20} {cnt:>5} ({cnt/len(unique)*100:.0f}%)")

    print(f"\nWritten to: {OUTPUT}")
    print(f"Total time: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
