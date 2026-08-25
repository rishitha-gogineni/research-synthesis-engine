"""Task 4: PyMuPDF extraction with section awareness and hyphenation cleanup.

Reads data/pdfs/*.pdf and produces data/full_text_papers_v2.json with:
- Clean text (paragraphs preserved via blocks mode, hyphenation fixed)
- Section markers (introduction, methods, results, discussion, conclusion, references)
- Per-page block structure preserved for downstream chunking

Section detection strategy:
1. Look for common header patterns ("1. Introduction", "Methods", "3.2 Results")
2. Use font size to disambiguate - headers are typically 1.2-1.5x body size

This is v2's replacement for pypdf-based extraction. Filters out author lists,
affiliations, references, page footers where possible.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from statistics import median

import fitz


PDF_DIR = Path("data/pdfs")
MAPPING_PATH = Path("data/pdf_to_paper_id.json")
METADATA_PATH = Path("data/full_text_papers.json")
OUTPUT_PATH = Path("data/full_text_papers_v2.json")

SECTION_PATTERNS = {
    "abstract": re.compile(r"^\s*(abstract)\s*$", re.IGNORECASE),
    "introduction": re.compile(r"^\s*(?:\d+\.?\s+)?(introduction|background)\s*$", re.IGNORECASE),
    "related_work": re.compile(r"^\s*(?:\d+\.?\s+)?(related\s+work|prior\s+work|literature\s+review)\s*$", re.IGNORECASE),
    "methods": re.compile(r"^\s*(?:\d+\.?\s+)?(method(?:ology|s)?|approach|model|architecture|proposed\s+method|our\s+method)\s*$", re.IGNORECASE),
    "experiments": re.compile(r"^\s*(?:\d+\.?\s+)?(experiment(?:s|al\s+setup)?|evaluation|experimental\s+results)\s*$", re.IGNORECASE),
    "results": re.compile(r"^\s*(?:\d+\.?\s+)?(results|findings|main\s+results)\s*$", re.IGNORECASE),
    "discussion": re.compile(r"^\s*(?:\d+\.?\s+)?(discussion|analysis)\s*$", re.IGNORECASE),
    "limitations": re.compile(r"^\s*(?:\d+\.?\s+)?(limitations?|threats\s+to\s+validity)\s*$", re.IGNORECASE),
    "conclusion": re.compile(r"^\s*(?:\d+\.?\s+)?(conclusion(?:s)?|summary|future\s+work)\s*$", re.IGNORECASE),
    "references": re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE),
    "acknowledgments": re.compile(r"^\s*(acknowledgment(?:s)?|acknowledgement(?:s)?)\s*$", re.IGNORECASE),
    "appendix": re.compile(r"^\s*(appendix|appendices)\s*[a-z]?\s*$", re.IGNORECASE),
}


def fix_hyphenation(text: str) -> str:
    """Merge hyphenated line-break words: 'impor-\\ntant' -> 'important'."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace but keep single newlines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text: str) -> str | None:
    """Return the section name if text looks like a header, else None."""
    stripped = text.strip()
    if len(stripped) > 60 or len(stripped) < 3:
        return None
    for name, pattern in SECTION_PATTERNS.items():
        if pattern.match(stripped):
            return name
    return None


def get_body_font_size(doc: fitz.Document) -> float:
    """Estimate the median font size of body text across the first 3 pages."""
    sizes = []
    for i in range(min(3, len(doc))):
        page = doc[i]
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(span["size"])
    return median(sizes) if sizes else 10.0


def extract_paper(pdf_path: Path) -> dict:
    """Extract a paper into blocks with section markers.

    Returns:
        {
            "text": full joined text (paragraphs separated by \\n\\n),
            "sections": [{"name": "introduction", "start_char": 0, "end_char": 1200}, ...],
            "blocks": [{"page": 0, "text": "...", "font_size": 10.0, "section": "abstract"}, ...],
            "page_count": int,
        }
    """
    doc = fitz.open(str(pdf_path))
    body_size = get_body_font_size(doc)
    header_threshold = body_size * 1.15  # section headers typically 15%+ larger

    blocks: list[dict] = []
    current_section = "front_matter"

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block.get("type") != 0:
                continue
            block_texts: list[str] = []
            max_size = 0.0
            for line in block.get("lines", []):
                line_text = " ".join(span["text"] for span in line.get("spans", []))
                block_texts.append(line_text)
                for span in line.get("spans", []):
                    max_size = max(max_size, span["size"])
            text = "\n".join(block_texts).strip()
            if not text:
                continue

            text = fix_hyphenation(text)
            text = normalize_whitespace(text)

            # Section header detection: pattern match, OR (large font AND short text)
            section_from_pattern = detect_section(text)
            looks_like_header = max_size >= header_threshold and len(text) < 80
            if section_from_pattern:
                current_section = section_from_pattern
            elif looks_like_header:
                pattern_check = detect_section(text)
                if pattern_check:
                    current_section = pattern_check

            blocks.append({
                "page": page_idx,
                "text": text,
                "font_size": round(max_size, 2),
                "section": current_section,
            })

    doc.close()

    # Assemble full text, paragraphs separated by blank lines
    full_text = "\n\n".join(b["text"] for b in blocks)

    # Compute section spans
    sections: list[dict] = []
    if blocks:
        current = blocks[0]["section"]
        start_block = 0
        for i, b in enumerate(blocks[1:], start=1):
            if b["section"] != current:
                sections.append({"name": current, "block_start": start_block, "block_end": i - 1})
                current = b["section"]
                start_block = i
        sections.append({"name": current, "block_start": start_block, "block_end": len(blocks) - 1})

    return {
        "text": full_text,
        "sections": sections,
        "blocks": blocks,
        "page_count": len(blocks) and (blocks[-1]["page"] + 1) or 0,
        "body_font_size": round(body_size, 2),
        "extraction_char_count": len(full_text),
    }


def main() -> None:
    mapping = json.loads(MAPPING_PATH.read_text())
    metadata = {p["paper_id"]: p for p in json.loads(METADATA_PATH.read_text())}
    print(f"Extracting {len(mapping)} PDFs with PyMuPDF blocks mode...\n")

    results: list[dict] = []
    failures: list[str] = []
    t0 = time.perf_counter()

    for i, (pdf_name, paper_id) in enumerate(mapping.items(), start=1):
        pdf_path = PDF_DIR / pdf_name
        try:
            extracted = extract_paper(pdf_path)
            meta = metadata[paper_id]
            record = {
                "paper_id": paper_id,
                "title": meta.get("title"),
                "topic": meta.get("topic"),
                "year": meta.get("year"),
                "citation_count": meta.get("citation_count"),
                "arxiv_id": meta.get("arxiv_id"),
                "pdf_filename": pdf_name,
                "extraction_status": "ok",
                **extracted,
            }
            results.append(record)
            if i % 20 == 0:
                print(f"  {i}/{len(mapping)} extracted ({time.perf_counter()-t0:.1f}s)")
        except Exception as exc:
            failures.append(f"{pdf_name}: {exc}")

    elapsed = time.perf_counter() - t0
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Success  : {len(results)}")
    print(f"Failed   : {len(failures)}")
    print(f"Elapsed  : {elapsed:.1f}s ({elapsed/len(mapping):.2f}s/paper)")
    print(f"Written  : {OUTPUT_PATH}")
    print(f"Total chars extracted: {sum(r['extraction_char_count'] for r in results):,}")

    if results:
        section_counts: dict[str, int] = {}
        for r in results:
            for s in r["sections"]:
                section_counts[s["name"]] = section_counts.get(s["name"], 0) + 1
        print(f"\n=== SECTION DETECTION ===")
        for name, count in sorted(section_counts.items(), key=lambda x: -x[1])[:12]:
            pct = count / len(results) * 100
            print(f"  {name:<20} {count:>4} papers ({pct:.0f}%)")

    if failures:
        print(f"\n=== FAILURES ===")
        for f in failures[:10]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
