"""Compare PDF text extractors on a single file.

Usage:
    python -m scripts.compare_pdf_extractors <path-to-pdf> [--pages 1-3]

Runs pypdf, pdfplumber, and PyMuPDF on the same PDF and prints quality
metrics plus a text preview for eyeball comparison.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path


def hyphenation_count(text: str) -> int:
    return len(re.findall(r"\w-\s*\n\s*\w", text))


def paragraph_breaks(text: str) -> int:
    return len(re.findall(r"\n\s*\n", text))


def lowercase_line_starts(text: str) -> float:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    lower = sum(1 for ln in lines if ln[0].islower())
    return lower / len(lines)


def repeated_lines(text: str, min_repeats: int = 3) -> int:
    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 5]
    counts = Counter(lines)
    return sum(1 for _, c in counts.items() if c >= min_repeats)


def preview(text: str, chars: int = 500) -> str:
    text = text.strip()
    return text[:chars] + ("..." if len(text) > chars else "")


def extract_pypdf(path: Path, page_range: tuple[int, int] | None) -> str:
    import pypdf
    reader = pypdf.PdfReader(str(path))
    pages = reader.pages
    if page_range:
        pages = pages[page_range[0] - 1:page_range[1]]
    return "\n".join(p.extract_text() or "" for p in pages)


def extract_pdfplumber(path: Path, page_range: tuple[int, int] | None) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        pages = pdf.pages
        if page_range:
            pages = pages[page_range[0] - 1:page_range[1]]
        for p in pages:
            parts.append(p.extract_text() or "")
    return "\n".join(parts)


def extract_pymupdf(path: Path, page_range: tuple[int, int] | None) -> str:
    import fitz
    doc = fitz.open(str(path))
    if page_range:
        pages = range(page_range[0] - 1, page_range[1])
    else:
        pages = range(len(doc))
    return "\n".join(doc[i].get_text("text") for i in pages)


def extract_pymupdf_blocks(path: Path, page_range: tuple[int, int] | None) -> str:
    """PyMuPDF 'blocks' mode - emits paragraph-like blocks with \\n\\n between."""
    import fitz
    doc = fitz.open(str(path))
    if page_range:
        pages = range(page_range[0] - 1, page_range[1])
    else:
        pages = range(len(doc))
    out = []
    for i in pages:
        blocks = doc[i].get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if text:
                out.append(text)
    return "\n\n".join(out)


EXTRACTORS = [
    ("pypdf (current)", extract_pypdf),
    ("pdfplumber", extract_pdfplumber),
    ("PyMuPDF text", extract_pymupdf),
    ("PyMuPDF blocks", extract_pymupdf_blocks),
]


def parse_page_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    if "-" in value:
        a, b = value.split("-", 1)
        return int(a), int(b)
    p = int(value)
    return p, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--pages", default="1-3", help="page range like 1-3 (default: 1-3)")
    ap.add_argument("--preview-chars", type=int, default=600)
    args = ap.parse_args()

    page_range = parse_page_range(args.pages)
    print(f"PDF: {args.pdf.name}  pages: {args.pages}")
    print("=" * 90)

    results = []
    for name, fn in EXTRACTORS:
        t0 = time.perf_counter()
        try:
            text = fn(args.pdf, page_range)
        except Exception as exc:
            print(f"\n[{name}] FAILED: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        results.append((name, text, elapsed))

    for name, text, elapsed in results:
        print(f"\n=== {name}  ({elapsed:.2f}s) ===")
        print(f"chars           : {len(text):,}")
        print(f"words           : {len(text.split()):,}")
        print(f"paragraph breaks: {paragraph_breaks(text)}")
        print(f"hyphenated cuts : {hyphenation_count(text)}   (lower = better)")
        print(f"lowercase-start : {lowercase_line_starts(text):.1%}  (lower = better; sentences got cut mid-flow)")
        print(f"repeated lines  : {repeated_lines(text)}   (page numbers / headers)")
        print(f"\n  PREVIEW:")
        for line in preview(text, args.preview_chars).splitlines():
            print(f"  | {line}")

    print("\n\n=== SUMMARY TABLE ===\n")
    print(f"{'Extractor':<20} {'chars':>8} {'words':>7} {'paragraph':>5} {'hyph':>5} {'lc%':>6} {'rep':>5} {'time':>7}")
    print("-" * 70)
    for name, text, elapsed in results:
        print(
            f"{name:<20} {len(text):>8,} {len(text.split()):>7,} "
            f"{paragraph_breaks(text):>5} {hyphenation_count(text):>5} "
            f"{lowercase_line_starts(text):>5.1%} {repeated_lines(text):>5} "
            f"{elapsed:>6.2f}s"
        )


if __name__ == "__main__":
    main()
