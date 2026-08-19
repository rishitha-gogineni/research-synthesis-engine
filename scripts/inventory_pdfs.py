"""Task 1: Inventory 152 PDFs — count pages, detect corrupt/scanned, sample quality."""

from __future__ import annotations

import sys
from pathlib import Path

import fitz


PDF_DIR = Path("data/pdfs")


def main() -> None:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found: {len(pdfs)} PDFs\n")

    total_pages = 0
    corrupt: list[str] = []
    scanned: list[str] = []
    ok: list[tuple[str, int, int]] = []  # (name, pages, chars_first_page)

    for pdf in pdfs:
        try:
            doc = fitz.open(str(pdf))
            pages = len(doc)
            first_text = doc[0].get_text("text").strip()
            char_count = len(first_text)
            doc.close()
            if char_count < 100:
                scanned.append(pdf.name)
            else:
                ok.append((pdf.name, pages, char_count))
                total_pages += pages
        except Exception as exc:
            corrupt.append(f"{pdf.name}: {exc}")

    print(f"=== SUMMARY ===")
    print(f"OK papers        : {len(ok)}")
    print(f"Corrupt          : {len(corrupt)}")
    print(f"Scanned (no text): {len(scanned)}")
    print(f"Total pages      : {total_pages}")
    if ok:
        pages_only = [p for _, p, _ in ok]
        print(f"Avg pages/paper  : {sum(pages_only) / len(pages_only):.1f}")
        print(f"Min/Max pages    : {min(pages_only)} / {max(pages_only)}")

    if scanned:
        print(f"\n=== SCANNED (skipping) ===")
        for s in scanned[:10]:
            print(f"  {s}")
        if len(scanned) > 10:
            print(f"  ... and {len(scanned) - 10} more")

    if corrupt:
        print(f"\n=== CORRUPT ===")
        for c in corrupt:
            print(f"  {c}")

    print(f"\n=== SAMPLE QUALITY (first page from 3 random papers) ===")
    import random
    random.seed(42)
    for name, pages, _ in random.sample(ok, min(3, len(ok))):
        print(f"\n--- {name[:60]}... ({pages} pages) ---")
        doc = fitz.open(str(PDF_DIR / name))
        text = doc[0].get_text("blocks")
        doc.close()
        # Print first 3 blocks
        for i, block in enumerate(text[:3]):
            block_text = block[4].strip().replace("\n", " ")
            print(f"  [block {i+1}] {block_text[:200]}")


if __name__ == "__main__":
    main()
