"""Task 2: Map local PDF filenames to existing RSE paper_ids.

Filename pattern: W{OPENALEX_ID}_{hash}_{title_slug}.pdf
Existing paper_id: https://openalex.org/W{OPENALEX_ID}

Produces data/pdf_to_paper_id.json for use by extraction step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


PDF_DIR = Path("data/pdfs")
METADATA_PATH = Path("data/full_text_papers.json")
OUTPUT_PATH = Path("data/pdf_to_paper_id.json")

FILENAME_RE = re.compile(r"^(W\d+)_", re.IGNORECASE)


def main() -> None:
    papers = json.loads(METADATA_PATH.read_text())
    known_ids = {p["paper_id"]: p for p in papers}
    print(f"Existing metadata: {len(papers)} papers\n")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    matched: dict[str, str] = {}
    unmatched: list[str] = []

    for pdf in pdfs:
        m = FILENAME_RE.match(pdf.name)
        if not m:
            unmatched.append(pdf.name)
            continue
        openalex_slug = m.group(1)
        paper_id = f"https://openalex.org/{openalex_slug}"
        if paper_id in known_ids:
            matched[pdf.name] = paper_id
        else:
            unmatched.append(pdf.name)

    OUTPUT_PATH.write_text(json.dumps(matched, indent=2))
    print(f"=== MAPPING RESULT ===")
    print(f"Matched   : {len(matched)}")
    print(f"Unmatched : {len(unmatched)}")
    print(f"Written to: {OUTPUT_PATH}")

    if unmatched:
        print(f"\n=== UNMATCHED (need investigation) ===")
        for f in unmatched[:15]:
            print(f"  {f}")
        if len(unmatched) > 15:
            print(f"  ... and {len(unmatched) - 15} more")


if __name__ == "__main__":
    main()
