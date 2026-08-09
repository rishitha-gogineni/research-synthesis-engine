"""One-command runbook to execute Day 2 tasks (embed → index → eval).

Usage:
    python scripts/run_day2_pipeline.py

Runs:
  Task 6a: Embed v2 chunks with text-embedding-3-large
  Task 6b: Upload to Qdrant Cloud into research_paper_chunks_v2 collection
  Task 8:  Run RSE eval against v2 collection + compare to baseline

Requires .env with OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parent.parent
CHUNKS_V2 = ROOT / "data" / "full_text_chunks_v2.json"
EMBEDDED_V2 = ROOT / "data" / "embedded_full_text_chunks_v2.json"
COLLECTION_V2 = "research_paper_chunks_v2"
EVAL_QUERIES = ROOT / "tests" / "fixtures" / "eval_queries_v2.json"
RESULTS_BASELINE = ROOT / "eval_v1_baseline.json"
RESULTS_V2 = ROOT / "eval_v2_pymupdf.json"


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*70}")
    print(f"STEP: {desc}")
    print(f"CMD:  {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {desc}")
        sys.exit(1)
    print(f"\n✅ DONE: {desc}")


def main() -> None:
    print("=" * 70)
    print("DAY 2 PIPELINE — PyMuPDF v2 migration")
    print("=" * 70)
    print(f"Chunks input : {CHUNKS_V2}")
    print(f"Collection   : {COLLECTION_V2}")
    print(f"Estimated cost: ~$3-5 (embeddings + eval LLM calls)")
    print(f"Estimated time: ~45-60 min")

    if not CHUNKS_V2.exists():
        print(f"\n❌ {CHUNKS_V2} missing. Run scripts/chunk_paragraph_aware.py first.")
        sys.exit(1)

    t0 = time.perf_counter()

    # Step 1: Embed
    run(
        ["python", "-m", "full_text.embed_chunks",
         "--input", str(CHUNKS_V2),
         "--output", str(EMBEDDED_V2)],
        "Embed v2 chunks with text-embedding-3-large"
    )

    # Step 2: Index into Qdrant (new collection so v1 stays intact)
    run(
        ["python", "-m", "full_text.index_chunks_qdrant",
         "--input", str(EMBEDDED_V2),
         "--collection", COLLECTION_V2],
        f"Index v2 chunks to Qdrant collection '{COLLECTION_V2}'"
    )

    # Step 3: Baseline eval (v1 collection) — capture JSON directly
    print("\n" + "=" * 70)
    print("STEP: Baseline eval on v1 collection (research_paper_chunks)")
    print("=" * 70 + "\n")
    result = subprocess.run(
        ["python", "-m", "retrieval.evaluate",
         "--queries", str(EVAL_QUERIES),
         "--json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"❌ Baseline eval failed:\n{result.stderr}")
        sys.exit(1)
    RESULTS_BASELINE.write_text(result.stdout)
    print(f"✅ Baseline saved: {RESULTS_BASELINE}")

    # Step 4: v2 eval
    print("\n" + "=" * 70)
    print(f"STEP: v2 eval on collection '{COLLECTION_V2}'")
    print("=" * 70 + "\n")
    result = subprocess.run(
        ["python", "-m", "retrieval.evaluate",
         "--queries", str(EVAL_QUERIES),
         "--chunk-collection", COLLECTION_V2,
         "--json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"❌ v2 eval failed:\n{result.stderr}")
        sys.exit(1)
    RESULTS_V2.write_text(result.stdout)
    print(f"✅ v2 saved: {RESULTS_V2}")

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {elapsed/60:.1f} min")
    print(f"{'='*70}")
    print(f"\nRun scripts/compare_v1_v2_metrics.py to see the diff.")


if __name__ == "__main__":
    main()
