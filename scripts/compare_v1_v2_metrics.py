"""Compare v1 (pypdf) vs v2 (PyMuPDF) eval results side-by-side.

Reads eval_v1_baseline.json and eval_v2_pymupdf.json (produced by
scripts/run_day2_pipeline.py) and prints a comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
V1 = ROOT / "eval_v1_baseline.json"
V2 = ROOT / "eval_v2_pymupdf.json"


def rate(v: dict | None) -> str:
    if not v or v.get("value") is None:
        return "n/a"
    return f"{v['value']:.3f}"


def delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "  n/a"
    diff = b - a
    pct = diff / a * 100 if a else 0
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.3f} ({sign}{pct:.1f}%)"


def main() -> None:
    if not V1.exists() or not V2.exists():
        print(f"Missing results file. Run scripts/run_day2_pipeline.py first.")
        return

    v1 = json.loads(V1.read_text())["summary"]
    v2 = json.loads(V2.read_text())["summary"]

    # Metric dicts use string keys after JSON round-trip
    def get_k(section: dict, k: int) -> dict | None:
        return section.get(str(k)) or section.get(k)

    print("=" * 80)
    print("RSE v1 (pypdf) vs v2 (PyMuPDF + paragraph chunking)")
    print("=" * 80)

    def val(section: dict, k: int) -> float | None:
        entry = get_k(section, k)
        return entry.get("value") if entry else None

    rows = [
        ("route_accuracy", v1["route_accuracy"], v2["route_accuracy"]),
        ("hit_rate@10", val(v1["id_relevant_hit_rate"], 10), val(v2["id_relevant_hit_rate"], 10)),
        ("hit_rate@20", val(v1["id_relevant_hit_rate"], 20), val(v2["id_relevant_hit_rate"], 20)),
        ("recall@10", val(v1["recall"], 10), val(v2["recall"], 10)),
        ("recall@20", val(v1["recall"], 20), val(v2["recall"], 20)),
        ("mrr", v1["mrr"]["value"], v2["mrr"]["value"]),
        ("confidence_accuracy", v1["confidence_decision_accuracy"]["value"], v2["confidence_decision_accuracy"]["value"]),
    ]

    print(f"\n{'Metric':<25} {'v1':>10} {'v2':>10} {'delta':>18}")
    print("-" * 68)
    for name, a, b in rows:
        a_str = f"{a:.3f}" if a is not None else "n/a"
        b_str = f"{b:.3f}" if b is not None else "n/a"
        print(f"{name:<25} {a_str:>10} {b_str:>10} {delta(a, b):>18}")

    print("\nInterpretation:")
    r1 = val(v1["recall"], 10) or 0
    r2 = val(v2["recall"], 10) or 0
    if r2 > r1 + 0.02:
        print(f"  [OK] Recall improved meaningfully (+{(r2-r1)*100:.1f} pts)")
    elif r2 > r1 - 0.02:
        print(f"  - Recall unchanged - PyMuPDF preserves what pypdf lost, but retrieval was already finding it")
    else:
        print(f"  [WARN]  Recall dropped - investigate; chunking may be too aggressive")


if __name__ == "__main__":
    main()
