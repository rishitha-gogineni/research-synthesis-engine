"""Compare v1 (pypdf) vs v2 (PyMuPDF+paragraph) vs v3 (PyMuPDF+semantic) side by side."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
V1 = ROOT / "eval_v1_baseline.json"
V2 = ROOT / "eval_v2_pymupdf.json"
V3 = ROOT / "eval_v3_semantic.json"


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())["summary"]


def val(section: dict, k: int) -> float | None:
    entry = section.get(str(k)) or section.get(k)
    return entry.get("value") if entry else None


def fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "  n/a"


def delta_str(base: float | None, new: float | None) -> str:
    if base is None or new is None:
        return "     n/a"
    d = new - base
    pct = (d / base * 100) if base else 0
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f} ({sign}{pct:.1f}%)"


def main() -> None:
    v1, v2, v3 = load(V1), load(V2), load(V3)
    if v1 is None:
        print(f"[ERROR] Missing {V1.name}")
        return

    print("=" * 100)
    print("RSE - three-way retrieval comparison")
    print("v1 = pypdf + fixed-word (production baseline)")
    print("v2 = PyMuPDF + paragraph-aware (structural upgrade, no overlap)")
    print("v3 = PyMuPDF + semantic chunking + overlap + keep appendix")
    print("=" * 100)

    metric_rows = [
        ("route_accuracy", v1["route_accuracy"], v2["route_accuracy"] if v2 else None, v3["route_accuracy"] if v3 else None),
        ("hit_rate@10", val(v1["id_relevant_hit_rate"], 10), val(v2["id_relevant_hit_rate"], 10) if v2 else None, val(v3["id_relevant_hit_rate"], 10) if v3 else None),
        ("recall@10", val(v1["recall"], 10), val(v2["recall"], 10) if v2 else None, val(v3["recall"], 10) if v3 else None),
        ("mrr", v1["mrr"]["value"], v2["mrr"]["value"] if v2 else None, v3["mrr"]["value"] if v3 else None),
        ("confidence_accuracy", v1["confidence_decision_accuracy"]["value"], v2["confidence_decision_accuracy"]["value"] if v2 else None, v3["confidence_decision_accuracy"]["value"] if v3 else None),
    ]

    print(f"\n{'Metric':<22}{'v1':>10}{'v2':>10}{'v3':>10}{'v2 vs v1':>18}{'v3 vs v1':>18}{'v3 vs v2':>18}")
    print("-" * 106)
    for name, a, b, c in metric_rows:
        print(
            f"{name:<22}"
            f"{fmt(a):>10}{fmt(b):>10}{fmt(c):>10}"
            f"{delta_str(a, b):>18}{delta_str(a, c):>18}{delta_str(b, c):>18}"
        )

    if v3:
        r1 = val(v1["recall"], 10) or 0
        r3 = val(v3["recall"], 10) or 0
        h1 = val(v1["id_relevant_hit_rate"], 10) or 0
        h3 = val(v3["id_relevant_hit_rate"], 10) or 0
        print("\n=== Verdict ===")
        if r3 > r1 + 0.02:
            print(f"  [OK] v3 recall meaningfully improved (+{(r3-r1)*100:.1f} pts)")
            print(f"  [OK] v3 hit@10 vs v1: {'up' if h3 > h1 else 'down'}")
            print(f"  -> Ship: merge branch to main, redeploy with RSE_CHUNK_COLLECTION=research_paper_chunks_v3")
        elif r3 > r1 - 0.02:
            print(f"  - v3 recall roughly flat vs v1 (delta={(r3-r1)*100:+.1f} pts)")
            print(f"  -> Interesting finding: chunking strategy is NOT the retrieval bottleneck")
            print(f"  -> Ship: 'measured, learned' story in README; keep v1 in production")
        else:
            print(f"  [ERROR] v3 recall dropped ({(r3-r1)*100:+.1f} pts)")
            print(f"  -> Do NOT merge; investigate")


if __name__ == "__main__":
    main()
