"""Compare v1/v2/v3 on the natural (chunking-agnostic) eval fixture."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
V1 = ROOT / "eval_v1_natural.json"
V2 = ROOT / "eval_v2_natural.json"
V3 = ROOT / "eval_v3_natural.json"


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())["summary"]


def val(section: dict, k: int) -> float | None:
    entry = section.get(str(k)) or section.get(k)
    return entry.get("value") if entry else None


def fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "  n/a"


def delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "     n/a"
    d = b - a
    pct = (d / a * 100) if a else 0
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f} ({sign}{pct:.1f}%)"


def main() -> None:
    v1, v2, v3 = load(V1), load(V2), load(V3)
    print("=" * 100)
    print("RSE — three-way retrieval comparison on NATURAL eval fixture")
    print("(154 queries with chunking-agnostic ground truth)")
    print("=" * 100)

    if v1 is None:
        print(f"\n❌ Missing {V1.name} — run eval on v1 collection first")
        return

    rows = [
        ("route_accuracy", v1["route_accuracy"], v2["route_accuracy"] if v2 else None, v3["route_accuracy"] if v3 else None),
        ("hit_rate@10", val(v1["id_relevant_hit_rate"], 10), val(v2["id_relevant_hit_rate"], 10) if v2 else None, val(v3["id_relevant_hit_rate"], 10) if v3 else None),
        ("recall@10", val(v1["recall"], 10), val(v2["recall"], 10) if v2 else None, val(v3["recall"], 10) if v3 else None),
        ("mrr", v1["mrr"]["value"], v2["mrr"]["value"] if v2 else None, v3["mrr"]["value"] if v3 else None),
        ("confidence_accuracy", v1["confidence_decision_accuracy"]["value"], v2["confidence_decision_accuracy"]["value"] if v2 else None, v3["confidence_decision_accuracy"]["value"] if v3 else None),
    ]

    print(f"\n{'Metric':<22}{'v1':>10}{'v2':>10}{'v3':>10}{'v2 vs v1':>18}{'v3 vs v1':>18}{'v3 vs v2':>18}")
    print("-" * 106)
    for name, a, b, c in rows:
        print(
            f"{name:<22}"
            f"{fmt(a):>10}{fmt(b):>10}{fmt(c):>10}"
            f"{delta(a, b):>18}{delta(a, c):>18}{delta(b, c):>18}"
        )

    if v3:
        r1 = val(v1["recall"], 10) or 0
        r3 = val(v3["recall"], 10) or 0
        h1 = val(v1["id_relevant_hit_rate"], 10) or 0
        h3 = val(v3["id_relevant_hit_rate"], 10) or 0
        print("\n=== Verdict on the CHUNKING-AGNOSTIC eval ===")
        if r3 > r1 + 0.02 or h3 > h1 + 0.02:
            print(f"  ✅ v3 (semantic) improved retrieval on fair eval")
            print(f"  → The old fixture's chunk_id coupling WAS hiding a real improvement")
            print(f"  → Ship: merge branch, deploy with RSE_CHUNK_COLLECTION=research_paper_chunks_v3")
        elif abs(r3 - r1) <= 0.02:
            print(f"  ➖ v3 roughly matches v1 on fair eval")
            print(f"  → PyMuPDF migration produces EQUIVALENT retrieval quality")
            print(f"  → Old fixture was penalizing it — the '20% drop' was measurement artifact")
            print(f"  → Ship: could deploy v3, or keep v1 (both fine)")
        else:
            print(f"  ❌ v3 still worse than v1 on fair eval ({(r3-r1)*100:+.1f} pts)")
            print(f"  → Migration genuinely doesn't help — chunking strategy isn't the bottleneck")
            print(f"  → Ship: 'measured, rejected hypothesis' story")


if __name__ == "__main__":
    main()
