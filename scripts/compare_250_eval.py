"""Compare v1/v2/v3 evaluation result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
V1 = ROOT / "eval_250_v1.json"
V2 = ROOT / "eval_250_v2.json"
V3 = ROOT / "eval_250_v3.json"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=V1, help=f"v1 result JSON (default: {V1.name})")
    parser.add_argument("--v2", type=Path, default=V2, help=f"v2 result JSON (default: {V2.name})")
    parser.add_argument("--v3", type=Path, default=V3, help=f"v3 result JSON (default: {V3.name})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v1, v2, v3 = load(args.v1), load(args.v2), load(args.v3)
    print("=" * 100)
    print("RSE - three-way retrieval comparison on 250-query eval fixture")
    print("(same 250-query fixture and paper-ID ground truth for all three runs)")
    print("=" * 100)

    if v1 is None:
        print(f"\n[ERROR] Missing {args.v1}")
        return

    rows = [
        ("route_accuracy", v1["route_accuracy"], v2["route_accuracy"] if v2 else None, v3["route_accuracy"] if v3 else None),
        ("hit_rate@5", val(v1["id_relevant_hit_rate"], 5), val(v2["id_relevant_hit_rate"], 5) if v2 else None, val(v3["id_relevant_hit_rate"], 5) if v3 else None),
        ("hit_rate@10", val(v1["id_relevant_hit_rate"], 10), val(v2["id_relevant_hit_rate"], 10) if v2 else None, val(v3["id_relevant_hit_rate"], 10) if v3 else None),
        ("recall@5", val(v1["recall"], 5), val(v2["recall"], 5) if v2 else None, val(v3["recall"], 5) if v3 else None),
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

    if v3 and v1:
        r1 = val(v1["recall"], 10) or 0
        r3 = val(v3["recall"], 10) or 0
        h1 = val(v1["id_relevant_hit_rate"], 10) or 0
        h3 = val(v3["id_relevant_hit_rate"], 10) or 0
        print("\n=== Verdict ===")
        if r3 > r1 + 0.02 or h3 > h1 + 0.02:
            print(f"  [OK] v3 (semantic) wins - deploy with RSE_CHUNK_COLLECTION=research_paper_chunks_v3")
        elif r3 >= r1 - 0.01:
            print(f"  - v3 roughly matches v1 - chunking isn't the bottleneck")
        else:
            print(f"  [ERROR] v3 worse than v1 - semantic chunking didn't help here")


if __name__ == "__main__":
    main()
