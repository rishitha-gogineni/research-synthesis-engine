"""Per-category breakdown of 250-query eval results.

Shows which of the 13 query categories are strong vs weak,
helping identify where to focus improvement effort.

Usage:
    python scripts/breakdown_250_eval.py [eval_250_v1.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parent.parent
DEFAULT_FILE = ROOT / "eval_250_v1.json"


def load(p: Path) -> dict:
    data = json.loads(p.read_text())
    return data


def fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "  n/a"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    if not path.exists():
        print(f"❌ Missing {path.name}")
        return

    data = load(path)
    evaluations = data["evaluations"]

    # Group by rationale prefix (our semantic category)
    categories = defaultdict(list)
    for ev in evaluations:
        rationale = ev.get("rationale", "") or ""
        if rationale.startswith("["):
            cat = rationale.split("]")[0].strip("[")
        else:
            cat = ev.get("evaluation_focus", "unknown")
        categories[cat].append(ev)

    print("=" * 110)
    print(f"PER-CATEGORY BREAKDOWN — {path.name} ({len(evaluations)} queries)")
    print("=" * 110)

    print(f"\n{'Category':<28}{'#':>4}{'route%':>8}{'hit@5':>8}{'hit@10':>8}{'recall@5':>9}{'recall@10':>10}{'MRR':>8}{'conf%':>8}")
    print("-" * 110)

    # Sort by recall@10 ascending (worst first)
    cat_stats = []
    for cat, evals in sorted(categories.items()):
        n = len(evals)

        # Route accuracy
        route_correct = sum(1 for e in evals if e["route_correct"]) / n if n else 0

        # Retrieval metrics (only for queries with relevant IDs)
        labeled = [e for e in evals if e["has_relevant_ids"]]
        n_labeled = len(labeled)

        if n_labeled:
            hit5 = sum(1 for e in labeled if e["id_hit_sets"].get("5") or e["id_hit_sets"].get(5)) / n_labeled
            hit10 = sum(1 for e in labeled if e["id_hit_sets"].get("10") or e["id_hit_sets"].get(10)) / n_labeled

            recall5_vals = [v for e in labeled if (v := (e["id_hit_fractions"].get("5") or e["id_hit_fractions"].get(5))) is not None]
            recall10_vals = [v for e in labeled if (v := (e["id_hit_fractions"].get("10") or e["id_hit_fractions"].get(10))) is not None]

            recall5 = sum(recall5_vals) / len(recall5_vals) if recall5_vals else None
            recall10 = sum(recall10_vals) / len(recall10_vals) if recall10_vals else None

            mrr_vals = [e["reciprocal_rank"] for e in labeled if e.get("reciprocal_rank") is not None]
            mrr = sum(mrr_vals) / len(mrr_vals) if mrr_vals else None
        else:
            hit5 = hit10 = recall5 = recall10 = mrr = None

        # Confidence accuracy
        conf_labeled = [e for e in evals if e.get("confidence_correct") is not None]
        conf_acc = sum(1 for e in conf_labeled if e["confidence_correct"]) / len(conf_labeled) if conf_labeled else None

        cat_stats.append((cat, n, route_correct, hit5, hit10, recall5, recall10, mrr, conf_acc, n_labeled))

    # Sort by recall@10 (worst first for easy identification)
    cat_stats.sort(key=lambda x: x[6] if x[6] is not None else 999)

    for cat, n, route_pct, hit5, hit10, recall5, recall10, mrr, conf, n_labeled in cat_stats:
        print(
            f"{cat:<28}{n:>4}"
            f"{route_pct:>7.0%}"
            f"{fmt(hit5):>8}{fmt(hit10):>8}"
            f"{fmt(recall5):>9}{fmt(recall10):>10}"
            f"{fmt(mrr):>8}"
            f"{fmt(conf):>8}"
        )

    # Worst performing queries
    print("\n\n" + "=" * 110)
    print("WORST QUERIES (recall@10 = 0, has relevant IDs)")
    print("=" * 110)

    zero_recall = [
        e for e in evaluations
        if e["has_relevant_ids"]
        and (e["id_hit_fractions"].get("10") or e["id_hit_fractions"].get(10) or 0) == 0
    ]

    print(f"\nTotal zero-recall queries: {len(zero_recall)} / {sum(1 for e in evaluations if e['has_relevant_ids'])}")
    print(f"\n{'Query':<70}{'Route':<15}{'Category'}")
    print("-" * 110)
    for e in zero_recall[:25]:
        rationale = e.get("rationale", "")[:20]
        print(f"{e['query'][:68]:<70}{e['actual_route']:<15}{rationale}")

    # Confidence gate failures
    print("\n\n" + "=" * 110)
    print("CONFIDENCE GATE FAILURES")
    print("=" * 110)

    conf_failures = [
        e for e in evaluations
        if e.get("confidence_correct") is not None and not e["confidence_correct"]
    ]
    print(f"\nTotal confidence failures: {len(conf_failures)}")
    if conf_failures:
        print(f"\n{'Query':<60}{'Expected':<25}{'Got'}")
        print("-" * 110)
        for e in conf_failures[:15]:
            print(f"{e['query'][:58]:<60}{str(e.get('expected_confidence_decision','')):<25}{str(e.get('actual_confidence_decision',''))}")


if __name__ == "__main__":
    main()
