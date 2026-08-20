from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic.evaluation import evaluate_agentic_responses, evaluate_planner, load_agentic_cases


DEFAULT_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "agentic_eval_queries.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agentic planning and recorded responses.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--responses",
        type=Path,
        help="Optional JSON object keyed by fixture id containing /agentic/research responses.",
    )
    args = parser.parse_args()

    cases = load_agentic_cases(args.fixture)
    report = {"planner": evaluate_planner(cases)}
    if args.responses:
        responses = json.loads(args.responses.read_text())
        if not isinstance(responses, dict):
            raise SystemExit("--responses must be a JSON object keyed by case id")
        report["responses"] = evaluate_agentic_responses(cases, responses)
    else:
        report["responses"] = {
            "status": "not_run",
            "message": "Pass --responses with recorded API responses to score live tool and citation behavior.",
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
