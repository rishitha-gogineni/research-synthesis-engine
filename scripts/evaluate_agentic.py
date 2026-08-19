from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic.evaluation import evaluate_planner


def main() -> None:
    print(json.dumps(evaluate_planner(), indent=2))


if __name__ == "__main__":
    main()
