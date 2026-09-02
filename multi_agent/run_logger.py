"""Persistent per-run logging so past runs can be audited after the fact.

The UI and API only ever held trace/agent data in memory for the current
request — closing the tab or making a new call lost it. This appends one
JSON line per completed run with enough to answer "did every agent work
properly": each agent's status/findings/errors, the judge scores, and any
hallucination flags.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = Path("data/multi_agent_run_log.jsonl")


def log_run(
    query: str,
    result: dict[str, Any],
    *,
    source: str = "ui",
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one run's outcome to the run log. Never raises: a logging
    failure shouldn't take down a research run that otherwise succeeded."""
    try:
        agents = result.get("store_summary", {}).get("agents", [])
        cited_report = result.get("cited_report", {})
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": source,
            "query": query,
            "effort_level": result.get("effort_level"),
            "agents": [
                {
                    "agent_type": a.get("agent_type"),
                    "status": a.get("status"),
                    "findings_count": len(a.get("findings", [])),
                    "error": a.get("error"),
                }
                for a in agents
            ],
            "judge_overall": result.get("judge_scores", {}).get("overall"),
            "judge_pass": result.get("judge_scores", {}).get("pass"),
            "hallucination_flags": (
                cited_report.get("hallucination_flags", [])
                if isinstance(cited_report, dict) else []
            ),
            "elapsed_seconds": result.get("trace", {}).get("elapsed_seconds"),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass
