"""Smoke-test the deployed Research Synthesis Engine API.

Usage:
    python scripts/live_smoke_test.py
    python scripts/live_smoke_test.py --api-url https://your-render-service.onrender.com
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_API_URL = "https://research-synthesis-engine-api.onrender.com"


@dataclass(frozen=True)
class SmokeCase:
    name: str
    question: str
    expected_confidence: set[str]
    expected_route: set[str] | None = None
    chat_history: list[dict[str, str]] | None = None
    require_answer: bool = False
    require_no_confident_answer: bool = False


def post_json(url: str, payload: dict[str, Any], request_id: str) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "X-Request-ID": request_id},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": {"code": "INVALID_RESPONSE", "message": raw[:500]}}
        return exc.code, body


def brief_answer(body: dict[str, Any]) -> str:
    brief = body.get("brief") or {}
    return str(brief.get("direct_answer") or "")


def run_case(api_url: str, index: int, case: SmokeCase) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": case.question,
        "top_k": 8,
        "include_debug": False,
        "include_evidence_matrix": True,
        "include_reading_path": False,
        "include_open_problems": False,
    }
    if case.chat_history:
        payload["chat_history"] = case.chat_history

    status, body = post_json(f"{api_url.rstrip('/')}/guidance", payload, f"live-smoke-{index}")
    retrieval = body.get("retrieval") or {}
    route = (retrieval.get("route") or {}).get("route")
    confidence = (body.get("confidence") or {}).get("decision")
    answer = brief_answer(body)

    failures: list[str] = []
    if status != 200:
        failures.append(f"HTTP {status}")
    if confidence not in case.expected_confidence:
        failures.append(f"confidence={confidence!r}, expected one of {sorted(case.expected_confidence)}")
    if case.expected_route and route not in case.expected_route:
        failures.append(f"route={route!r}, expected one of {sorted(case.expected_route)}")
    if case.require_answer and not answer.strip():
        failures.append("expected a non-empty direct answer")
    if case.require_no_confident_answer and confidence == "sufficient_evidence":
        failures.append("expected refusal/insufficient evidence, got sufficient_evidence")
    if case.name == "metadata_listing" and "cannot answer" in answer.lower():
        failures.append("metadata listing returned guarded cannot-answer wording")

    return {
        "case": case.name,
        "status": status,
        "confidence": confidence,
        "route": route,
        "papers": retrieval.get("paper_result_count"),
        "chunks": retrieval.get("chunk_result_count"),
        "answer_empty": not bool(answer.strip()),
        "passed": not failures,
        "failures": failures,
    }


def smoke_cases() -> list[SmokeCase]:
    comparison_history = [
        {"role": "user", "content": "Compare LoRA and BitFit for parameter-efficient fine-tuning."},
        {
            "role": "assistant",
            "content": "LoRA adapts low-rank weight updates while BitFit fine-tunes bias terms only.",
        },
    ]
    return [
        SmokeCase(
            name="comparison_answer",
            question="Compare LoRA and BitFit for parameter-efficient fine-tuning.",
            expected_confidence={"sufficient_evidence"},
            expected_route={"hybrid_both"},
            require_answer=True,
        ),
        SmokeCase(
            name="paper_lookup",
            question="Explain the BitFit paper.",
            expected_confidence={"sufficient_evidence"},
            expected_route={"hybrid_both"},
            require_answer=True,
        ),
        SmokeCase(
            name="metadata_listing",
            question="Show me highly cited AI agent survey papers published after 2023.",
            expected_confidence={"broaden_search", "sufficient_evidence"},
            expected_route={"metadata_filter"},
        ),
        SmokeCase(
            name="out_of_corpus_refusal",
            question="What does this system know about marine biology and coral bleaching?",
            expected_confidence={"insufficient_evidence", "ask_clarifying_question", "broaden_search"},
            require_no_confident_answer=True,
        ),
        SmokeCase(
            name="contextual_followup",
            question="What are the limitations?",
            chat_history=comparison_history,
            expected_confidence={"sufficient_evidence", "broaden_search"},
            expected_route={"hybrid_both", "chunk_level", "paper_level"},
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    args = parser.parse_args()

    results = [run_case(args.api_url, index, case) for index, case in enumerate(smoke_cases(), start=1)]
    print(json.dumps({"api_url": args.api_url, "results": results}, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
