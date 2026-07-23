"""Measure API latency for demo research questions."""

from __future__ import annotations

import argparse
import time
import uuid
from typing import Any

import requests


DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_QUESTIONS = [
    "What are the main approaches for reducing hallucinations in LLMs?",
    "Compare RAG and self-verification methods for reducing hallucinations.",
    "Which LoRA and PEFT papers should I read first?",
    "What are common limitations in AI agent tool-use papers?",
]


def build_payload(question: str, *, top_k: int) -> dict[str, Any]:
    return {"question": question, "top_k": top_k, "include_debug": True}


def call_endpoint(base_url: str, endpoint: str, question: str, *, top_k: int, timeout: int) -> dict[str, Any]:
    request_id = f"bench-{uuid.uuid4()}"
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}{endpoint}",
        json=build_payload(question, top_k=top_k),
        headers={"Content-Type": "application/json", "X-Request-ID": request_id},
        timeout=timeout,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": {"code": "INVALID_RESPONSE", "message": response.text}}
    metrics = payload.get("metrics") or payload.get("retrieval", {}).get("metrics") or {}
    error = payload.get("error") or {}
    return {
        "endpoint": endpoint,
        "question": question,
        "status": response.status_code,
        "request_id": response.headers.get("X-Request-ID") or request_id,
        "wall_ms": elapsed_ms,
        "api_total_ms": metrics.get("total_ms"),
        "retrieval_ms": metrics.get("retrieval_ms"),
        "confidence_ms": metrics.get("confidence_ms"),
        "brief_ms": metrics.get("brief_ms"),
        "evidence_matrix_ms": metrics.get("evidence_matrix_ms"),
        "reading_path_ms": metrics.get("reading_path_ms"),
        "open_problems_ms": metrics.get("open_problems_ms"),
        "error": error.get("code"),
    }


def format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "endpoint",
        "status",
        "wall_ms",
        "api_total_ms",
        "retrieval_ms",
        "confidence_ms",
        "brief_ms",
        "evidence_matrix_ms",
        "reading_path_ms",
        "open_problems_ms",
        "error",
    ]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in rows:
        lines.append(" | ".join(str(row.get(header) if row.get(header) is not None else "-") for header in headers))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Research Synthesis Engine API latency.")
    parser.add_argument("--base-url", default=DEFAULT_API_URL)
    parser.add_argument("--endpoint", action="append", choices=["/guidance", "/agent/research"], default=None)
    parser.add_argument("--question", action="append", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoints = args.endpoint or ["/guidance", "/agent/research"]
    questions = args.question or DEFAULT_QUESTIONS
    rows = []
    for question in questions:
        print(f"\nQuestion: {question}")
        question_rows = [
            call_endpoint(args.base_url, endpoint, question, top_k=args.top_k, timeout=args.timeout)
            for endpoint in endpoints
        ]
        rows.extend(question_rows)
        print(format_table(question_rows))
    print("\nSummary")
    print(format_table(rows))


if __name__ == "__main__":
    main()
