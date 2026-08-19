"""Prompt templates for bounded, evidence-grounded agentic research."""
from __future__ import annotations
import json
from typing import Any

def build_tool_system_prompt() -> str:
    return (
        "You are a research assistant. Use only the provided evidence and the approved research tools. "
        "Call a tool only when the evidence is missing or a current source is required. "
        "Do not reveal private chain-of-thought or hidden reasoning. "
        "Return a concise answer with citations in the form [source_1], [source_2]. "
        "If the evidence does not support an answer, say so plainly and do not guess."
    )

def format_evidence(evidence: list[dict[str, Any]], max_chars: int = 18000) -> str:
    rows = []
    for index, item in enumerate(evidence, start=1):
        payload = dict(item)
        payload.setdefault("source_ref", f"source_{index}")
        rows.append(payload)
    text = json.dumps(rows, ensure_ascii=False, default=str)
    return text[:max_chars]

def build_tool_user_prompt(query: str, evidence: list[dict[str, Any]]) -> str:
    return f"Question: {query}\nEvidence records:\n{format_evidence(evidence)}"
