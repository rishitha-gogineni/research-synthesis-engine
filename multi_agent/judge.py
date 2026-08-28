"""LLM-as-Judge evaluation for research output quality."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from multi_agent.config import DEFAULT_MODEL
from dataclasses import dataclass

from multi_agent.prompts import JUDGE_SYSTEM_PROMPT
from multi_agent.trace import Tracer


@dataclass
class JudgeResult:
    factual_accuracy: float
    citation_accuracy: float
    completeness: float
    source_quality: float
    tool_efficiency: float
    overall: float
    passed: bool
    reasoning: str


def evaluate_output(
    query: str,
    cited_report: dict[str, Any],
    trace_summary: dict[str, Any],
    tracer: Tracer,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Score the research output on 5 quality dimensions."""
    if client is None:
        client = OpenAI()

    tracer.log("judge", "start")

    report_text = cited_report.get("cited_report", "")
    references = cited_report.get("references", [])
    uncited = cited_report.get("uncited_claims", [])

    prompt = f"""Evaluate this research output:

Original query: {query}

Research report:
{report_text}

References: {json.dumps(references[:20], indent=2)}
Uncited claims: {json.dumps(uncited)}

Trace info:
- Total agents used: {trace_summary.get('total_agents', 0)}
- Total findings: {trace_summary.get('total_findings', 0)}
- Elapsed seconds: {trace_summary.get('elapsed_seconds', 0):.1f}

Score each dimension 0.0-1.0 and provide an overall score and pass/fail.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    content = response.choices[0].message.content or "{}"
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "factual_accuracy": 0.0,
            "citation_accuracy": 0.0,
            "completeness": 0.0,
            "source_quality": 0.0,
            "tool_efficiency": 0.0,
            "overall": 0.0,
            "pass": False,
            "reasoning": "Failed to parse judge response",
        }

    tracer.log("judge", "complete", overall=result.get("overall", 0.0))
    return result
