"""CitationAgent — validates and attributes sources in the final report."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from multi_agent.config import DEFAULT_MODEL
from multi_agent.findings_store import FindingsStore
from multi_agent.prompts import CITATION_SYSTEM_PROMPT
from multi_agent.schemas import validate_or_raw, CitationSchema
from multi_agent.trace import Tracer

# Matches numbers worth fact-checking against findings — citation counts,
# percentages, dates, parameter counts. Single-digit numbers (list indices,
# "one of three") are excluded; they're too noisy and rarely the kind of
# claim a report needs grounding for.
_NUMERIC_CLAIM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")


def _find_unverified_numbers(report_text: str, findings_text: str) -> list[str]:
    """Flag numeric claims in the report that don't appear in any finding.

    The LLM-as-judge and citation LLM are both asked to police their own
    grounding, but an LLM can still state a specific, plausible-sounding
    number that isn't actually backed by any finding (the same failure mode
    that caused a fabricated citation count for an empty-findings case
    earlier). This is a deterministic backstop: it doesn't understand
    meaning, just whether the digits showed up in the source material at
    all, so it can't catch a wrong-but-present number — only an invented
    one is guaranteed to be missing outright.
    """
    seen: set[str] = set()
    flags: list[str] = []
    for match in _NUMERIC_CLAIM_RE.finditer(report_text):
        number = match.group()
        digit_count = sum(c.isdigit() for c in number)
        if digit_count < 2 or number in seen:
            continue
        seen.add(number)
        if number not in findings_text:
            flags.append(number)
    return flags


def add_citations(
    synthesis: dict[str, Any],
    store: FindingsStore,
    tracer: Tracer,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Process the synthesis report and add proper source citations."""
    if client is None:
        client = OpenAI()

    tracer.log("citation_agent", "start")

    report_text = synthesis.get("synthesis", "")
    sources_used = synthesis.get("sources_used", [])

    # Also gather all findings as potential citation sources
    all_findings = store.get_all_findings()
    source_list = []
    for i, f in enumerate(all_findings[:30], 1):
        source_list.append(
            f"[{i}] {f.title} | Source: {f.source} | URL: {f.url or 'N/A'}"
        )
    sources_text = "\n".join(source_list)

    prompt = f"""Research report to cite:
{report_text}

Available sources:
{sources_text}

Add inline citations [1], [2], etc. to the report and build a references list.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CITATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    if response.usage is not None:
        tracer.log(
            "citation_agent", "llm_usage",
            model=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    content = response.choices[0].message.content or "{}"
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "cited_report": report_text,
            "references": [],
            "uncited_claims": ["Failed to process citations"],
        }
    else:
        result = validate_or_raw(CitationSchema, result)

    findings_text = "\n".join(f"{f.title} {f.content}" for f in all_findings)
    cited_text = result.get("cited_report", "") if isinstance(result, dict) else ""
    hallucination_flags = _find_unverified_numbers(cited_text, findings_text)
    result["hallucination_flags"] = hallucination_flags

    tracer.log(
        "citation_agent",
        "complete",
        references_count=len(result.get("references", [])),
        uncited_count=len(result.get("uncited_claims", [])),
        hallucination_flags_count=len(hallucination_flags),
    )
    return result
