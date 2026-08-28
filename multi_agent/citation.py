"""CitationAgent — validates and attributes sources in the final report."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from multi_agent.config import DEFAULT_MODEL
from multi_agent.findings_store import FindingsStore
from multi_agent.prompts import CITATION_SYSTEM_PROMPT
from multi_agent.trace import Tracer


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

    content = response.choices[0].message.content or "{}"
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "cited_report": report_text,
            "references": [],
            "uncited_claims": ["Failed to process citations"],
        }

    tracer.log(
        "citation_agent",
        "complete",
        references_count=len(result.get("references", [])),
        uncited_count=len(result.get("uncited_claims", [])),
    )
    return result
