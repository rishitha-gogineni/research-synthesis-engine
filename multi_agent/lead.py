"""LeadResearcher agent — plans, delegates, and synthesizes."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from multi_agent.config import DEFAULT_MODEL, classify_effort, EffortLevel
from multi_agent.findings_store import FindingsStore
from multi_agent.prompts import (
    LEAD_SYSTEM_PROMPT,
    LEAD_PLAN_PROMPT,
    LEAD_SYNTHESIS_PROMPT,
    LEAD_MORE_RESEARCH_PROMPT,
)
from multi_agent.trace import Tracer


class PlanningError(RuntimeError):
    pass


def _call_llm(
    client: OpenAI,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlanningError(f"Failed to parse LLM response: {content[:200]}") from exc


def create_plan(
    query: str,
    tracer: Tracer,
    *,
    client: OpenAI | None = None,
    effort: EffortLevel | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Decompose a research query into subtasks for parallel execution."""
    if client is None:
        client = OpenAI()
    if effort is None:
        effort = classify_effort(query)

    tracer.log("lead", "plan_start", query=query, effort=effort.name)

    prompt = LEAD_PLAN_PROMPT.format(
        query=query,
        effort_level=effort.name,
        max_subagents=effort.max_subagents,
        max_tool_calls=effort.max_tool_calls_per_agent,
    )

    plan = _call_llm(client, LEAD_SYSTEM_PROMPT, prompt, model)

    subtasks = plan.get("subtasks", [])
    if not subtasks:
        raise PlanningError("Lead agent produced no subtasks")

    # Cap subtasks to effort level
    subtasks = subtasks[: effort.max_subagents]
    plan["subtasks"] = subtasks

    tracer.log(
        "lead",
        "plan_complete",
        subtask_count=len(subtasks),
        reasoning=plan.get("reasoning", ""),
    )
    return plan


def synthesize_findings(
    query: str,
    store: FindingsStore,
    tracer: Tracer,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Merge findings from all subagents into a coherent synthesis."""
    if client is None:
        client = OpenAI()

    completed = store.get_completed()
    if not completed:
        return {
            "synthesis": "No findings available from subagents.",
            "key_themes": [],
            "sources_used": [],
            "gaps": ["All subagents failed or returned no results"],
            "confidence": "low",
            "needs_more_research": False,
            "follow_up_subtasks": [],
        }

    # Build findings text for the LLM
    findings_parts = []
    for result in completed:
        findings_parts.append(f"\n--- Agent: {result.agent_type} | Task: {result.subtask} ---")
        for f in result.findings[:10]:
            findings_parts.append(f"  • {f.title}: {f.content[:200]}")
    findings_text = "\n".join(findings_parts)

    tracer.log("lead", "synthesis_start", agent_count=len(completed))

    prompt = LEAD_SYNTHESIS_PROMPT.format(
        query=query,
        findings_text=findings_text,
    )

    synthesis = _call_llm(client, LEAD_SYSTEM_PROMPT, prompt, model)
    tracer.log(
        "lead",
        "synthesis_complete",
        confidence=synthesis.get("confidence", "unknown"),
        needs_more=synthesis.get("needs_more_research", False),
    )
    return synthesis


def plan_follow_up(
    query: str,
    synthesis: dict[str, Any],
    tracer: Tracer,
    *,
    client: OpenAI | None = None,
    effort: EffortLevel | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Create additional subtasks based on identified gaps."""
    if client is None:
        client = OpenAI()
    if effort is None:
        effort = classify_effort(query)

    gaps = synthesis.get("gaps", [])
    if not gaps:
        return {"subtasks": []}

    tracer.log("lead", "follow_up_plan", gaps=gaps)

    prompt = LEAD_MORE_RESEARCH_PROMPT.format(
        query=query,
        confidence=synthesis.get("confidence", "low"),
        gaps=json.dumps(gaps),
    )

    plan = _call_llm(client, LEAD_SYSTEM_PROMPT, prompt, model)
    subtasks = plan.get("subtasks", [])[:effort.max_subagents]
    plan["subtasks"] = subtasks

    tracer.log("lead", "follow_up_complete", subtask_count=len(subtasks))
    return plan
