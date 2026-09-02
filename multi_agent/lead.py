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
from multi_agent.schemas import validate_or_raw, PlanSchema, FollowUpPlanSchema, SynthesisSchema
from multi_agent.trace import Tracer


class PlanningError(RuntimeError):
    pass


# Intent signals that require external sources even when the corpus matches:
# - citation counts / most-cited → semantic_scholar (corpus has no live citations)
# - explicit recency ("latest", "2026", "newest") → arxiv/web (corpus is static)
# - comparisons against outside work → external for the "other side"
import re as _re

_EXTERNAL_INTENT_PATTERNS = [
    r"\bcitation\s+count",
    r"\bmost\s+cited\b",
    r"\bhighly\s+cited\b",
    r"\bhow\s+many\s+citations\b",
    r"\bcited\b.*\b(paper|papers|work)\b",
    r"\blatest\b",
    r"\bnewest\b",
    r"\brecent(ly)?\b",
    r"\b20(2[5-9]|[3-9]\d)\b",  # 2025 and later → beyond corpus cutoff
    r"\bcompare\b.*\b(latest|arxiv|external|newest|semantic\s+scholar)\b",
]


def _needs_external_despite_corpus(query: str) -> bool:
    """True if the query needs external sources even when the corpus matches."""
    q = query.lower()
    return any(_re.search(p, q) for p in _EXTERNAL_INTENT_PATTERNS)


def _call_llm(
    client: OpenAI,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    *,
    tracer: Tracer | None = None,
    agent_id: str = "lead",
    schema: type | None = None,
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
    if tracer is not None and response.usage is not None:
        tracer.log(
            agent_id, "llm_usage",
            model=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlanningError(f"Failed to parse LLM response: {content[:200]}") from exc
    if schema is not None:
        data = validate_or_raw(schema, data)
    return data


def create_plan(
    query: str,
    tracer: Tracer,
    *,
    client: OpenAI | None = None,
    effort: EffortLevel | None = None,
    model: str = DEFAULT_MODEL,
    corpus_searcher: Any | None = None,
) -> dict[str, Any]:
    """Decompose a research query into subtasks for parallel execution."""
    if client is None:
        client = OpenAI()
    if effort is None:
        effort = classify_effort(query)

    tracer.log("lead", "plan_start", query=query, effort=effort.name)

    # Corpus pre-check: is this query answerable from local_corpus?
    from multi_agent.corpus_relevance import check_corpus_relevance, format_relevance_context
    relevance = check_corpus_relevance(query, searcher=corpus_searcher)
    tracer.log("lead", "corpus_precheck",
               state=relevance.state,
               matches=len(relevance.matching_papers),
               reason=relevance.reason)
    relevance_context = format_relevance_context(relevance)

    prompt = LEAD_PLAN_PROMPT.format(
        query=query,
        effort_level=effort.name,
        max_subagents=effort.max_subagents,
        max_tool_calls=effort.max_tool_calls_per_agent,
    )
    prompt = f"{relevance_context}\n\n{prompt}"

    plan = _call_llm(client, LEAD_SYSTEM_PROMPT, prompt, model, tracer=tracer, agent_id="lead", schema=PlanSchema)

    subtasks = plan.get("subtasks", [])
    if not subtasks:
        raise PlanningError("Lead agent produced no subtasks")

    # Cap subtasks to effort level
    subtasks = subtasks[: effort.max_subagents]

    # Enforce the pre-check routing budget deterministically — the LLM does not
    # reliably obey "spawn only N" from the prompt alone. full_text_match means
    # the corpus has the depth (1 agent); abstract_only means corpus + arxiv (2).
    # BUT: some queries need external sources even when the corpus matches —
    # citation counts (semantic_scholar), comparisons, and recent-year requests
    # (arxiv/web). For those, skip the cap so external agents aren't starved.
    precheck_cap = {"full_text_match": 1, "abstract_only": 2}.get(relevance.state)
    if precheck_cap is not None and _needs_external_despite_corpus(query):
        tracer.log("lead", "precheck_cap_skipped",
                   state=relevance.state, reason="external_intent")
        precheck_cap = None
    if precheck_cap is not None and len(subtasks) > precheck_cap:
        tracer.log("lead", "precheck_cap_applied",
                   state=relevance.state, before=len(subtasks), after=precheck_cap)
        subtasks = subtasks[:precheck_cap]

    plan["subtasks"] = subtasks
    plan["corpus_precheck"] = {
        "state": relevance.state,
        "matching_papers": relevance.matching_papers,
        "topics": relevance.topics,
    }

    tracer.log(
        "lead",
        "plan_complete",
        subtask_count=len(subtasks),
        reasoning=plan.get("reasoning", ""),
    )
    return plan


def _build_findings_text(store: FindingsStore) -> str:
    """Render completed subagent findings into text for an LLM prompt."""
    completed = store.get_completed()
    findings_parts: list[str] = []
    for result in completed:
        findings_parts.append(f"\n--- Agent: {result.agent_type} | Task: {result.subtask} ---")
        for f in result.findings[:10]:
            meta_parts = []
            if f.url:
                meta_parts.append(f"URL: {f.url}")
            citation_count = f.metadata.get("citation_count")
            if citation_count:
                meta_parts.append(f"Citations: {citation_count}")
            published = f.metadata.get("published_date")
            if published:
                meta_parts.append(f"Published: {published}")
            meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
            findings_parts.append(f"  • [{f.source}] {f.title}{meta_str}: {f.content[:300]}")
    return "\n".join(findings_parts)


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
    total_findings = sum(len(r.findings) for r in completed)
    if not completed or total_findings == 0:
        # Don't hand an LLM an empty findings block and trust it to say "I
        # don't know" — with nothing to ground on, it will still sometimes
        # answer from its own training data instead (e.g. inventing a
        # specific citation count), which the prompt's "don't guess" rule
        # doesn't reliably prevent. Skip the LLM call entirely here.
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
    findings_text = _build_findings_text(store)

    tracer.log("lead", "synthesis_start", agent_count=len(completed))

    prompt = LEAD_SYNTHESIS_PROMPT.format(
        query=query,
        findings_text=findings_text,
    )

    synthesis = _call_llm(client, LEAD_SYSTEM_PROMPT, prompt, model, tracer=tracer, agent_id="lead", schema=SynthesisSchema)
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

    plan = _call_llm(client, LEAD_SYSTEM_PROMPT, prompt, model, tracer=tracer, agent_id="lead", schema=FollowUpPlanSchema)
    subtasks = plan.get("subtasks", [])[:effort.max_subagents]
    plan["subtasks"] = subtasks

    tracer.log("lead", "follow_up_complete", subtask_count=len(subtasks))
    return plan


REFINEMENT_PROMPT = """\
You are refining a research synthesis based on judge feedback.

Original query: {query}

Current synthesis:
{current_synthesis}

Subagent findings (the ONLY evidence you may cite):
{findings_text}

Judge feedback:
{judge_feedback}

Judge scores: {judge_scores}

Instructions:
1. Address each issue raised by the judge
2. Improve factual accuracy by re-checking claims against the findings above
3. Fix citation issues — ensure every claim maps to a finding
4. Fill completeness gaps if the findings support it
5. Do NOT add claims not supported by the findings above
6. NEVER return an empty synthesis — if evidence is thin, synthesize what IS \
present and note the gaps. An empty answer is always worse than the original.

Return ONLY valid JSON in the same format as the original synthesis, no markdown fences.\
"""


def refine_synthesis(
    query: str,
    synthesis: dict[str, Any],
    judge_scores: dict[str, Any],
    store: FindingsStore,
    tracer: Tracer,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Refine synthesis based on judge feedback (evaluator-optimizer pattern)."""
    if client is None:
        client = OpenAI()

    tracer.log("lead", "refinement_start",
               original_score=judge_scores.get("overall", 0.0))

    findings_text = _build_findings_text(store)

    prompt = REFINEMENT_PROMPT.format(
        query=query,
        current_synthesis=synthesis.get("synthesis", ""),
        findings_text=findings_text,
        judge_feedback=judge_scores.get("reasoning", ""),
        judge_scores=json.dumps({
            k: v for k, v in judge_scores.items()
            if k in ("factual_accuracy", "citation_accuracy", "completeness",
                      "source_quality", "tool_efficiency")
        }),
    )

    refined = _call_llm(client, LEAD_SYSTEM_PROMPT, prompt, model, tracer=tracer, agent_id="lead", schema=SynthesisSchema)
    tracer.log("lead", "refinement_complete")
    return refined
