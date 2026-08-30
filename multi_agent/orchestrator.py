"""Orchestrator — runs subagents in parallel and manages the research loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from openai import OpenAI

from multi_agent.config import (
    DEFAULT_MODEL,
    SUBAGENT_TIMEOUT_SECONDS,
    classify_effort,
)
from multi_agent.citation import add_citations
from multi_agent.findings_store import FindingsStore
from multi_agent.guardrails import check_guardrails
from multi_agent.judge import evaluate_output
from multi_agent.lead import create_plan, synthesize_findings, plan_follow_up, refine_synthesis
from multi_agent.subagent import run_subagent
from multi_agent.trace import Tracer
from agentic.external import ExternalSearchClient, DEFAULT_EXTERNAL_CLIENT


def _run_subagents_parallel(
    subtasks: list[dict[str, Any]],
    store: FindingsStore,
    tracer: Tracer,
    openai_client: OpenAI,
    external_client: ExternalSearchClient,
    max_tool_calls: int,
) -> None:
    """Run subagents concurrently using a thread pool."""
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(subtasks), 5)
    ) as executor:
        futures = []
        for subtask in subtasks:
            future = executor.submit(
                run_subagent,
                subtask,
                store,
                tracer,
                openai_client=openai_client,
                external_client=external_client,
                max_tool_calls=max_tool_calls,
            )
            futures.append(future)

        concurrent.futures.wait(
            futures, timeout=SUBAGENT_TIMEOUT_SECONDS + 10
        )

        for future in futures:
            if future.exception():
                tracer.log(
                    "orchestrator",
                    "subagent_error",
                    error=str(future.exception()),
                )


def run_research(
    query: str,
    *,
    openai_client: OpenAI | None = None,
    external_client: ExternalSearchClient | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Execute the full multi-agent research pipeline.

    Flow:
    1. Lead agent plans subtasks
    2. Subagents execute in parallel
    3. Lead synthesizes findings
    4. If gaps remain and iterations allow, spawn more subagents
    5. Citation agent attributes sources
    6. Return final report with traces
    """
    if openai_client is None:
        openai_client = OpenAI()
    if external_client is None:
        external_client = DEFAULT_EXTERNAL_CLIENT

    store = FindingsStore()
    tracer = Tracer()
    effort = classify_effort(query)

    tracer.log("orchestrator", "start", query=query, effort=effort.name)

    # Step 0 + Step 1: Guardrail check runs in parallel with planning
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        guardrail_future = executor.submit(check_guardrails, query)
        plan_future = executor.submit(
            create_plan, query, tracer, client=openai_client, effort=effort, model=model
        )

        guardrail_result = guardrail_future.result(timeout=10)
        tracer.log("guardrail", "complete", safe=guardrail_result.safe,
                    category=guardrail_result.category, reason=guardrail_result.reason)

        if not guardrail_result.safe:
            return {
                "query": query,
                "synthesis": {"synthesis": guardrail_result.reason, "confidence": "n/a"},
                "cited_report": {"cited_report": guardrail_result.reason, "references": [], "uncited_claims": []},
                "judge_scores": {"overall": 0.0, "pass": False, "reasoning": f"Blocked by guardrail: {guardrail_result.category}"},
                "store_summary": store.summary(),
                "trace": tracer.summary(),
                "trace_events": tracer.to_json(),
                "effort_level": effort.name,
                "plan": {"subtasks": []},
                "guardrail": {"safe": guardrail_result.safe, "category": guardrail_result.category, "reason": guardrail_result.reason},
            }

        plan = plan_future.result(timeout=SUBAGENT_TIMEOUT_SECONDS)
    subtasks = plan.get("subtasks", [])
    executed_subtasks = list(subtasks)

    # Step 2: Execute subagents in parallel
    _run_subagents_parallel(
        subtasks,
        store,
        tracer,
        openai_client,
        external_client,
        effort.max_tool_calls_per_agent,
    )

    # Step 3: Synthesize
    synthesis = synthesize_findings(
        query, store, tracer, client=openai_client, model=model
    )

    # Step 4: Iterative loop — spawn more if needed
    iteration = 1
    while (
        synthesis.get("needs_more_research", False)
        and iteration < effort.max_iterations
    ):
        tracer.log("orchestrator", "iteration", number=iteration + 1)

        follow_up = plan_follow_up(
            query, synthesis, tracer, client=openai_client, effort=effort, model=model
        )
        follow_up_subtasks = follow_up.get("subtasks", [])
        if not follow_up_subtasks:
            break

        executed_subtasks.extend(follow_up_subtasks)
        _run_subagents_parallel(
            follow_up_subtasks,
            store,
            tracer,
            openai_client,
            external_client,
            effort.max_tool_calls_per_agent,
        )

        synthesis = synthesize_findings(
            query, store, tracer, client=openai_client, model=model
        )
        iteration += 1

    tracer.log("orchestrator", "complete", iterations=iteration)

    # Step 5: Citation agent
    cited_report = add_citations(
        synthesis, store, tracer, client=openai_client, model=model
    )

    # Step 6: Judge evaluation
    judge_scores = evaluate_output(
        query, cited_report, store.summary(), tracer, client=openai_client, model=model
    )

    # Step 7: Evaluator-optimizer — if judge scores low, refine and re-judge once.
    # Keep the refinement ONLY if it actually improves the score and is non-empty;
    # otherwise fall back to the original (refinement can degrade or empty the answer).
    REFINEMENT_THRESHOLD = 0.6
    if judge_scores.get("overall", 1.0) < REFINEMENT_THRESHOLD and judge_scores.get("pass") is not True:
        tracer.log("orchestrator", "refinement_triggered",
                   original_score=judge_scores.get("overall", 0.0))

        original_score = judge_scores.get("overall", 0.0)
        original_synthesis = synthesis
        original_cited = cited_report

        refined_synthesis = refine_synthesis(
            query, synthesis, judge_scores, store, tracer,
            client=openai_client, model=model,
        )

        refined_candidate = (refined_synthesis or {}).get("synthesis", "")
        refined_answer = refined_candidate.strip() if isinstance(refined_candidate, str) else ""
        if not refined_answer:
            tracer.log("orchestrator", "refinement_rejected", reason="empty_or_malformed_synthesis")
        else:
            refined_cited = add_citations(
                refined_synthesis, store, tracer, client=openai_client, model=model
            )
            refined_scores = evaluate_output(
                query, refined_cited, store.summary(), tracer, client=openai_client, model=model
            )
            if refined_scores.get("overall", 0.0) > original_score:
                synthesis = refined_synthesis
                cited_report = refined_cited
                judge_scores = refined_scores
                tracer.log("orchestrator", "refinement_accepted",
                           new_score=refined_scores.get("overall", 0.0))
            else:
                synthesis = original_synthesis
                cited_report = original_cited
                tracer.log("orchestrator", "refinement_rejected",
                           reason="no_improvement",
                           original_score=original_score,
                           refined_score=refined_scores.get("overall", 0.0))

    return {
        "query": query,
        "synthesis": synthesis,
        "cited_report": cited_report,
        "judge_scores": judge_scores,
        "store_summary": store.summary(),
        "trace": tracer.summary(),
        "trace_events": tracer.to_json(),
        "effort_level": effort.name,
        "plan": {**plan, "subtasks": executed_subtasks},
        "guardrail": {"safe": True, "category": "safe", "reason": "Query passed all checks."},
    }


async def run_research_async(
    query: str,
    *,
    openai_client: OpenAI | None = None,
    external_client: ExternalSearchClient | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Async wrapper for run_research (runs in executor to not block event loop)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_research(
            query,
            openai_client=openai_client,
            external_client=external_client,
            model=model,
        ),
    )
