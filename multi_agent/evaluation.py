"""End-to-end evaluation harness for the multi-agent research system.

Runs test queries through the full pipeline and reports:
- Primary: judge pass rate and average overall score
- Diagnostic: routing hint match rate, tool coverage
- Failure traces: saved to data/eval_traces/ for root-causing
- Human review queue: lowest-scoring + random passing cases

Usage:
    python3 -m multi_agent.evaluation
    python3 -m multi_agent.evaluation --fixture path/to/cases.json --limit 5
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from ingestion.embed import load_env_file


DEFAULT_FIXTURE = Path("tests/fixtures/multi_agent_tool_eval.json")
DEFAULT_RESULTS_OUTPUT = Path("data/multi_agent_eval_results.json")
DEFAULT_HISTORY_OUTPUT = Path("data/multi_agent_eval_history.jsonl")
DEFAULT_REVIEW_OUTPUT = Path("data/human_review_queue.json")
DEFAULT_TRACE_DIR = Path("data/eval_traces")

ALL_KNOWN_SOURCES = {"local_corpus", "arxiv", "semantic_scholar", "web"}

TOOL_KEYWORDS = {
    "local_corpus": ["local_corpus", "local", "corpus", "qdrant"],
    "arxiv": ["arxiv"],
    "semantic_scholar": ["semantic_scholar", "s2", "semantic scholar"],
    "web": ["tavily", "web"],
}


@dataclass
class ToolEvalCase:
    id: str
    query: str
    category: str
    expected_primary_source: list[str]
    expected_precheck: str | None = None
    expected_guardrail_blocked: bool = False


@dataclass
class CaseResult:
    case_id: str
    query: str
    category: str
    expected_primary_source: list[str]
    judge_overall: float = 0.0
    judge_pass: bool = False
    judge_reasoning: str = ""
    sources_assigned: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    routing_hint_matched: bool = False
    answer_snippet: str = ""
    elapsed_seconds: float = 0.0
    error: str | None = None
    guardrail_blocked: bool = False
    expected_precheck: str | None = None
    actual_precheck: str | None = None
    precheck_matched: bool | None = None
    expected_guardrail_blocked: bool = False
    guardrail_correct: bool | None = None
    hallucination_flags: list[str] = field(default_factory=list)


def load_tool_eval_cases(path: Path = DEFAULT_FIXTURE) -> list[ToolEvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for item in raw:
        expected = item.get("expected_primary_source", [])
        if isinstance(expected, str):
            expected = [expected]
        cases.append(ToolEvalCase(
            id=item["id"],
            query=item["query"],
            category=item["category"],
            expected_primary_source=expected,
            expected_precheck=item.get("expected_precheck"),
            expected_guardrail_blocked=item.get("expected_guardrail_blocked", False),
        ))
    return cases


def extract_sources(result: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Extract assigned sources (from plan) and actual sources (from findings)."""
    assigned = set()
    plan = result.get("plan", {})
    for st in plan.get("subtasks", []):
        if isinstance(st, dict):
            src = st.get("source", "")
            if src:
                assigned.add(src.lower())

    used = set()
    store_summary = result.get("store_summary", {})
    for agent_result in store_summary.get("agents", []):
        if isinstance(agent_result, dict):
            agent_type = agent_result.get("agent_type", "").lower()
            if agent_type:
                used.add(agent_type)
            for f in agent_result.get("findings", []):
                source = f.get("source", "").lower() if isinstance(f, dict) else ""
                if source:
                    used.add(source)

    return assigned, used


def check_routing_hint(
    expected: list[str],
    assigned: set[str],
    used: set[str],
) -> bool:
    """Check if any expected source was used (diagnostic, not a gate)."""
    all_signals = assigned | used
    for exp in expected:
        keywords = TOOL_KEYWORDS.get(exp, [exp])
        for src in all_signals:
            if any(kw in src for kw in keywords):
                return True
    return False


def run_tool_eval(
    cases: list[ToolEvalCase],
    *,
    run_fn: Callable[..., dict[str, Any]] | None = None,
    client: OpenAI | None = None,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    fail_threshold: float = 0.5,
    pace_seconds: float = 8.0,
    rate_limit_retries: int = 3,
) -> dict[str, Any]:
    """Run evaluation across all cases and compute metrics.

    pace_seconds spaces out cases to stay under per-minute token limits;
    rate_limit_retries retries a case that still hits a 429 after backing off.
    """
    if run_fn is None:
        from multi_agent.orchestrator import run_research
        run_fn = run_research
        if client is None:
            # SDK-level exponential backoff smooths transient 429s within a case.
            client = OpenAI(max_retries=6)

    trace_dir.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    all_sources_invoked: set[str] = set()

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case.id}: {case.query[:60]}...")
        cr = CaseResult(
            case_id=case.id,
            query=case.query,
            category=case.category,
            expected_primary_source=case.expected_primary_source,
        )

        if i > 1 and pace_seconds > 0:
            time.sleep(pace_seconds)

        start = time.time()
        kwargs = {"openai_client": client} if client else {}
        result = None
        for attempt in range(rate_limit_retries + 1):
            try:
                result = run_fn(case.query, **kwargs)
                break
            except Exception as exc:
                is_rate_limit = "rate_limit" in str(exc).lower() or "429" in str(exc)
                if is_rate_limit and attempt < rate_limit_retries:
                    wait = 20.0 * (attempt + 1)
                    print(f"  rate-limited, waiting {wait:.0f}s (retry {attempt + 1}/{rate_limit_retries})...")
                    time.sleep(wait)
                    continue
                cr.elapsed_seconds = time.time() - start
                cr.error = str(exc)
                print(f"  ERROR: {exc}")
                break
        if result is None:
            results.append(cr)
            continue
        cr.elapsed_seconds = time.time() - start

        # Check guardrail
        guardrail = result.get("guardrail", {})
        actual_blocked = not guardrail.get("safe", True)
        cr.guardrail_blocked = actual_blocked
        cr.expected_guardrail_blocked = case.expected_guardrail_blocked
        cr.guardrail_correct = actual_blocked == case.expected_guardrail_blocked
        if actual_blocked:
            cr.judge_reasoning = f"Blocked: {guardrail.get('reason', '')}"
            results.append(cr)
            tag = "expected" if cr.guardrail_correct else "UNEXPECTED (false positive)"
            print(f"  BLOCKED by guardrail: {guardrail.get('category')} [{tag}]")
            continue
        elif case.expected_guardrail_blocked:
            print("  WARNING: expected guardrail block but query passed through (regression)")

        # Extract judge scores
        judge = result.get("judge_scores", {})
        cr.judge_overall = judge.get("overall", 0.0)
        cr.judge_pass = judge.get("pass", False)
        cr.judge_reasoning = judge.get("reasoning", "")

        # Extract sources
        assigned, used = extract_sources(result)
        cr.sources_assigned = sorted(assigned)
        cr.sources_used = sorted(used)
        all_sources_invoked |= assigned | used

        # Routing hint check
        cr.routing_hint_matched = check_routing_hint(
            case.expected_primary_source, assigned, used
        )

        # Pre-check tier: did Qdrant-first routing land on the expected tier?
        cr.expected_precheck = case.expected_precheck
        plan = result.get("plan", {})
        precheck = plan.get("corpus_precheck", {}) if isinstance(plan, dict) else {}
        cr.actual_precheck = precheck.get("state")
        if case.expected_precheck is not None:
            cr.precheck_matched = cr.actual_precheck == case.expected_precheck

        # Answer snippet
        synthesis = result.get("synthesis", {})
        answer = synthesis.get("synthesis", "") if isinstance(synthesis, dict) else ""
        cr.answer_snippet = answer[:200]

        # Hallucination flags (numeric claims not grounded in any finding)
        cited_report = result.get("cited_report", {})
        cr.hallucination_flags = (
            cited_report.get("hallucination_flags", []) if isinstance(cited_report, dict) else []
        )

        # Print agent details
        store_summary = result.get("store_summary", {})
        agents = store_summary.get("agents", [])
        total_findings = store_summary.get("total_findings", 0)
        print(f"  Agents: {len(agents)} | Findings: {total_findings} | Latency: {cr.elapsed_seconds:.1f}s")
        for agent in agents:
            agent_type = agent.get("agent_type", "?")
            agent_status = agent.get("status", "?")
            agent_findings = len(agent.get("findings", []))
            print(f"    → {agent_type}: {agent_findings} findings ({agent_status})")

        # Save trace on failure
        if not cr.judge_pass or cr.judge_overall < fail_threshold:
            trace_path = trace_dir / f"{case.id}.json"
            trace_path.write_text(json.dumps({
                "case_id": case.id,
                "query": case.query,
                "judge_scores": judge,
                "plan": result.get("plan", {}),
                "store_summary": result.get("store_summary", {}),
                "trace_events": result.get("trace_events", ""),
            }, indent=2, default=str), encoding="utf-8")

        status = "PASS" if cr.judge_pass else "FAIL"
        route = "OK" if cr.routing_hint_matched else "MISS"
        pc = ""
        if cr.expected_precheck is not None:
            tier = "OK" if cr.precheck_matched else f"MISS(got {cr.actual_precheck})"
            pc = f" | Precheck: {tier}"
        hc = f" | Hallucinations: {cr.hallucination_flags}" if cr.hallucination_flags else ""
        print(f"  Judge: {cr.judge_overall:.2f} ({status}) | Route: {route}{pc} | Sources: {cr.sources_assigned}{hc}")
        results.append(cr)

    # Compute aggregate metrics
    valid = [r for r in results if not r.error and not r.guardrail_blocked]
    pass_count = sum(1 for r in valid if r.judge_pass)
    route_match_count = sum(1 for r in valid if r.routing_hint_matched)
    precheck_checked = [r for r in valid if r.precheck_matched is not None]
    precheck_match_count = sum(1 for r in precheck_checked if r.precheck_matched)
    untested = sorted(ALL_KNOWN_SOURCES - all_sources_invoked)
    guardrail_checked = [r for r in results if not r.error]
    guardrail_correct_count = sum(1 for r in guardrail_checked if r.guardrail_correct)
    hallucination_cases = sum(1 for r in valid if r.hallucination_flags)

    report = {
        "total_cases": len(cases),
        "completed": len(valid),
        "errors": sum(1 for r in results if r.error),
        "guardrail_blocked": sum(1 for r in results if r.guardrail_blocked),
        "guardrail_accuracy": round(guardrail_correct_count / len(guardrail_checked), 3) if guardrail_checked else None,
        "pass_rate": round(pass_count / len(valid), 3) if valid else 0.0,
        "avg_judge_overall": round(sum(r.judge_overall for r in valid) / len(valid), 3) if valid else 0.0,
        "routing_hint_match_rate": round(route_match_count / len(valid), 3) if valid else 0.0,
        "precheck_match_rate": round(precheck_match_count / len(precheck_checked), 3) if precheck_checked else None,
        "precheck_cases_checked": len(precheck_checked),
        "untested_tools": untested,
        "all_sources_invoked": sorted(all_sources_invoked),
        "avg_elapsed_seconds": round(sum(r.elapsed_seconds for r in valid) / len(valid), 1) if valid else 0.0,
        "hallucination_flagged_cases": hallucination_cases,
        "cases": [asdict(r) for r in results],
    }

    return report


def build_human_review_queue(
    report: dict[str, Any],
    sample_size: int = 5,
) -> list[dict[str, Any]]:
    """Build a queue of cases for human review."""
    cases = report.get("cases", [])
    valid = [c for c in cases if not c.get("error") and not c.get("guardrail_blocked")]

    sorted_by_score = sorted(valid, key=lambda c: c.get("judge_overall", 0.0))

    # Take the worst cases
    worst = sorted_by_score[:min(3, len(sorted_by_score))]

    # Add random passing cases for calibration
    passing = [c for c in valid if c.get("judge_pass")]
    random_passing = random.sample(passing, min(2, len(passing))) if passing else []

    queue = []
    seen = set()
    for c in worst + random_passing:
        if c["case_id"] not in seen:
            seen.add(c["case_id"])
            queue.append({
                "case_id": c["case_id"],
                "query": c["query"],
                "answer_snippet": c.get("answer_snippet", ""),
                "judge_overall": c.get("judge_overall", 0.0),
                "judge_reasoning": c.get("judge_reasoning", ""),
                "human_verdict": None,
                "human_notes": "",
            })

    return queue[:sample_size]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_OUTPUT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument(
        "--pace-seconds", type=float,
        default=float(os.environ.get("RSE_EVAL_PACE_SECONDS", 8.0)),
        help="Seconds to sleep between cases to avoid rate limits (default 8.0, "
             "or $RSE_EVAL_PACE_SECONDS). Lower it if your OpenAI TPM limit has headroom.",
    )
    args = parser.parse_args()

    load_env_file(Path(".env"))
    cases = load_tool_eval_cases(args.fixture)
    if args.limit:
        cases = cases[:args.limit]

    # Warm up the reranker's cross-encoder model before timing starts — it's
    # lazily downloaded/loaded from Hugging Face on first use, and that cold
    # start can eat enough of case 1's time budget to make its search return
    # empty (case 1 looks like a routing/search bug when it's really just an
    # untimed download racing the request).
    print("Warming up reranker model...")
    from retrieval.rerank import load_cross_encoder
    load_cross_encoder()

    print(f"Running {len(cases)} eval cases...")
    report = run_tool_eval(cases, trace_dir=args.trace_dir, pace_seconds=args.pace_seconds)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {args.output}")

    # Append to history
    history_line = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_cases": report["total_cases"],
        "pass_rate": report["pass_rate"],
        "avg_judge_overall": report["avg_judge_overall"],
        "routing_hint_match_rate": report["routing_hint_match_rate"],
        "precheck_match_rate": report.get("precheck_match_rate"),
        "guardrail_accuracy": report.get("guardrail_accuracy"),
        "untested_tools": report["untested_tools"],
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with open(args.history, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_line) + "\n")
    print(f"History appended to {args.history}")

    # Build and save human review queue
    queue = build_human_review_queue(report)
    args.review.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    print(f"Human review queue saved to {args.review} ({len(queue)} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Cases: {report['completed']}/{report['total_cases']} completed")
    print(f"Pass rate: {report['pass_rate']:.1%}")
    print(f"Avg judge score: {report['avg_judge_overall']:.3f}")
    print(f"Routing hint match: {report['routing_hint_match_rate']:.1%}")
    if report.get("precheck_match_rate") is not None:
        print(f"Pre-check tier match: {report['precheck_match_rate']:.1%} ({report['precheck_cases_checked']} corpus cases)")
    if report.get("guardrail_accuracy") is not None:
        print(f"Guardrail accuracy: {report['guardrail_accuracy']:.1%}")
    print(f"Avg latency: {report['avg_elapsed_seconds']:.1f}s")
    if report.get("hallucination_flagged_cases"):
        print(f"WARNING: unverified numeric claims in {report['hallucination_flagged_cases']} case(s)")
    if report["untested_tools"]:
        print(f"WARNING: untested tools: {report['untested_tools']}")
    else:
        print(f"All 4 tools exercised: {report['all_sources_invoked']}")
    if report["errors"]:
        print(f"Errors: {report['errors']}")
    if report["guardrail_blocked"]:
        print(f"Guardrail blocked: {report['guardrail_blocked']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
