"""Benchmark: single-agent vs multi-agent research performance comparison.

Runs the same set of queries through both the existing single-agent pipeline
and the new multi-agent pipeline, comparing quality scores from LLM-as-judge.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from multi_agent.orchestrator import run_research
from multi_agent.config import classify_effort
from multi_agent.judge import evaluate_output
from multi_agent.trace import Tracer


DEFAULT_EVAL_QUERIES = Path("tests/fixtures/eval_queries_100_chunk_grounded.json")
DEFAULT_RESULTS_OUTPUT = Path("data/multi_agent_benchmark_results.json")

# Subset of eval queries suitable for multi-agent testing (complex queries)
MULTI_AGENT_EVAL_QUERIES = [
    "Compare the effectiveness of different RAG architectures for scientific literature review",
    "What are the trade-offs between dense and sparse retrieval methods in hybrid search systems?",
    "How do transformer-based models handle hallucination detection and mitigation?",
    "Survey the state of multi-agent systems for automated research tasks",
    "What are the main challenges in fine-tuning large language models for domain-specific applications?",
    "Analyze the relationship between retrieval augmented generation and knowledge graph approaches",
    "What datasets and benchmarks are used to evaluate RAG system performance?",
    "Compare attention mechanisms used in different transformer architectures",
    "How do agentic AI systems coordinate tool use across multiple steps?",
    "What are the current limitations of vector similarity search for academic paper retrieval?",
    "Explain the differences between RLHF, DPO, and PPO for language model alignment",
    "How has the field of prompt engineering evolved since 2023?",
    "What are the key findings on scaling laws for large language models?",
    "Compare chunking strategies for document retrieval in RAG systems",
    "How do multi-modal AI systems combine text and image understanding?",
    "What approaches exist for reducing inference latency in LLM applications?",
    "Survey methods for evaluating faithfulness and groundedness in generated text",
    "How do knowledge distillation techniques transfer capabilities from large to small models?",
    "What are the architectural differences between encoder-only, decoder-only, and encoder-decoder transformers?",
    "Compare approaches to long-context handling in modern language models",
]


@dataclass
class BenchmarkResult:
    query: str
    effort_level: str
    # Multi-agent results
    multi_agent_synthesis: str = ""
    multi_agent_judge_scores: dict = field(default_factory=dict)
    multi_agent_elapsed: float = 0.0
    multi_agent_agents_used: int = 0
    multi_agent_findings_count: int = 0
    multi_agent_tokens_used: int = 0
    # Single-agent results (from existing agentic pipeline)
    single_agent_synthesis: str = ""
    single_agent_judge_scores: dict = field(default_factory=dict)
    single_agent_elapsed: float = 0.0
    # Comparison
    improvement: float = 0.0
    error: str | None = None


def run_single_agent(query: str, client: OpenAI) -> dict[str, Any]:
    """Run existing single-agent pipeline for comparison."""
    try:
        import sys
        if sys.version_info < (3, 11):
            return {
                "synthesis": "",
                "elapsed": 0.0,
                "error": "single-agent graph requires Python 3.11+ (NotRequired)",
            }

        from agent.research_graph import run_research_agent

        start = time.time()
        state = run_research_agent(query)
        elapsed = time.time() - start

        synthesis = ""
        brief = state.get("brief")
        if brief:
            if hasattr(brief, "direct_answer"):
                synthesis = brief.direct_answer
            elif isinstance(brief, dict):
                synthesis = brief.get("direct_answer", str(brief))
            else:
                synthesis = str(brief)

        return {
            "synthesis": synthesis,
            "elapsed": elapsed,
            "error": None,
        }
    except Exception as exc:
        return {
            "synthesis": "",
            "elapsed": 0.0,
            "error": str(exc),
        }


def run_benchmark(
    queries: list[str] | None = None,
    output_path: Path = DEFAULT_RESULTS_OUTPUT,
    limit: int | None = None,
) -> list[BenchmarkResult]:
    """Run benchmark comparing single-agent vs multi-agent."""
    if queries is None:
        queries = MULTI_AGENT_EVAL_QUERIES

    if limit is not None:
        queries = queries[:limit]

    client = OpenAI()
    results: list[BenchmarkResult] = []

    print(f"Running benchmark on {len(queries)} queries...")

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query[:60]}...")
        result = BenchmarkResult(
            query=query,
            effort_level=classify_effort(query).name,
        )

        # Run multi-agent
        try:
            start = time.time()
            ma_result = run_research(query, openai_client=client)
            result.multi_agent_elapsed = time.time() - start
            result.multi_agent_synthesis = ma_result.get("synthesis", {}).get("synthesis", "")
            result.multi_agent_judge_scores = ma_result.get("judge_scores", {})
            result.multi_agent_agents_used = ma_result.get("store_summary", {}).get("total_agents", 0)
            result.multi_agent_findings_count = ma_result.get("store_summary", {}).get("total_findings", 0)
            result.multi_agent_tokens_used = ma_result.get("store_summary", {}).get("total_tokens", 0)
        except Exception as exc:
            result.error = f"Multi-agent failed: {exc}"
            print(f"  Multi-agent ERROR: {exc}")

        # Run single-agent
        try:
            sa_result = run_single_agent(query, client)
            result.single_agent_elapsed = sa_result["elapsed"]
            result.single_agent_synthesis = sa_result["synthesis"]

            if sa_result["synthesis"]:
                tracer = Tracer()
                sa_judge = evaluate_output(
                    query,
                    {"cited_report": sa_result["synthesis"], "references": [], "uncited_claims": []},
                    {"total_agents": 1, "total_findings": 0, "elapsed_seconds": sa_result["elapsed"]},
                    tracer,
                    client=client,
                )
                result.single_agent_judge_scores = sa_judge
        except Exception as exc:
            if not result.error:
                result.error = f"Single-agent failed: {exc}"
            print(f"  Single-agent ERROR: {exc}")

        # Calculate improvement
        ma_overall = result.multi_agent_judge_scores.get("overall", 0.0)
        sa_overall = result.single_agent_judge_scores.get("overall", 0.0)
        if sa_overall > 0:
            result.improvement = ((ma_overall - sa_overall) / sa_overall) * 100
        elif ma_overall > 0:
            result.improvement = 100.0

        print(f"  MA: {ma_overall:.2f} | SA: {sa_overall:.2f} | Δ: {result.improvement:+.1f}%")
        print(f"  MA time: {result.multi_agent_elapsed:.1f}s | SA time: {result.single_agent_elapsed:.1f}s")

        results.append(result)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str),
        encoding="utf-8",
    )

    # Print summary
    _print_summary(results)
    return results


def _print_summary(results: list[BenchmarkResult]) -> None:
    """Print benchmark summary statistics."""
    valid = [r for r in results if not r.error]
    if not valid:
        print("\nNo valid results to summarize.")
        return

    ma_scores = [r.multi_agent_judge_scores.get("overall", 0) for r in valid]
    sa_scores = [r.single_agent_judge_scores.get("overall", 0) for r in valid]
    improvements = [r.improvement for r in valid]
    ma_times = [r.multi_agent_elapsed for r in valid]
    sa_times = [r.single_agent_elapsed for r in valid]

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"Queries evaluated: {len(valid)}/{len(results)}")
    print(f"\nMulti-agent avg score:  {sum(ma_scores)/len(ma_scores):.3f}")
    print(f"Single-agent avg score: {sum(sa_scores)/len(sa_scores):.3f}")
    print(f"Avg improvement:        {sum(improvements)/len(improvements):+.1f}%")
    print(f"\nMulti-agent avg time:   {sum(ma_times)/len(ma_times):.1f}s")
    print(f"Single-agent avg time:  {sum(sa_times)/len(sa_times):.1f}s")
    print(f"\nMulti-agent avg agents: {sum(r.multi_agent_agents_used for r in valid)/len(valid):.1f}")
    print(f"Multi-agent avg findings: {sum(r.multi_agent_findings_count for r in valid)/len(valid):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_OUTPUT)
    args = parser.parse_args()

    from ingestion.embed import load_env_file
    load_env_file(Path(".env"))

    run_benchmark(limit=args.limit, output_path=args.output)
