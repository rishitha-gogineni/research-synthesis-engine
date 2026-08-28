"""Multi-Agent Research tab for the Streamlit UI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def render_multi_agent_page():
    """Render the multi-agent research interface."""
    st.title("Multi-Agent Research")
    st.caption(
        "Orchestrator-worker pattern: a lead agent decomposes your query, "
        "spawns parallel subagents, synthesizes findings, and scores quality."
    )

    query = st.text_area(
        "Research query",
        placeholder="e.g., Compare the trade-offs between dense and sparse retrieval in hybrid search systems",
        height=80,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        run_full = st.button("Run Full Pipeline", type="primary", use_container_width=True)
    with col2:
        run_plan = st.button("Preview Plan Only", use_container_width=True)

    if run_plan and query:
        _run_plan_preview(query)

    if run_full and query:
        _run_full_research(query)


def _run_plan_preview(query: str):
    """Show the lead agent's plan without executing subagents."""
    with st.spinner("Lead agent planning..."):
        try:
            from multi_agent.lead import create_plan
            from multi_agent.trace import Tracer
            from multi_agent.config import classify_effort
            from openai import OpenAI

            tracer = Tracer()
            effort = classify_effort(query)
            client = OpenAI()
            plan = create_plan(query, tracer, client=client, effort=effort)

            st.success(f"Effort level: **{effort.name}** ({effort.max_subagents} subagents)")
            st.markdown(f"**Reasoning:** {plan.get('reasoning', 'N/A')}")

            st.markdown("### Subtasks")
            for i, task in enumerate(plan.get("subtasks", []), 1):
                with st.expander(f"Subtask {i}: {task.get('source', '?')} — {task.get('objective', '')[:60]}"):
                    st.json(task)
        except Exception as exc:
            st.error(f"Planning failed: {exc}")


def _run_full_research(query: str):
    """Execute the full multi-agent pipeline."""
    progress = st.progress(0, text="Starting multi-agent research...")

    try:
        from multi_agent.orchestrator import run_research
        from openai import OpenAI

        progress.progress(10, text="Lead agent planning...")
        start = time.time()
        result = run_research(query, openai_client=OpenAI())
        elapsed = time.time() - start
        progress.progress(100, text=f"Complete in {elapsed:.1f}s")

        # Synthesis
        st.markdown("---")
        st.markdown("### Research Synthesis")
        synthesis = result.get("synthesis", {})
        st.markdown(synthesis.get("synthesis", "No synthesis generated."))

        # Key themes
        themes = synthesis.get("key_themes", [])
        if themes:
            st.markdown("**Key themes:** " + ", ".join(themes))

        # Confidence
        confidence = synthesis.get("confidence", "unknown")
        if confidence == "high":
            st.success(f"Confidence: {confidence}")
        elif confidence == "medium":
            st.warning(f"Confidence: {confidence}")
        else:
            st.error(f"Confidence: {confidence}")

        # Cited report
        cited = result.get("cited_report", {})
        if cited.get("cited_report"):
            with st.expander("Cited Report"):
                st.markdown(cited["cited_report"])
                refs = cited.get("references", [])
                if refs:
                    st.markdown("**References:**")
                    for ref in refs:
                        st.markdown(f"- [{ref.get('id', '?')}] {ref.get('title', 'Unknown')} ({ref.get('source', '')})")

        # Judge scores
        judge = result.get("judge_scores", {})
        if judge:
            with st.expander("Quality Scores (LLM-as-Judge)"):
                cols = st.columns(5)
                dimensions = ["factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"]
                for col, dim in zip(cols, dimensions):
                    score = judge.get(dim, 0)
                    col.metric(dim.replace("_", " ").title(), f"{score:.2f}")
                st.metric("Overall", f"{judge.get('overall', 0):.2f}")
                passed = judge.get("pass", False)
                if passed:
                    st.success("PASSED")
                else:
                    st.warning("DID NOT PASS")

        # Store summary
        store_summary = result.get("store_summary", {})
        with st.expander("Execution Summary"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Agents", store_summary.get("total_agents", 0))
            col2.metric("Findings", store_summary.get("total_findings", 0))
            col3.metric("Failed", store_summary.get("failed", 0))
            col4.metric("Time (s)", f"{store_summary.get('elapsed_seconds', 0):.1f}")
            st.markdown(f"**Effort level:** {result.get('effort_level', 'unknown')}")

        # Trace
        trace = result.get("trace", {})
        with st.expander("Agent Trace"):
            st.json(trace)

    except Exception as exc:
        progress.empty()
        st.error(f"Research failed: {exc}")
        import traceback
        st.code(traceback.format_exc())


def main():
    st.set_page_config(page_title="Multi-Agent Research", page_icon="🔬", layout="wide")
    render_multi_agent_page()


if __name__ == "__main__":
    main()
