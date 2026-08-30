"""Multi-Agent Research System — Streamlit UI.

Orchestrator-worker pipeline: guardrail → corpus pre-check → lead agent plans →
parallel subagents → synthesis → citations → LLM-as-judge → evaluator-optimizer.
"""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #FFFFFF; --panel: #F7F7F7; --card: #FFFFFF;
        --ink: #111111; --muted: #5F6368; --line: #E5E5E5;
        --accent: #0B0B0B;
        --font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
        --success-bg: #DCFCE7; --success-text: #166534; --success-border: #86EFAC;
        --warn-bg: #FEF3C7; --warn-text: #92400E; --warn-border: #FCD34D;
        --danger-bg: #FEE2E2; --danger-text: #991B1B; --danger-border: #FCA5A5;
    }
    .stApp { background: var(--bg); color: var(--ink); font-family: var(--font); }
    .block-container { max-width: 1080px; padding-top: 2.5rem; }
    section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
    h1 { font-family: var(--font) !important; font-size: 1.8rem !important; font-weight: 760 !important; margin: 0 0 0.15rem 0 !important; }
    h2 { font-size: 1.1rem !important; font-weight: 680 !important; }
    h3 { font-size: 0.95rem !important; color: var(--muted); font-weight: 620 !important; }
    .ma-muted { color: var(--muted); font-size: 0.88rem; margin-bottom: 0.7rem; }
    .ma-question {
        background: var(--card); border: 1px solid var(--line); border-radius: 14px;
        padding: 0.9rem 1.1rem; margin: 0.8rem 0;
        box-shadow: 0 8px 24px rgba(17,17,17,0.035);
    }
    .ma-question strong { color: var(--ink); font-weight: 650; }
    .ma-badge-row { display: flex; gap: 0.5rem; align-items: center; margin: 0.5rem 0 0.7rem; flex-wrap: wrap; }
    .ma-badge {
        display: inline-block; font-size: 0.76rem; font-weight: 680;
        padding: 0.2rem 0.58rem; border-radius: 999px; border: 1px solid transparent;
    }
    .ma-badge-safe { background: var(--success-bg); color: var(--success-text); border-color: var(--success-border); }
    .ma-badge-blocked { background: var(--danger-bg); color: var(--danger-text); border-color: var(--danger-border); }
    .ma-badge-precheck { background: #DBEAFE; color: #1E40AF; border-color: #93C5FD; }
    .ma-badge-effort { background: #F3F4F6; color: #374151; border-color: #D1D5DB; }
    .ma-badge-corpus { background: #DCFCE7; color: #166534; border-color: #86EFAC; }
    .ma-badge-arxiv { background: #DBEAFE; color: #1D4ED8; border-color: #93C5FD; }
    .ma-badge-s2 { background: #F3E8FF; color: #6B21A8; border-color: #D8B4FE; }
    .ma-badge-web { background: #FEF3C7; color: #92400E; border-color: #FCD34D; }
    .ma-badge-stat { color: var(--muted); font-size: 0.82rem; font-weight: 500; }
    .ma-precheck {
        border: 1px solid var(--line); border-left: 4px solid #1E40AF;
        border-radius: 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; background: #F8FAFF;
    }
    .ma-agent-card {
        border: 1px solid var(--line); border-radius: 10px;
        padding: 0.65rem 0.85rem; margin: 0.45rem 0; background: var(--card);
    }
    .ma-answer {
        border: 1px solid var(--line); border-left: 4px solid var(--ink);
        border-radius: 14px; padding: 1rem 1.1rem; margin: 0.7rem 0;
        background: var(--card); box-shadow: 0 8px 24px rgba(17,17,17,0.035);
    }
    .ma-answer p { font-size: 1rem; line-height: 1.65; margin: 0.5rem 0; }
    .ma-kicker { color: var(--muted); font-size: 0.76rem; font-weight: 720; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.2rem; }
    .score-good { color: #166534; } .score-mid { color: #92400E; } .score-bad { color: #991B1B; }
    div[data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line);
        padding: 0.6rem 0.7rem; border-radius: 10px;
    }
    div[data-testid="stMetric"] label { color: var(--muted) !important; }
    .stButton > button {
        border-radius: 10px; border: 1px solid var(--accent);
        background: var(--accent); color: #FFF; font-weight: 720; min-height: 2.5rem;
    }
    div[data-testid="stExpander"] details { background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SOURCE_BADGE = {
    "local_corpus": "ma-badge-corpus",
    "arxiv": "ma-badge-arxiv",
    "semantic_scholar": "ma-badge-s2",
    "web": "ma-badge-web",
}


def _score_cls(v: float) -> str:
    return "score-good" if v >= 0.7 else ("score-mid" if v >= 0.4 else "score-bad")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("### System Info")
        st.caption(
            "Orchestrator-worker: lead agent plans, "
            "parallel subagents search, LLM-as-judge scores quality."
        )
        st.markdown("---")
        st.markdown("### Agent Patterns")
        st.markdown(
            "1. Effort scaling\n"
            "2. Corpus pre-check (Qdrant-first)\n"
            "3. Guardrails\n"
            "4. Agent-to-agent awareness\n"
            "5. Error recovery & fallbacks\n"
            "6. Evaluator-optimizer\n"
            "7. Human-in-the-loop\n"
            "8. LLM-as-judge (5 dimensions)"
        )


# ---------------------------------------------------------------------------
# Plan preview (Human-in-the-loop)
# ---------------------------------------------------------------------------

def preview_plan(question: str):
    """Show the lead agent's plan without executing — human-in-the-loop."""
    with st.status("Lead agent planning...", expanded=True) as status:
        try:
            from multi_agent.lead import create_plan
            from multi_agent.trace import Tracer
            from multi_agent.config import classify_effort
            from multi_agent.guardrails import check_guardrails
            from openai import OpenAI

            guardrail = check_guardrails(question)
            if not guardrail.safe:
                status.update(label="Blocked by guardrail", state="error")
                st.error(f"Guardrail ({guardrail.category}): {guardrail.reason}")
                return

            tracer = Tracer()
            effort = classify_effort(question)
            client = OpenAI()
            plan = create_plan(question, tracer, client=client, effort=effort)
            status.update(label=f"Plan ready — {effort.name} ({len(plan.get('subtasks', []))} subtasks)", state="complete", expanded=False)

            precheck = plan.get("corpus_precheck", {})
            pcol1, pcol2 = st.columns(2)
            pcol1.metric("Effort", f"{effort.name} (max {effort.max_subagents})")
            pcol2.metric("Pre-check", precheck.get("state", "n/a"))

            st.markdown(f"**Reasoning:** {plan.get('reasoning', 'N/A')}")

            for i, task in enumerate(plan.get("subtasks", []), 1):
                src = task.get("source", "?")
                obj = task.get("objective", "")[:70]
                with st.expander(f"Subtask {i}: {src} — {obj}"):
                    st.json(task)

            st.info("Review the plan above. Click **Approve & Execute** to run, or edit your query.")
            if st.button("Approve & Execute", type="primary", key="approve_exec"):
                run_pipeline(question)

        except Exception as exc:
            status.update(label="Planning failed", state="error")
            st.error(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Full pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(question: str):
    """Execute the full multi-agent research pipeline."""
    started_at = time.perf_counter()
    with st.status("Running multi-agent research...", expanded=True) as status:
        status.write("Checking guardrails...")
        status.write("Pre-checking corpus (Qdrant)...")
        status.write("Lead agent planning subtasks...")
        status.write("Running subagents in parallel...")
        status.write("Synthesizing findings...")
        status.write("Adding citations...")
        status.write("Judge scoring quality...")
        try:
            from multi_agent.orchestrator import run_research
            from openai import OpenAI
            result = run_research(question, openai_client=OpenAI())
        except Exception as exc:
            status.update(label="Pipeline failed", state="error", expanded=True)
            st.error(f"Error: {exc}")
            import traceback
            st.code(traceback.format_exc())
            return
        elapsed = time.perf_counter() - started_at
        status.update(label=f"Complete in {elapsed:.1f}s", state="complete", expanded=False)

    st.session_state["result"] = result
    st.rerun()


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def render_results(result: dict):
    """Render the full multi-agent result with all 8 pattern visibility."""
    query = html.escape(str(result.get("query", "")))
    st.markdown(
        f"<div class='ma-question'><span style='color:var(--muted);font-size:0.82rem'>Query</span>"
        f"<br/><strong>{query}</strong></div>",
        unsafe_allow_html=True,
    )

    guardrail = result.get("guardrail", {})
    plan = result.get("plan", {})
    precheck = plan.get("corpus_precheck", {})
    judge = result.get("judge_scores", {})
    store = result.get("store_summary", {})
    effort = result.get("effort_level", "?")

    # Badge row
    guard_cls = "ma-badge-safe" if guardrail.get("safe", True) else "ma-badge-blocked"
    guard_lbl = guardrail.get("category", "safe")
    pc_state = precheck.get("state", "n/a")
    st.markdown(
        f"""<div class="ma-badge-row">
            <span class="ma-badge {guard_cls}">Guardrail: {html.escape(guard_lbl)}</span>
            <span class="ma-badge ma-badge-precheck">Pre-check: {html.escape(pc_state)}</span>
            <span class="ma-badge ma-badge-effort">Effort: {html.escape(effort)}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Metrics
    cols = st.columns(4)
    cols[0].metric("Agents", store.get("total_agents", 0))
    cols[1].metric("Findings", store.get("total_findings", 0))
    cols[2].metric("Latency", f"{store.get('elapsed_seconds', 0):.1f}s")
    overall = judge.get("overall", 0)
    cols[3].metric("Judge", f"{overall:.2f} {'PASS' if judge.get('pass') else 'FAIL'}")

    # Pre-check card
    if pc_state not in ("n/a", ""):
        matching = precheck.get("matching_papers", [])
        route_map = {
            "full_text_match": "1 subagent (local_corpus only)",
            "abstract_only": "2 subagents (local_corpus + external)",
            "no_match": "External sources only (skip corpus)",
        }
        routing = route_map.get(pc_state, pc_state)
        paper_html = ""
        for p in matching[:3]:
            t = html.escape(str(p.get("title", "")))
            paper_html += f"<br/>&nbsp;&nbsp;{t} (score: {p.get('score', '?')}, {p.get('level', '?')})"
        topics = ", ".join(precheck.get("topics", []))
        topic_html = f"<br/>Topics: {html.escape(topics)}" if topics else ""
        st.markdown(
            f"""<div class="ma-precheck">
                <div class="ma-kicker">Corpus Pre-Check (Qdrant-first routing)</div>
                <strong>{html.escape(pc_state)}</strong>{paper_html}{topic_html}
                <br/>Routing: {html.escape(routing)}
            </div>""",
            unsafe_allow_html=True,
        )

    # Synthesis answer
    synthesis = result.get("synthesis", {})
    answer = synthesis.get("synthesis", "") if isinstance(synthesis, dict) else ""
    if answer:
        paras = [p.strip() for p in str(answer).split("\n") if p.strip()]
        body = "".join(f"<p>{html.escape(p)}</p>" for p in paras)
        st.markdown(
            f"<div class='ma-answer'><div class='ma-kicker'>Research Synthesis</div>{body}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("No synthesis generated.")

    themes = synthesis.get("key_themes", []) if isinstance(synthesis, dict) else []
    if themes:
        st.markdown("**Key themes:** " + " · ".join(themes))

    confidence = synthesis.get("confidence", "unknown") if isinstance(synthesis, dict) else "unknown"
    if confidence == "high":
        st.success(f"Confidence: {confidence}")
    elif confidence == "medium":
        st.warning(f"Confidence: {confidence}")
    else:
        st.info(f"Confidence: {confidence}")

    # --- Tabs ---
    tab_agents, tab_cited, tab_judge, tab_trace = st.tabs(
        ["Agents", "Cited Report", "Judge Scores", "Trace"]
    )

    # Tab: Agents
    with tab_agents:
        agents = store.get("agents", [])
        subtasks = plan.get("subtasks", [])
        if not agents:
            st.info("No agents were executed.")
        for i, agent in enumerate(agents):
            atype = agent.get("agent_type", "?")
            badge = SOURCE_BADGE.get(atype, "ma-badge-precheck")
            n_findings = len(agent.get("findings", []))
            elapsed_a = agent.get("elapsed_seconds", 0)
            status_a = agent.get("status", "?")
            obj = subtasks[i].get("objective", "") if i < len(subtasks) else ""
            queries = agent.get("queries_used", [])
            q_str = ", ".join(f'"{q}"' for q in queries[:3]) if queries else "n/a"
            st.markdown(
                f"""<div class="ma-agent-card">
                    <span class="ma-badge {badge}">{html.escape(atype)}</span>
                    <span class="ma-badge-stat">{n_findings} findings | {elapsed_a:.1f}s | {status_a}</span>
                    <br/><strong>{html.escape(obj[:80])}</strong>
                    <br/><span style="color:var(--muted);font-size:0.82rem">Queries: {html.escape(q_str)}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    # Tab: Cited Report
    with tab_cited:
        cited = result.get("cited_report", {})
        report = cited.get("cited_report", "") if isinstance(cited, dict) else ""
        if report:
            st.markdown(report)
            refs = cited.get("references", [])
            if refs:
                st.markdown("**References:**")
                for ref in refs:
                    st.markdown(f"- [{ref.get('id', '?')}] {ref.get('title', 'Unknown')} ({ref.get('source', '')})")
            uncited = cited.get("uncited_claims", [])
            if uncited:
                st.warning(f"{len(uncited)} uncited claims: " + "; ".join(str(c) for c in uncited[:3]))
        else:
            st.info("No cited report generated.")

    # Tab: Judge Scores
    with tab_judge:
        dims = ["factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"]
        jcols = st.columns(5)
        for col, dim in zip(jcols, dims):
            score = judge.get(dim, 0)
            cls = _score_cls(score)
            col.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:0.76rem;color:var(--muted)'>{dim.replace('_',' ').title()}</div>"
                f"<div class='{cls}' style='font-size:1.35rem;font-weight:760'>{score:.2f}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        ov = judge.get("overall", 0)
        passed = judge.get("pass", False)
        st.markdown(f"**Overall: {ov:.2f}** — {'PASSED' if passed else 'DID NOT PASS'}")
        reasoning = judge.get("reasoning", "")
        if reasoning:
            st.caption(reasoning)

        # Evaluator-optimizer
        trace_data = result.get("trace", {})
        events = trace_data.get("events", []) if isinstance(trace_data, dict) else []
        refine_evts = [e for e in events if isinstance(e, dict) and "refine" in str(e.get("step", "")).lower()]
        if refine_evts:
            st.markdown("---")
            st.markdown("**Evaluator-Optimizer**")
            for ev in refine_evts:
                st.caption(f"{ev.get('step', '')}: {ev.get('detail', ev)}")

    # Tab: Trace
    with tab_trace:
        trace_data = result.get("trace", {})
        if isinstance(trace_data, dict):
            events = trace_data.get("events", [])
            if events and isinstance(events[0], dict):
                rows = []
                for ev in events:
                    rows.append({
                        "Agent": ev.get("agent", ev.get("actor", "")),
                        "Step": ev.get("step", ev.get("event", "")),
                        "Detail": str(ev.get("detail", ""))[:120],
                    })
                if rows:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.json(trace_data, expanded=False)
            else:
                st.json(trace_data, expanded=False)
        else:
            st.info("No trace available.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.session_state.setdefault("result", None)

    render_sidebar()

    st.markdown("<h1>Multi-Agent Research System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ma-muted'>Ask a research question. The system checks your corpus first, "
        "plans the search, spawns parallel agents, synthesizes findings, and scores quality.</div>",
        unsafe_allow_html=True,
    )

    # If we have results, show them with a "New query" button
    if st.session_state.get("result"):
        top_left, top_right = st.columns([5, 1])
        with top_right:
            if st.button("New query", use_container_width=True):
                st.session_state["result"] = None
                st.rerun()
        render_results(st.session_state["result"])
        return

    # Query input
    query = st.text_area(
        "Research query",
        placeholder="e.g., How does LoRA reduce GPU memory during fine-tuning?",
        height=80,
        key="query_input",
    )

    b1, b2, _ = st.columns([1.3, 1.3, 4.4])
    with b1:
        run_btn = st.button("Run Research", type="primary", use_container_width=True)
    with b2:
        plan_btn = st.button("Preview Plan", use_container_width=True)

    if plan_btn and query and query.strip():
        preview_plan(query.strip())

    if run_btn and query and query.strip():
        run_pipeline(query.strip())


if __name__ == "__main__":
    main()
