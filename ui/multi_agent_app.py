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
        --bg: #FAFAFC; --panel: #F3F1FF; --card: #FFFFFF;
        --ink: #14131F; --muted: #6B6B7B; --line: #E9E7F5;
        --accent: #6D5BF6; --accent2: #22C1DC;
        --font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
        --success-bg: #E3FCEF; --success-text: #0B7A46; --success-border: #86EFAC;
        --warn-bg: #FEF3C7; --warn-text: #92400E; --warn-border: #FCD34D;
        --danger-bg: #FEE2E2; --danger-text: #991B1B; --danger-border: #FCA5A5;
    }
    .stApp {
        background: radial-gradient(circle at 15% 0%, #F1EEFF 0%, var(--bg) 45%);
        color: var(--ink); font-family: var(--font);
    }
    .block-container { max-width: 1080px; padding-top: 1.5rem; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FBFAFF 0%, var(--panel) 100%);
        border-right: 1px solid var(--line);
    }
    h2 { font-size: 1.1rem !important; font-weight: 680 !important; }
    h3 { font-size: 0.95rem !important; color: var(--muted); font-weight: 620 !important; }
    .ma-muted { color: var(--muted); font-size: 0.92rem; margin-bottom: 0.7rem; }

    .ma-hero {
        background: linear-gradient(120deg, var(--accent) 0%, #8B7DFA 45%, var(--accent2) 100%);
        border-radius: 20px; padding: 1.6rem 1.8rem; margin-bottom: 1.4rem;
        box-shadow: 0 16px 40px rgba(109,91,246,0.28);
        position: relative; overflow: hidden;
    }
    .ma-hero::after {
        content: ""; position: absolute; inset: 0;
        background: radial-gradient(circle at 90% 10%, rgba(255,255,255,0.25), transparent 55%);
    }
    .ma-hero h1 {
        color: #fff !important; font-family: var(--font) !important;
        font-size: 2rem !important; font-weight: 780 !important; margin: 0 !important;
        letter-spacing: -0.01em; position: relative; z-index: 1;
    }
    .ma-hero p {
        color: rgba(255,255,255,0.92); font-size: 0.96rem; margin: 0.45rem 0 0 0;
        position: relative; z-index: 1; max-width: 640px;
    }
    .ma-pill-strip { display: flex; gap: 0.4rem; margin-top: 0.9rem; flex-wrap: wrap; position: relative; z-index: 1; }
    .ma-pill {
        background: rgba(255,255,255,0.18); color: #fff; font-size: 0.72rem; font-weight: 650;
        padding: 0.22rem 0.65rem; border-radius: 999px; border: 1px solid rgba(255,255,255,0.35);
        backdrop-filter: blur(4px);
    }

    .ma-question {
        background: var(--card); border: 1px solid var(--line); border-radius: 16px;
        padding: 0.9rem 1.15rem; margin: 0.8rem 0;
        box-shadow: 0 10px 28px rgba(20,19,31,0.05);
    }
    .ma-question strong { color: var(--ink); font-weight: 650; }
    .ma-badge-row { display: flex; gap: 0.5rem; align-items: center; margin: 0.5rem 0 0.7rem; flex-wrap: wrap; }
    .ma-badge {
        display: inline-block; font-size: 0.76rem; font-weight: 680;
        padding: 0.25rem 0.65rem; border-radius: 999px; border: 1px solid transparent;
    }
    .ma-badge-safe { background: var(--success-bg); color: var(--success-text); border-color: var(--success-border); }
    .ma-badge-blocked { background: var(--danger-bg); color: var(--danger-text); border-color: var(--danger-border); }
    .ma-badge-precheck { background: #E3E8FF; color: #3730A3; border-color: #C7D2FE; }
    .ma-badge-effort { background: #F3F1FF; color: #5B21B6; border-color: #DDD6FE; }
    .ma-badge-corpus { background: #E3FCEF; color: #0B7A46; border-color: #86EFAC; }
    .ma-badge-arxiv { background: #DBEAFE; color: #1D4ED8; border-color: #93C5FD; }
    .ma-badge-s2 { background: #F3E8FF; color: #6B21A8; border-color: #D8B4FE; }
    .ma-badge-web { background: #FEF3C7; color: #92400E; border-color: #FCD34D; }
    .ma-badge-stat { color: var(--muted); font-size: 0.82rem; font-weight: 500; }
    .ma-precheck {
        border: 1px solid var(--line); border-left: 4px solid var(--accent);
        border-radius: 14px; padding: 0.85rem 1.1rem; margin: 0.6rem 0;
        background: linear-gradient(90deg, #F5F3FF 0%, #FAFAFF 100%);
    }
    .ma-agent-card {
        border: 1px solid var(--line); border-radius: 14px;
        padding: 0.75rem 1rem; margin: 0.5rem 0; background: var(--card);
        box-shadow: 0 4px 14px rgba(20,19,31,0.04);
        transition: box-shadow 0.15s ease;
    }
    .ma-agent-card:hover { box-shadow: 0 8px 22px rgba(109,91,246,0.14); }
    .ma-answer {
        border: 1px solid var(--line); border-left: 4px solid var(--accent);
        border-radius: 16px; padding: 1.1rem 1.25rem; margin: 0.7rem 0;
        background: var(--card); box-shadow: 0 12px 32px rgba(20,19,31,0.06);
    }
    .ma-answer p { font-size: 1.02rem; line-height: 1.7; margin: 0.5rem 0; }
    .ma-kicker {
        color: var(--accent); font-size: 0.76rem; font-weight: 750; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.3rem;
    }
    .score-good { color: #0B7A46; } .score-mid { color: #92400E; } .score-bad { color: #991B1B; }
    div[data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line);
        padding: 0.7rem 0.8rem; border-radius: 14px;
        box-shadow: 0 4px 14px rgba(20,19,31,0.04);
    }
    div[data-testid="stMetric"] label { color: var(--muted) !important; }
    .stButton > button {
        border-radius: 12px; border: 1px solid var(--accent);
        background: linear-gradient(120deg, var(--accent) 0%, #8B7DFA 100%);
        color: #FFF; font-weight: 720; min-height: 2.6rem;
        box-shadow: 0 8px 20px rgba(109,91,246,0.3);
        transition: transform 0.1s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); }
    div[data-testid="stExpander"] details {
        background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    }
    div[data-testid="stTabs"] button[role="tab"] { font-weight: 640; }

    .ma-sys-card {
        background: var(--card); border: 1px solid var(--line); border-radius: 14px;
        padding: 0.8rem 0.9rem; margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(20,19,31,0.04);
    }
    .ma-sys-card p { color: var(--muted); font-size: 0.83rem; line-height: 1.5; margin: 0.3rem 0 0 0; }
    .ma-pattern {
        display: flex; align-items: flex-start; gap: 0.55rem;
        padding: 0.45rem 0; border-bottom: 1px solid var(--line);
    }
    .ma-pattern:last-child { border-bottom: none; }
    .ma-pattern-icon {
        flex: 0 0 auto; width: 1.6rem; height: 1.6rem; border-radius: 8px;
        background: var(--panel); display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem;
    }
    .ma-pattern-body strong { font-size: 0.83rem; color: var(--ink); display: block; }
    .ma-pattern-body span { font-size: 0.76rem; color: var(--muted); }
    .ma-source-row {
        display: flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0;
        font-size: 0.85rem; color: var(--ink);
    }
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

SOURCE_ICON = {
    "local_corpus": "📚",
    "arxiv": "🎓",
    "semantic_scholar": "🔬",
    "web": "🌐",
}


def _score_cls(v: float) -> str:
    return "score-good" if v >= 0.7 else ("score-mid" if v >= 0.4 else "score-bad")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

AGENT_PATTERNS = [
    ("📈", "Effort scaling", "Query complexity sets how many subagents run"),
    ("🗂️", "Corpus pre-check", "Qdrant is checked first before external search"),
    ("🛡️", "Guardrails", "Unsafe or out-of-scope queries are blocked early"),
    ("🤝", "Agent-to-agent awareness", "Subagents see what peers already found"),
    ("🔁", "Error recovery & fallbacks", "A failed source retries via a backup"),
    ("🧪", "Evaluator-optimizer", "Weak syntheses trigger a refinement pass"),
    ("🙋", "Human-in-the-loop", "Preview the plan before agents run"),
    ("⚖️", "LLM-as-judge", "5-dimension score: accuracy, citations, more"),
]

SOURCE_LIST = [
    ("📚", "Local corpus", "Qdrant"),
    ("🎓", "arXiv", "papers"),
    ("🔬", "Semantic Scholar", "papers"),
    ("🌐", "Web", "Tavily"),
]


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """<div class="ma-sys-card">
                <strong>🧠 System Info</strong>
                <p>Orchestrator-worker: a lead agent plans, parallel subagents search,
                and an LLM judge scores quality.</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### ⚙️ Agent Patterns")
        pattern_html = "".join(
            f"""<div class="ma-pattern">
                <div class="ma-pattern-icon">{icon}</div>
                <div class="ma-pattern-body"><strong>{name}</strong><span>{desc}</span></div>
            </div>"""
            for icon, name, desc in AGENT_PATTERNS
        )
        st.markdown(f'<div class="ma-sys-card">{pattern_html}</div>', unsafe_allow_html=True)

        st.markdown("### 🔌 Sources")
        source_html = "".join(
            f'<div class="ma-source-row">{icon} <strong>{name}</strong>'
            f'<span style="color:var(--muted);font-size:0.78rem;margin-left:auto">{tag}</span></div>'
            for icon, name, tag in SOURCE_LIST
        )
        st.markdown(f'<div class="ma-sys-card">{source_html}</div>', unsafe_allow_html=True)


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

        from multi_agent.run_logger import log_run
        log_run(question, result, source="ui")

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
            icon = SOURCE_ICON.get(atype, "🔎")
            n_findings = len(agent.get("findings", []))
            elapsed_a = agent.get("elapsed_seconds", 0)
            status_a = agent.get("status", "?")
            obj = subtasks[i].get("objective", "") if i < len(subtasks) else ""
            queries = agent.get("queries_used", [])
            q_str = ", ".join(f'"{q}"' for q in queries[:3]) if queries else "n/a"
            st.markdown(
                f"""<div class="ma-agent-card">
                    <span class="ma-badge {badge}">{icon} {html.escape(atype)}</span>
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

    st.markdown(
        """
        <div class="ma-hero">
            <h1>🧭 Multi-Agent Research System</h1>
            <p>Ask a research question. The system checks your corpus first, plans the search,
            spawns parallel agents, synthesizes findings, and scores quality — start to finish.</p>
            <div class="ma-pill-strip">
                <span class="ma-pill">📚 Local corpus</span>
                <span class="ma-pill">🎓 arXiv</span>
                <span class="ma-pill">🔬 Semantic Scholar</span>
                <span class="ma-pill">🌐 Web</span>
                <span class="ma-pill">⚖️ LLM-as-judge</span>
            </div>
        </div>
        """,
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
