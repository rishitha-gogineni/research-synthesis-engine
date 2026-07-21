"""Streamlit research analyst workbench for the Research Synthesis Engine."""

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.api_client import (
    SUPPORTED_RESEARCH_TOPICS,
    SUGGESTED_QUESTIONS,
    build_guidance_payload,
    confidence_style,
    error_message,
    is_answerable,
    evidence_rows,
    get_api,
    metric_rows,
    new_request_id,
    open_problem_rows,
    post_api,
    reading_path_rows,
    route_label,
    section_counts,
    source_rows,
    summary_items,
    theme_rows,
    top_supporting_evidence,
    trust_summary,
    weak_evidence_guidance,
)


st.set_page_config(page_title="Research Synthesis Engine", page_icon=None, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
        --rse-bg: #FFFFFF;
        --rse-panel: #F7F7F7;
        --rse-card: #FFFFFF;
        --rse-ink: #111111;
        --rse-muted: #5F6368;
        --rse-soft: #8A8A8A;
        --rse-line: #E5E5E5;
        --rse-accent: #0B0B0B;
        --rse-font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
        --rse-success-bg: #DCFCE7;
        --rse-success-text: #166534;
        --rse-success-border: #86EFAC;
        --rse-warning-bg: #FEF3C7;
        --rse-warning-text: #92400E;
        --rse-warning-border: #FCD34D;
        --rse-danger-bg: #FEE2E2;
        --rse-danger-text: #991B1B;
        --rse-danger-border: #FCA5A5;
        --rse-route-bg: #DBEAFE;
        --rse-route-text: #1D4ED8;
        --rse-route-border: #93C5FD;
    }
    .stApp { background: var(--rse-bg); color: var(--rse-ink); font-family: var(--rse-font); }
    .block-container { max-width: 1080px; padding-top: 3rem; }
    section[data-testid="stSidebar"] { background: var(--rse-panel); border-right: 1px solid var(--rse-line); }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 2rem; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--rse-ink); }
    h1, h2, h3 { letter-spacing: 0 !important; }
    h1 { font-family: var(--rse-font) !important; font-size: 2rem !important; font-weight: 760 !important; color: var(--rse-ink); margin: 0 0 0.25rem 0 !important; line-height: 1.18 !important; }
    h2 { font-size: 1.12rem !important; color: var(--rse-ink); font-weight: 680 !important; }
    h3 { font-size: 1rem !important; color: var(--rse-muted); font-weight: 620 !important; }
    .rse-title-wrap { display: flex; align-items: center; gap: 0.78rem; margin-bottom: 0.15rem; }
    .rse-title-emoji { font-size: 1.75rem; line-height: 1; transform: translateY(-1px); }
    .rse-muted { color: var(--rse-muted); font-size: 0.9rem; margin-bottom: 0.8rem; }
    .rse-page-shell { max-width: 920px; margin: 0 auto; }
    .rse-query-card, .rse-card, .rse-status {
        background: var(--rse-card);
        border: 1px solid var(--rse-line);
        border-radius: 14px;
        box-shadow: 0 10px 28px rgba(17, 17, 17, 0.04);
    }
    .rse-query-card { padding: 1rem 1.1rem; margin: 1.25rem 0 0.9rem; }
    .rse-card { padding: 1rem 1.1rem; margin: 0.85rem 0; }
    .rse-status { padding: 0.72rem 0.85rem; font-size: 0.84rem; line-height: 1.45; margin-top: 0.15rem; }
    .rse-symbol-label { color: var(--rse-muted); font-size: 0.82rem; font-weight: 650; margin: 0.2rem 0 0.35rem; }
    .stSelectbox [data-baseweb="select"] > div { min-height: 3rem; }
    div[data-testid="stMetric"] {
        background: var(--rse-card);
        border: 1px solid var(--rse-line);
        padding: 0.65rem 0.75rem;
        border-radius: 10px;
        box-shadow: none;
    }
    div[data-testid="stMetric"] label { color: var(--rse-muted) !important; }
    div[data-testid="stMetricValue"] { color: var(--rse-ink) !important; font-size: 0.98rem !important; }
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--rse-accent);
        background: var(--rse-accent);
        color: #FFFFFF;
        font-weight: 720;
        min-height: 2.55rem;
    }
    .stButton > button:hover { border-color: var(--rse-accent); color: #FFFFFF; }
    .stSlider [data-baseweb="slider"] div { color: var(--rse-ink) !important; }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: var(--rse-ink) !important;
        border-color: var(--rse-ink) !important;
        box-shadow: 0 0 0 2px #FFFFFF !important;
    }
    .stSlider [data-baseweb="slider"] > div > div > div {
        background-color: var(--rse-ink) !important;
    }
    .stSlider [data-testid="stThumbValue"] { color: var(--rse-ink) !important; }
    .rse-warning {
        border-left: 3px solid var(--rse-warning-text);
        background: var(--rse-warning-bg);
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        margin: 0.45rem 0;
    }
    .rse-source-id { color: var(--rse-ink); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
    .rse-badge-row { display: flex; gap: 0.5rem; align-items: center; margin: 0.55rem 0 0.85rem; flex-wrap: wrap; }
    .rse-badge {
        display: inline-block;
        font-size: 0.76rem;
        font-weight: 680;
        padding: 0.22rem 0.62rem;
        border-radius: 999px;
        border: 1px solid transparent;
        text-transform: capitalize;
    }
    .rse-badge-route { background: var(--rse-route-bg); color: var(--rse-route-text); border-color: var(--rse-route-border); }
    .rse-badge-success { background: var(--rse-success-bg); color: var(--rse-success-text); border-color: var(--rse-success-border); }
    .rse-badge-warning { background: var(--rse-warning-bg); color: var(--rse-warning-text); border-color: var(--rse-warning-border); }
    .rse-badge-danger { background: var(--rse-danger-bg); color: var(--rse-danger-text); border-color: var(--rse-danger-border); }
    .rse-badge-stat { color: var(--rse-muted); font-size: 0.84rem; font-weight: 500; }
    div[data-testid="stExpander"] details { background: var(--rse-card); border: 1px solid var(--rse-line); border-radius: 10px; }
    div[data-testid="stTabs"] button { font-weight: 650; color: var(--rse-muted); }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--rse-ink); }
    .rse-kicker { color: var(--rse-muted); font-size: 0.76rem; font-weight: 720; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem; }
    .rse-answer-card, .rse-trust-card { border: 1px solid var(--rse-line); border-radius: 14px; background: var(--rse-card); padding: 1rem 1.1rem; margin: 0.8rem 0; box-shadow: 0 10px 28px rgba(17, 17, 17, 0.035); }
    .rse-answer-card { border-left: 4px solid var(--rse-ink); }
    .rse-answer-card p { font-size: 1rem; line-height: 1.68; color: var(--rse-ink); margin: 0.65rem 0; }
    .rse-trust-card { background: #FAFAFA; }
    .rse-trust-title { font-weight: 760; margin-bottom: 0.35rem; }
    .rse-trust-meta { color: var(--rse-muted); font-size: 0.86rem; line-height: 1.5; }
    .rse-question-card { border-bottom: 1px solid var(--rse-line); padding: 0.45rem 0 0.8rem; margin-bottom: 0.85rem; color: var(--rse-muted); }
    .rse-question-card strong { color: var(--rse-ink); font-weight: 650; }
    .rse-empty-state { border: 1px solid var(--rse-line); border-left: 4px solid var(--rse-warning-text); border-radius: 14px; padding: 1rem 1.1rem; background: #FFFBEB; margin: 0.8rem 0; }
    .rse-empty-state h3 { color: var(--rse-ink); margin-top: 0; }
    .rse-source-card-title { font-weight: 720; margin-bottom: 0.25rem; }
    .rse-source-card-meta { color: var(--rse-muted); font-size: 0.82rem; margin-bottom: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def dataframe(rows: list[dict], *, hide_index: bool = True):
    if not rows:
        st.info("No rows returned.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=hide_index)


def render_health():
    try:
        health, _ = get_api("/health", timeout=8)
    except requests.RequestException:
        return {"status": "offline", "dependencies": {}}
    return health


def render_corpus_stats():
    try:
        stats, _ = get_api("/corpus/stats", timeout=8)
    except requests.RequestException:
        return {}
    return stats if not error_message(stats) else {}


def render_title():
    st.markdown(
        "<div class='rse-title-wrap'><span class='rse-title-emoji'>📚</span><h1>Research Synthesis Engine</h1></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='rse-muted'>Ask a question, inspect the evidence, trace the sources.</div>", unsafe_allow_html=True)


def render_status(health: dict, stats: dict):
    status = health.get("status", "unknown")
    papers = stats.get("paper_count") or stats.get("enriched_papers") or "-"
    chunks = stats.get("chunk_count") or stats.get("full_text_chunks") or "-"
    topics = stats.get("topics") or "-"
    st.markdown(
        f"<div class='rse-status'><strong>API:</strong> {status}<br/><strong>Corpus:</strong> {papers} papers | {chunks} chunks | {topics} topics</div>",
        unsafe_allow_html=True,
    )


def render_filter_sidebar(health: dict, stats: dict):
    with st.sidebar:
        st.markdown("### Controls")
        render_status(health, stats)
        st.markdown("### Filters")
        st.multiselect("Research area", SUPPORTED_RESEARCH_TOPICS, key="research_areas")
        st.slider("Publication years", min_value=2017, max_value=2026, key="year_range", step=1)
        st.slider("Top K", min_value=3, max_value=20, key="top_k", step=1)
        st.checkbox("Full text only", key="full_text_only")
        st.checkbox("Diagnostics", key="include_debug")


def render_badge_row(payload: dict):
    retrieval = payload.get("retrieval") or {}
    route = (retrieval.get("route") or {}).get("route")
    confidence_decision = (payload.get("confidence") or {}).get("decision")
    kind, label = confidence_style(confidence_decision)
    paper_count = retrieval.get("paper_result_count", 0)
    chunk_count = retrieval.get("chunk_result_count", 0)
    st.markdown(
        f"""<div class="rse-badge-row">
            <span class="rse-badge rse-badge-route">{route_label(route)}</span>
            <span class="rse-badge rse-badge-{kind}">{label}</span>
            <span class="rse-badge-stat">{paper_count} papers · {chunk_count} chunks</span>
        </div>""",
        unsafe_allow_html=True,
    )


def render_summary(payload: dict, request_id: str | None):
    render_badge_row(payload)
    items = {k: v for k, v in summary_items(payload, request_id).items() if k not in ("Route", "Confidence")}
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items.items()):
        col.metric(label, value)
    warnings = payload.get("warnings") or payload.get("retrieval", {}).get("warnings") or []
    for warning in warnings:
        st.markdown(f"<div class='rse-warning'>{html.escape(str(warning))}</div>", unsafe_allow_html=True)


def render_question_context(payload: dict):
    question = html.escape(str(payload.get("question") or st.session_state.get("question", "")))
    st.markdown(
        f"<div class='rse-question-card'><span>Question</span><br/><strong>{question}</strong></div>",
        unsafe_allow_html=True,
    )


def render_evidence_gate(payload: dict):
    summary = trust_summary(payload)
    passed = is_answerable(payload)
    title = "Evidence gate passed" if passed else "Evidence gate did not pass"
    route = route_label(summary.get("route"))
    reason = html.escape(str(summary.get("reason") or "No routing reason returned."))
    label = html.escape(str(summary.get("label") or "unknown"))
    counts = f"{summary.get('paper_count', 0)} papers · {summary.get('chunk_count', 0)} chunks"
    st.markdown(
        f"""
        <div class='rse-trust-card'>
            <div class='rse-kicker'>Evidence check</div>
            <div class='rse-trust-title'>{title}</div>
            <div class='rse-trust-meta'>Confidence: <strong>{label}</strong> · Route: <strong>{route}</strong> · Retrieved: <strong>{counts}</strong><br/>{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_weak_evidence_state(payload: dict):
    items = "".join(f"<li>{html.escape(item)}</li>" for item in weak_evidence_guidance(payload))
    st.markdown(
        f"""
        <div class='rse-empty-state'>
            <h3>No grounded answer shown</h3>
            <p>The system found evidence, but it did not meet the confidence threshold for a direct answer.</p>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_card(payload: dict):
    brief = payload.get("brief") or {}
    direct_answer = brief.get("direct_answer")
    if not direct_answer:
        st.info("No direct answer was returned for this query.")
        return
    paragraphs = [part.strip() for part in direct_answer.split("\n") if part.strip()]
    body = "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)
    st.markdown(
        f"""
        <div class='rse-answer-card'>
            <div class='rse-kicker'>⊙ Direct Answer</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_brief(payload: dict):
    if not is_answerable(payload):
        render_weak_evidence_state(payload)
        return
    render_answer_card(payload)
    themes = theme_rows(payload)
    if themes:
        st.subheader("Research Themes")
        dataframe(themes)
    brief = payload.get("brief") or {}
    bullets = brief.get("evidence_bullets") or []
    if bullets:
        st.subheader("Evidence Highlights")
        for item in bullets:
            st.write(f"- {item}")
    limitations = brief.get("limitations") or []
    if limitations:
        st.subheader("What The Evidence Does Not Establish")
        for item in limitations:
            st.write(f"- {item}")


def render_top_evidence(payload: dict):
    rows = top_supporting_evidence(payload)
    if not rows:
        st.info("No supporting evidence returned.")
        return
    for row in rows:
        title = html.escape(str(row.get("Title") or "Untitled source"))
        detail = html.escape(f"{row.get('Source', 'source')} | {row.get('Year') or 'unknown year'} | {row.get('Citations') or 0} citations")
        source_id = html.escape(str(row.get("Source ID") or "source"))
        why = row.get("Why It Matters") or "No evidence summary returned."
        with st.container(border=True):
            st.markdown(f"<div class='rse-source-card-title'>{title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='rse-source-card-meta'>{detail}</div>", unsafe_allow_html=True)
            st.write(why)
            st.markdown(f"<span class='rse-source-id'>{source_id}</span>", unsafe_allow_html=True)


def render_evidence(payload: dict):
    rows = evidence_rows(payload)
    if not rows:
        st.info("No structured claims were returned for this query.")
        return
    dataframe(rows)


def render_reading_path(payload: dict):
    path = payload.get("reading_path") or {}
    stages = path.get("stages", []) or []
    if not stages:
        st.info("No staged reading path was returned for this query.")
    for stage in stages:
        with st.container(border=True):
            st.subheader(stage.get("stage") or "Reading stage")
            for paper in stage.get("papers", []) or []:
                order = paper.get("order") or "-"
                title = paper.get("title") or "Untitled paper"
                year = paper.get("publication_year") or "unknown year"
                citations = paper.get("citation_count") or 0
                reason = paper.get("reason_to_read") or "No reason returned."
                st.markdown(f"**{order}. {title}**")
                st.caption(f"{year} · {citations} citations")
                st.write(reason)
    for limitation in path.get("limitations", []) or []:
        st.markdown(f"<div class='rse-warning'>{html.escape(str(limitation))}</div>", unsafe_allow_html=True)


def render_open_problems(payload: dict):
    report = payload.get("open_problems") or {}
    problems = report.get("problems", []) or []
    if not problems:
        st.info("No open problems were returned for this query.")
    for problem in problems:
        with st.container(border=True):
            st.markdown(f"**{problem.get('title') or 'Open problem'}**")
            st.caption(f"{problem.get('category') or 'uncategorized'} · {problem.get('evidence_strength') or 'unknown'} evidence")
            st.write(problem.get("why_it_matters") or "No explanation returned.")
            sources = ", ".join(problem.get("supporting_source_ids", []) or [])
            if sources:
                st.markdown(f"<span class='rse-source-id'>{html.escape(sources)}</span>", unsafe_allow_html=True)
    if report.get("recurring_limitations"):
        st.subheader("Recurring Limitations")
        for item in report["recurring_limitations"]:
            st.write(f"- {item}")
    if report.get("evidence_gaps"):
        st.subheader("Evidence Gaps")
        for item in report["evidence_gaps"]:
            st.write(f"- {item}")


def render_sources(payload: dict):
    paper_rows, chunk_rows = source_rows(payload)
    paper_tab, chunk_tab = st.tabs(["Papers", "Chunks"])
    with paper_tab:
        dataframe(paper_rows)
        for paper in (payload.get("retrieval") or {}).get("paper_results", []) or []:
            with st.expander(paper.get("title", "Paper")):
                st.write(paper.get("abstract") or paper.get("main_contribution") or "No abstract returned.")
                st.markdown(f"<span class='rse-source-id'>paper:{html.escape(str(paper.get('paper_id')))}</span>", unsafe_allow_html=True)
    with chunk_tab:
        dataframe(chunk_rows)
        for chunk in (payload.get("retrieval") or {}).get("chunk_results", []) or []:
            label = chunk.get("section_hint") or chunk.get("title") or "Chunk"
            with st.expander(label):
                st.write(chunk.get("text", ""))
                st.markdown(f"<span class='rse-source-id'>chunk:{html.escape(str(chunk.get('chunk_id')))}</span>", unsafe_allow_html=True)


def render_diagnostics(payload: dict):
    retrieval = payload.get("retrieval") or {}
    route = retrieval.get("route") or {}
    confidence = payload.get("confidence") or {}
    st.subheader("Route")
    st.json(route, expanded=False)
    st.subheader("Confidence")
    st.json(confidence, expanded=False)
    st.subheader("Timing")
    dataframe(metric_rows(payload))
    debug = payload.get("debug") or retrieval.get("debug")
    if debug:
        st.subheader("Debug")
        st.json(debug, expanded=False)


def run_guidance(payload: dict, request_id: str) -> tuple[dict, str | None]:
    return post_api("/guidance", payload, request_id=request_id, timeout=180)


def preview_route(payload: dict) -> dict:
    try:
        route, _ = post_api("/route", payload, timeout=30)
        return route
    except requests.RequestException as exc:
        return {"error": {"code": "CONNECTION_ERROR", "message": str(exc)}}


def initialize_state():
    st.session_state.setdefault("view", "query")
    st.session_state.setdefault("question", SUGGESTED_QUESTIONS[0])
    st.session_state.setdefault("suggested_question", SUGGESTED_QUESTIONS[0])
    st.session_state.setdefault("research_areas", [])
    st.session_state.setdefault("year_range", (2017, 2026))
    st.session_state.setdefault("top_k", 8)
    st.session_state.setdefault("full_text_only", False)
    st.session_state.setdefault("include_debug", False)


def build_payload_from_state() -> dict:
    year_min, year_max = st.session_state.get("year_range", (2017, 2026))
    return build_guidance_payload(
        question=st.session_state.get("question", ""),
        top_k=st.session_state.get("top_k", 8),
        research_areas=st.session_state.get("research_areas") or [],
        publication_year_min=year_min,
        publication_year_max=year_max,
        full_text_only=st.session_state.get("full_text_only", False),
        include_debug=st.session_state.get("include_debug", False),
    )


def sync_question_from_suggestion():
    st.session_state["question"] = st.session_state.get("suggested_question", SUGGESTED_QUESTIONS[0])


def render_query_page(health: dict, stats: dict):
    render_title()

    st.selectbox("Suggested question", SUGGESTED_QUESTIONS, key="suggested_question", on_change=sync_question_from_suggestion)
    st.markdown("<div class='rse-symbol-label'>⌕ Question</div>", unsafe_allow_html=True)
    st.text_area(
        "Question",
        key="question",
        height=88,
        label_visibility="collapsed",
        placeholder="Ask a question about the corpus...",
    )
    b1, b2, _ = st.columns([1.25, 1.25, 4])
    with b1:
        run_button = st.button("Run analysis", type="primary", use_container_width=True)
    with b2:
        preview_button = st.button("Preview route", use_container_width=True)
    if preview_button:
        route = preview_route({"question": st.session_state.get("question", "")})
        msg = error_message(route)
        if msg:
            st.error(msg)
        else:
            st.markdown("<div class='rse-symbol-label'>? Route preview</div>", unsafe_allow_html=True)
            st.json(route, expanded=False)

    if run_button:
        if not st.session_state.get("question", "").strip():
            st.error("VALIDATION_ERROR: question is required.")
            return
        request_id = new_request_id()
        payload = build_payload_from_state()
        started_at = time.perf_counter()
        with st.status("Preparing analysis", expanded=True) as status:
            status.write("Routing the question to choose the retrieval path.")
            route_preview = preview_route({"question": payload["question"]})
            route_error = error_message(route_preview)
            if route_error:
                status.write("Route preview was unavailable; continuing with full guidance.")
            else:
                route = route_preview.get("selected_route") or "unknown"
                status.write(f"Selected route: {route_label(route)}.")
            status.write("Running retrieval, confidence check, and synthesis from the indexed corpus.")
            try:
                result, response_id = run_guidance(payload, request_id)
            except requests.RequestException as exc:
                status.update(label="Analysis failed", state="error", expanded=True)
                st.error(f"CONNECTION_ERROR: {exc}")
                return
            elapsed = time.perf_counter() - started_at
            status.update(label=f"Analysis complete in {elapsed:.1f}s", state="complete", expanded=False)
        msg = error_message(result)
        if msg:
            st.error(msg)
            return
        st.session_state["guidance_result"] = result
        st.session_state["request_id"] = response_id or request_id
        st.session_state["show_diagnostics"] = st.session_state.get("include_debug", False)
        st.session_state["view"] = "results"
        st.rerun()


def render_results_page(health: dict, stats: dict):
    result = st.session_state.get("guidance_result")
    if not result:
        st.session_state["view"] = "query"
        st.rerun()

    top_left, top_right = st.columns([4, 1])
    with top_left:
        render_title()
    with top_right:
        if st.button("Back to query", use_container_width=True):
            st.session_state["view"] = "query"
            st.rerun()
    render_question_context(result)
    render_evidence_gate(result)
    render_summary(result, st.session_state.get("request_id"))
    render_brief(result)

    st.subheader("Top Supporting Evidence")
    render_top_evidence(result)

    counts = section_counts(result)
    tab_specs = [
        (f"▦ Evidence Matrix · {counts.get('Evidence', '0 claims')}", render_evidence),
        (f"→ Reading Path · {counts.get('Reading Path', '0 stages')}", render_reading_path),
        (f"◇ Open Problems · {counts.get('Open Problems', '0 found')}", render_open_problems),
        (f"§ Sources · {counts.get('Sources', '0 papers / 0 chunks')}", render_sources),
    ]
    if st.session_state.get("show_diagnostics"):
        tab_specs.append(("⌁ Diagnostics", render_diagnostics))
    tabs = st.tabs([label for label, _ in tab_specs])
    for tab, (_, renderer) in zip(tabs, tab_specs):
        with tab:
            renderer(result)


def main():
    initialize_state()
    health = render_health()
    stats = render_corpus_stats()
    render_filter_sidebar(health, stats)
    if st.session_state.get("view") == "results":
        render_results_page(health, stats)
    else:
        render_query_page(health, stats)


if __name__ == "__main__":
    main()
