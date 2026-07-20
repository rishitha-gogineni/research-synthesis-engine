"""Streamlit research analyst workbench for the Research Synthesis Engine."""

from __future__ import annotations

import sys
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
    error_message,
    evidence_rows,
    get_api,
    metric_rows,
    new_request_id,
    open_problem_rows,
    ordered_sections,
    post_api,
    reading_path_map_dot,
    reading_path_rows,
    source_rows,
    summary_items,
    theme_rows,
    top_supporting_evidence,
)


st.set_page_config(page_title="Research Synthesis Engine", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    :root {
        --rse-bg: #F4EFE6;
        --rse-paper: #FFFCF7;
        --rse-ink: #111111;
        --rse-muted: #5F5A52;
        --rse-line: #D8CFC0;
        --rse-bronze: #8A5A2B;
        --rse-green: #2F6F4E;
        --rse-amber: #A16207;
        --rse-red: #9F1D1D;
    }
    .stApp {
        background: var(--rse-bg);
        color: var(--rse-ink);
        font-family: Georgia, Cambria, "Times New Roman", serif;
    }
    .block-container { max-width: 1320px; padding-top: 2.1rem; }
    section[data-testid="stSidebar"] { display: none; }
    h1, h2, h3, h4, p, label, span, div { letter-spacing: 0 !important; }
    h1, h2, h3 { font-family: Georgia, Cambria, "Times New Roman", serif !important; color: var(--rse-ink); }
    h1 { font-size: 2.05rem !important; line-height: 1.12 !important; margin-bottom: 0.15rem !important; }
    h2 { font-size: 1.3rem !important; margin-top: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--rse-paper);
        border: 1px solid var(--rse-line);
        border-radius: 4px;
        box-shadow: 0 18px 45px rgba(17, 17, 17, 0.06);
    }
    div[data-testid="stMetric"] {
        background: var(--rse-paper);
        border: 1px solid var(--rse-line);
        padding: 0.75rem 0.9rem;
        border-radius: 4px;
        box-shadow: none;
    }
    div[data-testid="stMetric"] label { color: var(--rse-muted) !important; font-family: Georgia, Cambria, "Times New Roman", serif !important; }
    div[data-testid="stMetricValue"] { color: var(--rse-ink) !important; font-size: 1.02rem !important; }
    .stButton > button {
        border-radius: 3px;
        border: 1px solid var(--rse-ink);
        background: var(--rse-ink);
        color: var(--rse-paper);
        font-family: Georgia, Cambria, "Times New Roman", serif;
        font-weight: 700;
    }
    .stButton > button[kind="secondary"] { background: var(--rse-paper); color: var(--rse-ink); }
    .rse-kicker { color: var(--rse-bronze); text-transform: uppercase; font-size: 0.74rem; letter-spacing: 0.08em !important; font-weight: 700; }
    .rse-status {
        border: 1px solid var(--rse-line);
        background: var(--rse-paper);
        border-radius: 4px;
        padding: 0.75rem 0.9rem;
        font-size: 0.92rem;
    }
    .rse-console-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.2rem; }
    .rse-muted { color: var(--rse-muted); font-size: 0.9rem; }
    .rse-warning {
        border-left: 3px solid var(--rse-amber);
        background: #FBF3E4;
        padding: 0.65rem 0.8rem;
        border-radius: 3px;
        margin: 0.4rem 0;
    }
    .rse-evidence-good { border-left: 4px solid var(--rse-green); }
    .rse-evidence-low { border-left: 4px solid var(--rse-red); }
    .rse-source-id { color: var(--rse-bronze); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
    .rse-section-label { color: var(--rse-bronze); font-size: 0.78rem; text-transform: uppercase; font-weight: 700; margin-top: 1.2rem; }
    div[data-testid="stExpander"] details {
        background: var(--rse-paper);
        border: 1px solid var(--rse-line);
        border-radius: 4px;
    }
    div[data-testid="stDataFrame"] { border: 1px solid var(--rse-line); }
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


def render_header(health: dict, stats: dict):
    left, right = st.columns([3, 2])
    with left:
        st.markdown("<div class='rse-kicker'>Evidence Workbench</div>", unsafe_allow_html=True)
        st.title("Research Synthesis Engine")
        st.markdown("<div class='rse-muted'>Grounded research briefs from an indexed AI literature corpus.</div>", unsafe_allow_html=True)
    with right:
        status = health.get("status", "unknown")
        papers = stats.get("paper_count") or stats.get("enriched_papers") or "-"
        chunks = stats.get("chunk_count") or stats.get("full_text_chunks") or "-"
        topics = stats.get("topics") or "-"
        st.markdown(
            f"<div class='rse-status'><strong>API</strong>: {status}<br/><strong>Corpus</strong>: {papers} papers | {chunks} chunks | {topics} topics</div>",
            unsafe_allow_html=True,
        )


def render_summary(payload: dict, request_id: str | None):
    items = summary_items(payload, request_id)
    cols = st.columns(5)
    for col, (label, value) in zip(cols, items.items()):
        col.metric(label, value)
    warnings = payload.get("warnings") or payload.get("retrieval", {}).get("warnings") or []
    for warning in warnings:
        st.markdown(f"<div class='rse-warning'>{warning}</div>", unsafe_allow_html=True)


def render_evidence_status(payload: dict):
    confidence = payload.get("confidence") or {}
    retrieval = payload.get("retrieval") or {}
    decision = confidence.get("decision", "unknown")
    score = confidence.get("confidence_score", "-")
    route = (retrieval.get("route") or {}).get("route", "unknown")
    css_class = "rse-evidence-good" if decision == "sufficient_evidence" else "rse-evidence-low"
    st.markdown(
        f"<div class='rse-status {css_class}'><strong>Evidence:</strong> {decision} &nbsp; | &nbsp; "
        f"<strong>Score:</strong> {score} &nbsp; | &nbsp; <strong>Route:</strong> {route} &nbsp; | &nbsp; "
        f"<strong>Sources:</strong> {retrieval.get('paper_result_count', 0)} papers / {retrieval.get('chunk_result_count', 0)} chunks</div>",
        unsafe_allow_html=True,
    )


def render_section_heading(title: str):
    st.markdown(f"<div class='rse-section-label'>{title}</div>", unsafe_allow_html=True)


def render_brief(payload: dict):
    brief = payload.get("brief") or {}
    direct_answer = brief.get("direct_answer")
    if direct_answer:
        st.subheader("Direct Answer")
        for paragraph in [part.strip() for part in direct_answer.split("\n") if part.strip()]:
            st.write(paragraph)
    themes = theme_rows(payload)
    if themes:
        st.subheader("Research Themes")
        dataframe(themes)
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
    if not any([direct_answer, themes, bullets, limitations]):
        st.info("No brief returned.")


def render_top_evidence(payload: dict):
    rows = top_supporting_evidence(payload)
    if not rows:
        st.info("No supporting evidence returned.")
        return
    for row in rows:
        title = row.get("Title") or "Untitled source"
        detail = f"{row.get('Source', 'source')} | {row.get('Year') or 'unknown year'} | {row.get('Citations') or 0} citations"
        with st.expander(f"{title} — {detail}", expanded=False):
            st.write(row.get("Why It Matters") or "No evidence summary returned.")
            st.markdown(f"<span class='rse-source-id'>{row.get('Source ID')}</span>", unsafe_allow_html=True)


def render_evidence(payload: dict):
    dataframe(evidence_rows(payload))


def render_reading_path(payload: dict):
    dot = reading_path_map_dot(payload)
    if dot:
        st.graphviz_chart(dot, use_container_width=True)
    rows = reading_path_rows(payload)
    dataframe(rows)
    path = payload.get("reading_path") or {}
    for limitation in path.get("limitations", []) or []:
        st.markdown(f"<div class='rse-warning'>{limitation}</div>", unsafe_allow_html=True)


def render_open_problems(payload: dict):
    dataframe(open_problem_rows(payload))
    report = payload.get("open_problems") or {}
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
                st.markdown(f"<span class='rse-source-id'>paper:{paper.get('paper_id')}</span>", unsafe_allow_html=True)
    with chunk_tab:
        dataframe(chunk_rows)
        for chunk in (payload.get("retrieval") or {}).get("chunk_results", []) or []:
            label = chunk.get("section_hint") or chunk.get("title") or "Chunk"
            with st.expander(label):
                st.write(chunk.get("text", ""))
                st.markdown(f"<span class='rse-source-id'>chunk:{chunk.get('chunk_id')}</span>", unsafe_allow_html=True)


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


def main():
    health = render_health()
    stats = render_corpus_stats()
    render_header(health, stats)

    with st.form("query_console"):
        st.markdown("<div class='rse-console-title'>Ask a research question</div>", unsafe_allow_html=True)
        preset = st.selectbox("Suggested question", SUGGESTED_QUESTIONS, index=0)
        question = st.text_area("Question", value=preset, height=95)
        c1, c2, c3, c4 = st.columns([2.2, 1.8, 1.2, 1.2])
        with c1:
            research_areas = st.multiselect("Research areas", SUPPORTED_RESEARCH_TOPICS)
        with c2:
            year_min, year_max = st.slider("Publication years", min_value=2017, max_value=2026, value=(2017, 2026), step=1)
        with c3:
            top_k = st.slider("Top K", min_value=3, max_value=20, value=8, step=1)
        with c4:
            full_text_only = st.checkbox("Full text only", value=False)
            include_debug = st.checkbox("Diagnostics", value=False)
        b1, b2, _ = st.columns([1.2, 1.2, 5])
        with b1:
            run_button = st.form_submit_button("Run analysis", type="primary", use_container_width=True)
        with b2:
            preview_button = st.form_submit_button("Preview route", use_container_width=True)

    payload = build_guidance_payload(
        question=question,
        top_k=top_k,
        research_areas=research_areas,
        publication_year_min=year_min,
        publication_year_max=year_max,
        full_text_only=full_text_only,
        include_debug=include_debug,
    )

    if preview_button:
        route = preview_route({"question": question})
        msg = error_message(route)
        if msg:
            st.error(msg)
        else:
            st.markdown("### Route Preview")
            st.json(route, expanded=False)

    if run_button:
        if not question.strip():
            st.error("VALIDATION_ERROR: question is required.")
            return
        request_id = new_request_id()
        with st.spinner("Running analysis"):
            try:
                result, response_id = run_guidance(payload, request_id)
            except requests.RequestException as exc:
                st.error(f"CONNECTION_ERROR: {exc}")
                return
        msg = error_message(result)
        if msg:
            st.error(msg)
            return
        st.session_state["guidance_result"] = result
        st.session_state["request_id"] = response_id or request_id
        st.session_state["show_diagnostics"] = include_debug

    result = st.session_state.get("guidance_result")
    if not result:
        st.markdown("<div class='rse-muted'>No analysis run in this session.</div>", unsafe_allow_html=True)
        return

    render_evidence_status(result)
    render_summary(result, st.session_state.get("request_id"))
    section_renderers = {
        "Brief": render_brief,
        "Top Evidence": render_top_evidence,
        "Evidence": render_evidence,
        "Reading Path": render_reading_path,
        "Open Problems": render_open_problems,
        "Sources": render_sources,
        "Diagnostics": render_diagnostics,
    }
    sections = ordered_sections(result.get("question") or payload.get("question") or "")
    if not st.session_state.get("show_diagnostics"):
        sections = [section for section in sections if section != "Diagnostics"]
    lead_sections = sections[:2]
    remaining_sections = sections[2:]
    for section in lead_sections:
        render_section_heading(section)
        section_renderers[section](result)
    if remaining_sections:
        tabs = st.tabs(remaining_sections)
        for tab, section in zip(tabs, remaining_sections):
            with tab:
                section_renderers[section](result)


if __name__ == "__main__":
    main()
