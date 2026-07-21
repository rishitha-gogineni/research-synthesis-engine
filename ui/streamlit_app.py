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
    confidence_style,
    error_message,
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
)


st.set_page_config(page_title="Research Synthesis Engine", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    :root {
        --rse-bg: #F7F5F0;
        --rse-panel: #F2EFE8;
        --rse-card: #FFFFFF;
        --rse-ink: #111111;
        --rse-muted: #737067;
        --rse-soft: #9A958B;
        --rse-line: #DED8CC;
        --rse-accent: #0B0B0B;
        --rse-bronze: #8A5A2B;
        --rse-font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
        --rse-title-font: Georgia, Cambria, "Times New Roman", serif;

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
    .block-container { max-width: 1180px; padding-top: 1.6rem; }
    h1, h2, h3 { letter-spacing: 0 !important; }
    h1 { font-family: var(--rse-title-font) !important; font-size: 1.55rem !important; font-weight: 700 !important; color: var(--rse-ink); margin-bottom: 0.1rem !important; }
    h2 { font-size: 1.08rem !important; color: var(--rse-ink); font-weight: 650 !important; }
    h3 { font-size: 0.98rem !important; color: var(--rse-muted); font-weight: 600 !important; }
    section[data-testid="stSidebar"] { background: var(--rse-panel); border-right: 1px solid var(--rse-line); }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-family: var(--rse-font) !important; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--rse-muted); }
    .rse-sidebar-title { color: var(--rse-muted); font-size: 0.9rem; font-weight: 650; margin-bottom: 0.4rem; }
    .rse-sidebar-note { color: var(--rse-muted); font-size: 0.78rem; line-height: 1.45; margin-top: 0.65rem; }
    div[data-testid="stMetric"] {
        background: var(--rse-card);
        border: 1px solid var(--rse-line);
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
        box-shadow: none;
    }
    div[data-testid="stMetric"] label { color: var(--rse-muted) !important; }
    div[data-testid="stMetricValue"] { color: var(--rse-ink) !important; font-size: 0.98rem !important; }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--rse-accent);
        background: var(--rse-accent);
        color: #FFFFFF;
        font-weight: 700;
    }
    .stButton > button:hover { border-color: var(--rse-accent); color: #FFFFFF; }
    .rse-status {
        border: 1px solid var(--rse-line);
        background: var(--rse-card);
        border-radius: 10px;
        padding: 0.65rem 0.75rem;
        font-size: 0.88rem;
    }
    .rse-muted { color: var(--rse-muted); font-size: 0.86rem; }
    .rse-card {
        background: var(--rse-card);
        border: 1px solid var(--rse-line);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.7rem 0;
    }
    .rse-warning {
        border-left: 3px solid var(--rse-warning-text);
        background: var(--rse-warning-bg);
        padding: 0.6rem 0.75rem;
        border-radius: 8px;
        margin: 0.35rem 0;
    }
    .rse-source-id { color: var(--rse-bronze); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
    .rse-badge-row { display: flex; gap: 0.5rem; align-items: center; margin: 0.45rem 0 0.8rem; flex-wrap: wrap; }
    .rse-badge {
        display: inline-block;
        font-size: 0.76rem;
        font-weight: 650;
        padding: 0.22rem 0.62rem;
        border-radius: 999px;
        border: 1px solid transparent;
        text-transform: capitalize;
    }
    .rse-badge-route { background: var(--rse-route-bg); color: var(--rse-route-text); border-color: var(--rse-route-border); }
    .rse-badge-success { background: var(--rse-success-bg); color: var(--rse-success-text); border-color: var(--rse-success-border); }
    .rse-badge-warning { background: var(--rse-warning-bg); color: var(--rse-warning-text); border-color: var(--rse-warning-border); }
    .rse-badge-danger { background: var(--rse-danger-bg); color: var(--rse-danger-text); border-color: var(--rse-danger-border); }
    .rse-badge-stat { color: var(--rse-muted); font-size: 0.82rem; font-weight: 500; }
    div[data-testid="stExpander"] details { background: var(--rse-card); border: 1px solid var(--rse-line); border-radius: 10px; }
    div[data-testid="stTabs"] button { font-weight: 650; color: var(--rse-muted); }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--rse-ink); }
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
        st.title("Research Synthesis Engine")
        st.markdown("<div class='rse-muted'>Ask a question, inspect the evidence, and trace the sources behind the answer.</div>", unsafe_allow_html=True)
    with right:
        status = health.get("status", "unknown")
        papers = stats.get("paper_count") or stats.get("enriched_papers") or "-"
        chunks = stats.get("chunk_count") or stats.get("full_text_chunks") or "-"
        topics = stats.get("topics") or "-"
        st.markdown(
            f"<div class='rse-status'><strong>API:</strong> {status}<br/><strong>Corpus:</strong> {papers} papers | {chunks} chunks | {topics} topics</div>",
            unsafe_allow_html=True,
        )


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
        st.markdown(f"<div class='rse-warning'>{warning}</div>", unsafe_allow_html=True)


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

    with st.sidebar:
        st.markdown("<div class='rse-sidebar-title'>Query Panel</div>", unsafe_allow_html=True)
        research_areas = st.multiselect("Research area", SUPPORTED_RESEARCH_TOPICS)
        preset = st.selectbox("Suggested question", SUGGESTED_QUESTIONS, index=0)
        question = st.text_area("Question", value=preset, height=115)
        top_k = st.slider("Top K", min_value=3, max_value=20, value=8, step=1)
        year_min, year_max = st.slider("Publication years", min_value=2017, max_value=2026, value=(2017, 2026), step=1)
        full_text_only = st.checkbox("Full text only", value=False)
        include_debug = st.checkbox("Diagnostics", value=False)
        run_button = st.button("Run analysis", type="primary", use_container_width=True)
        preview_button = st.button("Preview route", use_container_width=True)
        papers = stats.get("paper_count") or stats.get("enriched_papers") or "-"
        chunks = stats.get("chunk_count") or stats.get("full_text_chunks") or "-"
        topics = stats.get("topics") or "-"
        st.markdown(
            f"<div class='rse-sidebar-note'>{papers} papers · {chunks} chunks · {topics} research areas<br/>Answers are generated only after evidence passes the confidence gate.</div>",
            unsafe_allow_html=True,
        )

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

    render_summary(result, st.session_state.get("request_id"))
    st.markdown("<div class='rse-card'>", unsafe_allow_html=True)
    render_brief(result)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Top Supporting Evidence")
    render_top_evidence(result)

    counts = section_counts(result)
    tab_specs = [
        (f"Evidence Matrix · {counts.get('Evidence', '0 claims')}", render_evidence),
        (f"Reading Path · {counts.get('Reading Path', '0 stages')}", render_reading_path),
        (f"Open Problems · {counts.get('Open Problems', '0 found')}", render_open_problems),
        (f"Sources · {counts.get('Sources', '0 papers / 0 chunks')}", render_sources),
    ]
    if st.session_state.get("show_diagnostics"):
        tab_specs.append(("Diagnostics", render_diagnostics))
    tabs = st.tabs([label for label, _ in tab_specs])
    for tab, (_, renderer) in zip(tabs, tab_specs):
        with tab:
            renderer(result)


if __name__ == "__main__":
    main()
