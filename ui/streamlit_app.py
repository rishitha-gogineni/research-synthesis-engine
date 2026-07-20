"""Streamlit research analyst workbench for the Research Synthesis Engine."""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

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
    post_api,
    reading_path_rows,
    source_rows,
    summary_items,
)


st.set_page_config(page_title="Research Synthesis Engine", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    :root {
        --rse-accent: #0f766e;
        --rse-ink: #1f2933;
        --rse-muted: #64748b;
        --rse-line: #d9e2ec;
        --rse-bg: #f8fafc;
    }
    .stApp { background: var(--rse-bg); color: var(--rse-ink); }
    h1, h2, h3 { letter-spacing: 0 !important; }
    h1 { font-size: 1.65rem !important; margin-bottom: 0.2rem !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1.0rem !important; }
    section[data-testid="stSidebar"] { border-right: 1px solid var(--rse-line); }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--rse-line);
        padding: 0.7rem 0.8rem;
        border-radius: 6px;
    }
    div[data-testid="stMetric"] label { color: var(--rse-muted) !important; }
    div[data-testid="stMetricValue"] { font-size: 1.05rem !important; }
    .rse-status {
        border: 1px solid var(--rse-line);
        background: #ffffff;
        border-radius: 6px;
        padding: 0.6rem 0.75rem;
        font-size: 0.9rem;
    }
    .rse-muted { color: var(--rse-muted); font-size: 0.86rem; }
    .rse-warning {
        border-left: 3px solid #b45309;
        background: #fffbeb;
        padding: 0.6rem 0.75rem;
        border-radius: 4px;
        margin: 0.35rem 0;
    }
    .rse-source-id { color: var(--rse-accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
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


def render_brief(payload: dict):
    brief = payload.get("brief") or {}
    direct_answer = brief.get("direct_answer")
    if direct_answer:
        st.subheader("Direct Answer")
        st.write(direct_answer)
    themes = brief.get("themes") or []
    if themes:
        st.subheader("Themes")
        for theme in themes:
            with st.expander(theme.get("theme", "Theme"), expanded=True):
                st.write(theme.get("summary", ""))
                if theme.get("supporting_source_ids"):
                    st.markdown("<span class='rse-source-id'>" + ", ".join(theme["supporting_source_ids"]) + "</span>", unsafe_allow_html=True)
    bullets = brief.get("evidence_bullets") or []
    if bullets:
        st.subheader("Evidence Bullets")
        for item in bullets:
            st.write(f"- {item}")
    limitations = brief.get("limitations") or []
    if limitations:
        st.subheader("Limitations")
        for item in limitations:
            st.write(f"- {item}")
    if not any([direct_answer, themes, bullets, limitations]):
        st.info("No brief returned.")


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
        st.header("Query")
        preset = st.selectbox("Suggested question", SUGGESTED_QUESTIONS, index=0)
        question = st.text_area("Question", value=preset, height=110)
        research_areas = st.multiselect("Research areas", SUPPORTED_RESEARCH_TOPICS)
        top_k = st.slider("Top K", min_value=3, max_value=20, value=8, step=1)
        year_min, year_max = st.slider("Publication years", min_value=2017, max_value=2026, value=(2017, 2026), step=1)
        full_text_only = st.checkbox("Full-text only", value=False)
        include_debug = st.checkbox("Diagnostics", value=True)
        run_button = st.button("Run analysis", type="primary", use_container_width=True)
        preview_button = st.button("Preview route", use_container_width=True)

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

    result = st.session_state.get("guidance_result")
    if not result:
        st.markdown("<div class='rse-muted'>No analysis run in this session.</div>", unsafe_allow_html=True)
        return

    render_summary(result, st.session_state.get("request_id"))
    brief_tab, evidence_tab, reading_tab, problems_tab, sources_tab, diagnostics_tab = st.tabs(
        ["Brief", "Evidence", "Reading Path", "Open Problems", "Sources", "Diagnostics"]
    )
    with brief_tab:
        render_brief(result)
    with evidence_tab:
        render_evidence(result)
    with reading_tab:
        render_reading_path(result)
    with problems_tab:
        render_open_problems(result)
    with sources_tab:
        render_sources(result)
    with diagnostics_tab:
        render_diagnostics(result)


if __name__ == "__main__":
    main()
