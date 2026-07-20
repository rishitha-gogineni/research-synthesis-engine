"""API client and formatting helpers for the Streamlit workbench."""

from __future__ import annotations

import os
import uuid
from typing import Any

import requests


DEFAULT_API_URL = "http://localhost:8000"
SUPPORTED_RESEARCH_TOPICS = [
    "Retrieval-Augmented Generation (RAG)",
    "Transformers / Attention Mechanisms",
    "LLM Evaluation & Hallucination Detection",
    "AI Agents & Tool Use",
    "Fine-tuning (LoRA / PEFT)",
]
SUGGESTED_QUESTIONS = [
    "What are the main approaches for reducing hallucinations in LLMs?",
    "Which datasets and metrics are used to evaluate hallucination detection?",
    "Compare RAG and self-verification methods for reducing hallucinations.",
    "What are common limitations in AI agent tool-use papers?",
    "Which LoRA and PEFT papers should I read first?",
]


def api_base_url() -> str:
    return (os.getenv("RSE_API_URL") or DEFAULT_API_URL).rstrip("/")


def new_request_id() -> str:
    return f"ui-{uuid.uuid4()}"


def build_guidance_payload(
    *,
    question: str,
    top_k: int,
    research_areas: list[str] | None = None,
    publication_year_min: int | None = None,
    publication_year_max: int | None = None,
    full_text_only: bool = False,
    include_debug: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question.strip(),
        "top_k": top_k,
        "include_debug": include_debug,
        "full_text_only": full_text_only,
    }
    if research_areas:
        payload["research_areas"] = research_areas
    if publication_year_min is not None:
        payload["publication_year_min"] = publication_year_min
    if publication_year_max is not None:
        payload["publication_year_max"] = publication_year_max
    return payload


def post_api(endpoint: str, payload: dict[str, Any], *, request_id: str | None = None, timeout: int = 120) -> tuple[dict[str, Any], str | None]:
    url = f"{api_base_url()}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-ID"] = request_id
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response_id = response.headers.get("X-Request-ID")
    try:
        body = response.json()
    except ValueError:
        body = {"error": {"code": "INVALID_RESPONSE", "message": response.text or "API returned a non-JSON response.", "request_id": response_id}}
    if response.status_code >= 400:
        return {"_error_status": response.status_code, **body}, response_id
    return body, response_id


def get_api(endpoint: str, *, timeout: int = 20) -> tuple[dict[str, Any], str | None]:
    response = requests.get(f"{api_base_url()}{endpoint}", timeout=timeout)
    response_id = response.headers.get("X-Request-ID")
    try:
        body = response.json()
    except ValueError:
        body = {"error": {"code": "INVALID_RESPONSE", "message": response.text or "API returned a non-JSON response.", "request_id": response_id}}
    if response.status_code >= 400:
        return {"_error_status": response.status_code, **body}, response_id
    return body, response_id


def error_message(payload: dict[str, Any]) -> str | None:
    error = payload.get("error") if isinstance(payload, dict) else None
    if not error:
        return None
    code = error.get("code", "ERROR")
    message = error.get("message") or payload.get("detail") or "Request failed."
    request_id = error.get("request_id")
    suffix = f" Request ID: {request_id}" if request_id else ""
    return f"{code}: {message}{suffix}"


def evidence_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("evidence_matrix", {}).get("rows", []) or []
    return [
        {
            "Claim": row.get("claim"),
            "Sources": ", ".join(row.get("source_ids", [])),
            "Methodology": row.get("methodology"),
            "Dataset": row.get("dataset"),
            "Key Result": row.get("key_result"),
            "Limitation": row.get("limitation"),
            "Strength": row.get("evidence_strength"),
        }
        for row in rows
    ]


def reading_path_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    path = payload.get("reading_path") or {}
    rows: list[dict[str, Any]] = []
    for stage in path.get("stages", []) or []:
        for paper in stage.get("papers", []) or []:
            rows.append(
                {
                    "Order": paper.get("order"),
                    "Stage": stage.get("stage"),
                    "Title": paper.get("title"),
                    "Year": paper.get("publication_year"),
                    "Citations": paper.get("citation_count"),
                    "Reason": paper.get("reason_to_read"),
                    "Sources": ", ".join(paper.get("source_ids", [])),
                }
            )
    return rows


def open_problem_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    report = payload.get("open_problems") or {}
    return [
        {
            "Problem": problem.get("title"),
            "Category": problem.get("category"),
            "Strength": problem.get("evidence_strength"),
            "Why It Matters": problem.get("why_it_matters"),
            "Sources": ", ".join(problem.get("supporting_source_ids", [])),
        }
        for problem in report.get("problems", []) or []
    ]


def source_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval = payload.get("retrieval") or {}
    paper_rows = [
        {
            "Title": paper.get("title"),
            "Topic": paper.get("topic"),
            "Year": paper.get("year"),
            "Citations": paper.get("citation_count"),
            "Paper ID": paper.get("paper_id"),
            "Score": paper.get("blended_score") or paper.get("rerank_score") or paper.get("hybrid_score"),
        }
        for paper in retrieval.get("paper_results", []) or []
    ]
    chunk_rows = [
        {
            "Title": chunk.get("title"),
            "Topic": chunk.get("topic"),
            "Section": chunk.get("section_hint"),
            "Chunk": chunk.get("chunk_id"),
            "Paper ID": chunk.get("paper_id"),
            "Score": chunk.get("blended_score") or chunk.get("rerank_score") or chunk.get("dense_score"),
        }
        for chunk in retrieval.get("chunk_results", []) or []
    ]
    return paper_rows, chunk_rows


def summary_items(payload: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
    retrieval = payload.get("retrieval") or {}
    route = retrieval.get("route") or {}
    confidence = payload.get("confidence") or {}
    return {
        "Route": route.get("route") or "unknown",
        "Confidence": confidence.get("decision") or "unknown",
        "Papers": retrieval.get("paper_result_count", 0),
        "Chunks": retrieval.get("chunk_result_count", 0),
        "Request ID": request_id or "-",
    }


def metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = payload.get("metrics") or payload.get("retrieval", {}).get("metrics") or {}
    return [{"Metric": key, "Milliseconds": value} for key, value in metrics.items() if value is not None]

def theme_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    brief = payload.get("brief") or {}
    return [
        {
            "Theme": theme.get("theme"),
            "Description": theme.get("summary"),
            "Sources": ", ".join(theme.get("supporting_source_ids", [])),
        }
        for theme in brief.get("themes", []) or []
    ]


def _source_score(item: dict[str, Any]) -> float:
    for key in ("blended_score", "rerank_score", "hybrid_score", "dense_score", "sparse_score", "score"):
        value = item.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def top_supporting_evidence(payload: dict[str, Any], *, limit: int = 6) -> list[dict[str, Any]]:
    retrieval = payload.get("retrieval") or {}
    candidates: list[dict[str, Any]] = []
    for paper in retrieval.get("paper_results", []) or []:
        candidates.append(
            {
                "Source": "paper",
                "Title": paper.get("title"),
                "Year": paper.get("year"),
                "Citations": paper.get("citation_count"),
                "Topic": paper.get("topic"),
                "Why It Matters": paper.get("main_contribution") or paper.get("key_result") or paper.get("abstract"),
                "Source ID": f"paper:{paper.get('paper_id')}",
                "Score": _source_score(paper),
            }
        )
    for chunk in retrieval.get("chunk_results", []) or []:
        text = chunk.get("text") or ""
        candidates.append(
            {
                "Source": "full text",
                "Title": chunk.get("title"),
                "Year": chunk.get("year"),
                "Citations": chunk.get("citation_count"),
                "Topic": chunk.get("topic"),
                "Why It Matters": text[:360],
                "Source ID": f"chunk:{chunk.get('chunk_id')}",
                "Score": _source_score(chunk),
            }
        )

    seen: set[str] = set()
    rows = []
    for item in sorted(candidates, key=lambda row: (row["Score"], row.get("Citations") or 0), reverse=True):
        source_id = item["Source ID"]
        if source_id in seen:
            continue
        seen.add(source_id)
        item["Why It Matters"] = (item.get("Why It Matters") or "No evidence summary returned.").strip()
        rows.append(item)
        if len(rows) >= limit:
            break
    return rows


def query_intent(question: str) -> str:
    lowered = question.lower()
    if any(token in lowered for token in ("read", "reading", "start", "first", "path", "papers should")):
        return "reading"
    if any(token in lowered for token in ("limitation", "limitations", "open problem", "future work", "unsolved", "challenge")):
        return "limitations"
    if any(token in lowered for token in ("dataset", "benchmark", "metric", "evaluate", "evaluation")):
        return "evaluation"
    if any(token in lowered for token in ("compare", "versus", " vs ", "difference", "tradeoff")):
        return "comparison"
    return "overview"


def ordered_sections(question: str) -> list[str]:
    intent = query_intent(question)
    if intent == "reading":
        return ["Reading Path", "Brief", "Top Evidence", "Sources", "Evidence", "Open Problems", "Diagnostics"]
    if intent == "limitations":
        return ["Brief", "Open Problems", "Top Evidence", "Evidence", "Sources", "Reading Path", "Diagnostics"]
    if intent == "evaluation":
        return ["Brief", "Evidence", "Top Evidence", "Sources", "Reading Path", "Open Problems", "Diagnostics"]
    if intent == "comparison":
        return ["Brief", "Evidence", "Top Evidence", "Sources", "Reading Path", "Open Problems", "Diagnostics"]
    return ["Brief", "Top Evidence", "Evidence", "Reading Path", "Open Problems", "Sources", "Diagnostics"]

