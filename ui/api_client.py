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
