"""FastAPI backend for the Research Synthesis Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from agent.evidence_matrix import EvidenceMatrixError, build_evidence_matrix
from agent.open_problems import OpenProblemsError, build_open_problems_report
from agent.reading_path import ReadingPathError, build_reading_path
from agent.research_guidance import ResearchGuidanceError
from agent.synthesis import SynthesisError, build_research_brief
from retrieval.confidence import ConfidenceError, assess_confidence
from retrieval.unified_search import UnifiedSearchError, run_unified_search
from shared.schemas import (
    ConfidenceAssessment,
    EvidenceMatrix,
    OpenProblemsReport,
    ReadingPath,
    ResearchBrief,
    UnifiedSearchResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TOP_K = 10
DEFAULT_MAX_PROBLEMS = 6


class ApiQueryRequest(BaseModel):
    """Request body for live query-time API endpoints."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    paper_top_k: int | None = Field(default=None, ge=1, le=50)
    chunk_top_k: int | None = Field(default=None, ge=1, le=50)
    dense_top_k: int = Field(default=20, ge=1, le=100)
    sparse_top_k: int = Field(default=20, ge=1, le=100)
    apply_reranking: bool = True
    max_papers: int = Field(default=8, ge=1, le=20)
    max_problems: int = Field(default=DEFAULT_MAX_PROBLEMS, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped


class ApiGuidanceResponse(BaseModel):
    """Combined API response for the main user-facing guidance endpoint."""

    query: str
    retrieval: UnifiedSearchResponse
    confidence: ConfidenceAssessment
    brief: ResearchBrief | None = None
    evidence_matrix: EvidenceMatrix | None = None
    reading_path: ReadingPath | None = None
    open_problems: OpenProblemsReport | None = None
    warnings: list[str] = Field(default_factory=list)


app = FastAPI(
    title="Research Synthesis Engine API",
    version="0.1.0",
    description="Route-aware retrieval and grounded research synthesis over the local AI research corpus.",
)


SERVICE_ERRORS = (
    UnifiedSearchError,
    ConfidenceError,
    SynthesisError,
    EvidenceMatrixError,
    ReadingPathError,
    OpenProblemsError,
    ResearchGuidanceError,
    ValueError,
)


def service_error(exc: Exception) -> HTTPException:
    status_code = 400 if isinstance(exc, ValueError) else 503
    return HTTPException(status_code=status_code, detail=str(exc))


def retrieve_for_request(request: ApiQueryRequest) -> UnifiedSearchResponse:
    return run_unified_search(
        request.query,
        top_k=request.top_k,
        paper_top_k=request.paper_top_k,
        chunk_top_k=request.chunk_top_k,
        dense_top_k=request.dense_top_k,
        sparse_top_k=request.sparse_top_k,
        apply_reranking=request.apply_reranking,
    )


def count_json_records(path: Path) -> int | None:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("papers", "chunks", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return None


def corpus_stats() -> dict[str, Any]:
    return {
        "topics": 5,
        "raw_papers": count_json_records(DATA_DIR / "raw_papers.json"),
        "enriched_papers": count_json_records(DATA_DIR / "enriched_papers_final.json"),
        "embedded_papers": count_json_records(DATA_DIR / "embedded_papers.json"),
        "full_text_sources": count_json_records(DATA_DIR / "full_text_sources.json"),
        "full_text_papers": count_json_records(DATA_DIR / "full_text_papers.json"),
        "full_text_chunks": count_json_records(DATA_DIR / "full_text_chunks.json"),
        "embedded_full_text_chunks": count_json_records(DATA_DIR / "embedded_full_text_chunks.json"),
        "qdrant_collections": ["research_papers", "research_paper_chunks"],
        "supported_research_topics": [
            "Retrieval-Augmented Generation (RAG)",
            "Transformers / Attention Mechanisms",
            "LLM Evaluation & Hallucination Detection",
            "AI Agents & Tool Use",
            "Fine-tuning (LoRA / PEFT)",
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "research-synthesis-engine"}


@app.get("/corpus/stats")
def get_corpus_stats() -> dict[str, Any]:
    return corpus_stats()


@app.post("/retrieve", response_model=UnifiedSearchResponse)
def retrieve(request: ApiQueryRequest) -> UnifiedSearchResponse:
    try:
        return retrieve_for_request(request)
    except SERVICE_ERRORS as exc:
        raise service_error(exc) from exc


@app.post("/confidence", response_model=ConfidenceAssessment)
def confidence(request: ApiQueryRequest) -> ConfidenceAssessment:
    try:
        retrieval = retrieve_for_request(request)
        return assess_confidence(retrieval)
    except SERVICE_ERRORS as exc:
        raise service_error(exc) from exc


@app.post("/brief", response_model=ResearchBrief)
def brief(request: ApiQueryRequest) -> ResearchBrief:
    try:
        retrieval = retrieve_for_request(request)
        confidence_result = assess_confidence(retrieval)
        return build_research_brief(retrieval, confidence=confidence_result)
    except SERVICE_ERRORS as exc:
        raise service_error(exc) from exc


@app.post("/evidence-matrix", response_model=EvidenceMatrix)
def evidence_matrix(request: ApiQueryRequest) -> EvidenceMatrix:
    try:
        retrieval = retrieve_for_request(request)
        return build_evidence_matrix(retrieval, max_rows=request.top_k)
    except SERVICE_ERRORS as exc:
        raise service_error(exc) from exc


@app.post("/reading-path", response_model=ReadingPath)
def reading_path(request: ApiQueryRequest) -> ReadingPath:
    try:
        retrieval = retrieve_for_request(request)
        confidence_result = assess_confidence(retrieval)
        return build_reading_path(retrieval, confidence=confidence_result, max_papers=request.max_papers)
    except SERVICE_ERRORS as exc:
        raise service_error(exc) from exc


@app.post("/open-problems", response_model=OpenProblemsReport)
def open_problems(request: ApiQueryRequest) -> OpenProblemsReport:
    try:
        retrieval = retrieve_for_request(request)
        confidence_result = assess_confidence(retrieval)
        return build_open_problems_report(retrieval, confidence=confidence_result, max_problems=request.max_problems)
    except SERVICE_ERRORS as exc:
        raise service_error(exc) from exc


@app.post("/guidance", response_model=ApiGuidanceResponse)
def guidance(request: ApiQueryRequest) -> ApiGuidanceResponse:
    try:
        retrieval = retrieve_for_request(request)
        confidence_result = assess_confidence(retrieval)
        brief_result = build_research_brief(retrieval, confidence=confidence_result)
        matrix_result = build_evidence_matrix(retrieval, brief=brief_result, max_rows=request.top_k)
        warnings: list[str] = []

        if confidence_result.decision == "sufficient_evidence":
            reading_path_result = build_reading_path(retrieval, confidence=confidence_result, max_papers=request.max_papers)
            open_problems_result = build_open_problems_report(
                retrieval,
                confidence=confidence_result,
                max_problems=request.max_problems,
            )
            warnings.extend(reading_path_result.limitations)
            warnings.extend(open_problems_result.corpus_limitations)
            warnings.extend(open_problems_result.evidence_gaps)
        else:
            reading_path_result = None
            open_problems_result = None
            warnings.extend([confidence_result.reason, confidence_result.recommended_action])

        if brief_result.warning:
            warnings.append(brief_result.warning)

        return ApiGuidanceResponse(
            query=request.query,
            retrieval=retrieval,
            confidence=confidence_result,
            brief=brief_result,
            evidence_matrix=matrix_result,
            reading_path=reading_path_result,
            open_problems=open_problems_result,
            warnings=list(dict.fromkeys(warnings)),
        )
    except SERVICE_ERRORS as exc:
        raise service_error(exc) from exc
