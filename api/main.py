"""FastAPI backend for the Research Synthesis Engine."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.evidence_matrix import EvidenceMatrixError, build_evidence_matrix
from agent.open_problems import OpenProblemsError, build_open_problems_report
from agent.query_rewriter import ChatTurn, QueryRewriteResult, rewrite_query
from agent.reading_path import ReadingPathError, build_reading_path
from agent.research_guidance import ResearchGuidanceError
from agent.research_graph import ResearchAgentState, run_research_agent
from agent.synthesis import SynthesisError, build_research_brief
from full_text.index_chunks_qdrant import DEFAULT_COLLECTION as DEFAULT_CHUNK_COLLECTION
from retrieval.confidence import ConfidenceError, assess_confidence
from retrieval.index_qdrant import DEFAULT_COLLECTION as DEFAULT_PAPER_COLLECTION
from retrieval.index_qdrant import DEFAULT_QDRANT_URL, get_qdrant_client, load_env_file
from retrieval.router import route_query
from retrieval.unified_search import UnifiedSearchError, run_unified_search
from shared.schemas import (
    ConfidenceAssessment,
    EvidenceMatrix,
    OpenProblemsReport,
    QueryRoute,
    ReadingPath,
    ResearchBrief,
    RetrievedChunk,
    RetrievedPaper,
    UnifiedSearchResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TOP_K = 10
DEFAULT_MAX_PROBLEMS = 6
MAX_QUESTION_LENGTH = 2000
SUPPORTED_RESEARCH_TOPICS = [
    "Retrieval-Augmented Generation (RAG)",
    "Transformers / Attention Mechanisms",
    "LLM Evaluation & Hallucination Detection",
    "AI Agents & Tool Use",
    "Fine-tuning (LoRA / PEFT)",
]
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")
LOGGER = logging.getLogger("research_synthesis_engine.api")


class RequestMetrics(BaseModel):
    """Lightweight request timing metrics, returned only in debug mode."""

    routing_ms: float | None = None
    retrieval_ms: float | None = None
    confidence_ms: float | None = None
    brief_ms: float | None = None
    evidence_matrix_ms: float | None = None
    reading_path_ms: float | None = None
    open_problems_ms: float | None = None
    total_ms: float


class ApiErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None
    request_id: str | None = None


class ApiErrorResponse(BaseModel):
    error: ApiErrorBody
    detail: str | None = None


class ApiQueryRequest(BaseModel):
    """Request body for live query-time API endpoints."""

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(
        ...,
        min_length=3,
        max_length=MAX_QUESTION_LENGTH,
        validation_alias=AliasChoices("question", "query"),
        json_schema_extra={"examples": ["Compare RAG and self-verification methods."]},
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    paper_top_k: int | None = Field(default=None, ge=1, le=50)
    chunk_top_k: int | None = Field(default=None, ge=1, le=50)
    dense_top_k: int = Field(default=20, ge=1, le=100)
    sparse_top_k: int = Field(default=20, ge=1, le=100)
    apply_reranking: bool = True
    max_papers: int = Field(default=8, ge=1, le=20)
    max_problems: int = Field(default=DEFAULT_MAX_PROBLEMS, ge=1, le=20)
    research_areas: list[str] | None = None
    publication_year_min: int | None = Field(default=None, ge=1900, le=2100)
    publication_year_max: int | None = Field(default=None, ge=1900, le=2100)
    full_text_only: bool = False
    include_debug: bool = False
    chat_history: list[ChatTurn] = Field(default_factory=list, max_length=12)

    @property
    def query(self) -> str:
        """Backward-compatible internal name used by earlier Day 20 code."""
        return self.question

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped

    @field_validator("research_areas")
    @classmethod
    def validate_research_areas(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [item.strip() for item in value if item and item.strip()]
        unsupported = sorted(set(cleaned) - set(SUPPORTED_RESEARCH_TOPICS))
        if unsupported:
            raise ValueError(f"unsupported research areas: {unsupported}")
        return cleaned

    @model_validator(mode="after")
    def validate_year_range(self) -> "ApiQueryRequest":
        if (
            self.publication_year_min is not None
            and self.publication_year_max is not None
            and self.publication_year_min > self.publication_year_max
        ):
            raise ValueError("publication_year_min must not exceed publication_year_max")
        return self


class ApiRoutePreviewResponse(BaseModel):
    selected_route: str
    route_confidence: float
    reason: str
    matched_signals: list[str]


class ApiRetrievalResponse(BaseModel):
    question: str
    route: QueryRoute
    paper_result_count: int
    chunk_result_count: int
    paper_results: list[RetrievedPaper]
    chunk_results: list[RetrievedChunk]
    warnings: list[str] = Field(default_factory=list)
    metrics: RequestMetrics | None = None
    debug: dict[str, Any] | None = None


class ApiGuidanceResponse(BaseModel):
    """Combined API response for the main user-facing guidance endpoint."""

    question: str
    standalone_query: str
    rewrite_used: bool = False
    rewrite_method: str = "none"
    rewrite_reason: str = "No rewrite needed."
    retrieval: ApiRetrievalResponse
    confidence: ConfidenceAssessment
    brief: ResearchBrief | None = None
    evidence_matrix: EvidenceMatrix | None = None
    reading_path: ReadingPath | None = None
    open_problems: OpenProblemsReport | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics: RequestMetrics | None = None
    debug: dict[str, Any] | None = None


class AgentTraceStep(BaseModel):
    step: str
    status: str = "completed"
    detail: str | None = None


class ApiAgentResearchResponse(BaseModel):
    """Response for the bounded research-agent loop."""

    original_query: str
    standalone_query: str
    attempted_queries: list[str]
    retry_count: int
    confidence_decision: str | None
    retrieved_paper_count: int
    retrieved_chunk_count: int
    retrieval: ApiRetrievalResponse | None = None
    confidence: ConfidenceAssessment | None = None
    brief: ResearchBrief | None = None
    warnings: list[str] = Field(default_factory=list)
    trace: list[AgentTraceStep] = Field(default_factory=list)
    metrics: RequestMetrics | None = None
    debug: dict[str, Any] | None = None


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.values: dict[str, float] = {}

    def record(self, name: str, start: float) -> None:
        self.values[name] = round((time.perf_counter() - start) * 1000, 3)

    def metrics(self) -> RequestMetrics:
        return RequestMetrics(total_ms=round((time.perf_counter() - self.start) * 1000, 3), **self.values)


def get_allowed_cors_origins() -> list[str]:
    raw = os.getenv("RSE_CORS_ORIGINS") or os.getenv("STREAMLIT_ORIGINS")
    if not raw:
        return ["http://localhost:8501", "http://127.0.0.1:8501"]
    return [origin.strip() for origin in raw.split(",") if origin.strip() and origin.strip() != "*"]


app = FastAPI(
    title="Research Synthesis Engine API",
    version="0.1.0",
    description="Typed API for route-aware retrieval and grounded research synthesis over the local AI research corpus.",
    responses={
        422: {"model": ApiErrorResponse, "description": "Validation error"},
        503: {"model": ApiErrorResponse, "description": "Service dependency or generation error"},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
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


def valid_request_id(value: str | None) -> bool:
    return bool(value and REQUEST_ID_PATTERN.match(value))


def request_id_from_request(request: Request | None) -> str | None:
    if request is None:
        return None
    return getattr(request.state, "request_id", None)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID")
    request_id = incoming if valid_request_id(incoming) else str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        LOGGER.info(
            "api_request",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "method": request.method,
                "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            },
        )
    response.headers["X-Request-ID"] = request_id
    return response


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def error_payload(
    *,
    code: str,
    message: str,
    request_id: str | None,
    details: Any | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": json_safe(details),
            "request_id": request_id,
        },
        "detail": detail or message,
    }


def classify_service_error(exc: Exception) -> tuple[str, int, str]:
    text = str(exc).lower()
    if isinstance(exc, ValueError):
        return "VALIDATION_ERROR", 400, "Invalid request."
    if "openai_api_key" in text or "api key" in text or "configuration" in text:
        return "CONFIGURATION_ERROR", 503, "Required service configuration is missing."
    if "qdrant" in text or "connection" in text or "collection" in text:
        return "QDRANT_UNAVAILABLE", 503, "Vector store is unavailable."
    if isinstance(exc, (SynthesisError, ReadingPathError, OpenProblemsError)):
        return "LLM_GENERATION_FAILED", 503, "Grounded generation failed."
    if "insufficient" in text:
        return "INSUFFICIENT_EVIDENCE", 200, "Retrieved evidence is insufficient for confident generation."
    if isinstance(exc, UnifiedSearchError):
        return "RETRIEVAL_FAILED", 503, "Unable to retrieve research evidence."
    return "INTERNAL_ERROR", 500, "Internal API error."


def service_error(exc: Exception, request: Request | None = None) -> HTTPException:
    code, status_code, message = classify_service_error(exc)
    request_id = request_id_from_request(request)
    return HTTPException(
        status_code=status_code,
        detail=error_payload(code=code, message=message, request_id=request_id, detail=str(exc)),
    )


def optional_generation_warning(section: str, exc: Exception) -> str:
    return f"{section} could not be generated for this request; the core answer still uses retrieved evidence."


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = request_id_from_request(request)
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        payload = exc.detail
        payload["error"]["request_id"] = payload["error"].get("request_id") or request_id
    else:
        payload = error_payload(
            code="INTERNAL_ERROR" if exc.status_code >= 500 else "VALIDATION_ERROR",
            message="Request failed.",
            request_id=request_id,
            detail=str(exc.detail),
        )
    return JSONResponse(status_code=exc.status_code, content=payload, headers={"X-Request-ID": request_id or ""})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = request_id_from_request(request)
    payload = error_payload(
        code="VALIDATION_ERROR",
        message="Invalid request.",
        request_id=request_id,
        details=exc.errors(),
        detail="Invalid request.",
    )
    return JSONResponse(status_code=422, content=payload, headers={"X-Request-ID": request_id or ""})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request_id_from_request(request)
    LOGGER.exception("api_unhandled_error", extra={"request_id": request_id, "endpoint": request.url.path})
    payload = error_payload(code="INTERNAL_ERROR", message="Internal API error.", request_id=request_id)
    return JSONResponse(status_code=500, content=payload, headers={"X-Request-ID": request_id or ""})


def retrieve_for_request(request: ApiQueryRequest, *, query_override: str | None = None) -> UnifiedSearchResponse:
    return run_unified_search(
        query_override or request.question,
        top_k=request.top_k,
        paper_top_k=request.paper_top_k,
        chunk_top_k=request.chunk_top_k,
        dense_top_k=request.dense_top_k,
        sparse_top_k=request.sparse_top_k,
        apply_reranking=request.apply_reranking,
    )


def candidate_year(candidate: Any) -> int | None:
    return getattr(candidate, "year", None)


def candidate_topic(candidate: Any) -> str | None:
    return getattr(candidate, "topic", None)


def matches_filters(candidate: Any, request: ApiQueryRequest) -> bool:
    if request.research_areas and candidate_topic(candidate) not in request.research_areas:
        return False
    year = candidate_year(candidate)
    if request.publication_year_min is not None and (year is None or year < request.publication_year_min):
        return False
    if request.publication_year_max is not None and (year is None or year > request.publication_year_max):
        return False
    return True


def apply_request_filters(response: UnifiedSearchResponse, request: ApiQueryRequest) -> tuple[UnifiedSearchResponse, list[str]]:
    warnings: list[str] = []
    has_filters = bool(
        request.research_areas
        or request.publication_year_min is not None
        or request.publication_year_max is not None
        or request.full_text_only
    )
    if not has_filters:
        return response, warnings

    paper_results = [paper for paper in response.paper_results if matches_filters(paper, request)]
    chunk_results = [chunk for chunk in response.chunk_results if matches_filters(chunk, request)]
    if request.full_text_only:
        paper_results = []
        warnings.append("full_text_only keeps chunk-level results and omits abstract-only paper results.")
    warnings.append("Filters are applied after retrieval in this API version; upstream retrieval is not yet constrained by these filters.")
    filtered = response.model_copy(
        update={
            "paper_results": paper_results,
            "chunk_results": chunk_results,
            "paper_result_count": len(paper_results),
            "chunk_result_count": len(chunk_results),
        }
    )
    return filtered, warnings


def strip_score_breakdown(candidate: Any) -> Any:
    return candidate.model_copy(update={"score_breakdown": None})


def sanitize_retrieval_for_debug(response: UnifiedSearchResponse, include_debug: bool) -> UnifiedSearchResponse:
    if include_debug:
        return response
    return response.model_copy(
        update={
            "route": response.route.model_copy(update={"matched_signals": []}),
            "paper_results": [strip_score_breakdown(paper) for paper in response.paper_results],
            "chunk_results": [strip_score_breakdown(chunk) for chunk in response.chunk_results],
        }
    )


def collect_score_breakdowns(response: UnifiedSearchResponse) -> dict[str, Any]:
    breakdowns: dict[str, Any] = {}
    for result in [*response.paper_results, *response.chunk_results]:
        source_id = getattr(result, "chunk_id", None) or getattr(result, "paper_id", None) or getattr(result, "title", "unknown")
        if getattr(result, "score_breakdown", None):
            breakdowns[str(source_id)] = result.score_breakdown
    return breakdowns


def retrieval_debug(response: UnifiedSearchResponse, confidence: ConfidenceAssessment | None = None, metrics: RequestMetrics | None = None) -> dict[str, Any]:
    debug = {
        "route_reason": response.route.reason,
        "matched_signals": response.route.matched_signals,
        "score_breakdowns": collect_score_breakdowns(response),
    }
    if confidence is not None:
        debug["confidence_signals"] = confidence.signals
    if metrics is not None:
        debug["metrics"] = metrics.model_dump()
    return debug


def build_retrieval_response(
    response: UnifiedSearchResponse,
    *,
    request: ApiQueryRequest,
    warnings: list[str] | None = None,
    metrics: RequestMetrics | None = None,
    confidence: ConfidenceAssessment | None = None,
) -> ApiRetrievalResponse:
    visible_response = sanitize_retrieval_for_debug(response, request.include_debug)
    return ApiRetrievalResponse(
        question=visible_response.query,
        route=visible_response.route,
        paper_result_count=visible_response.paper_result_count,
        chunk_result_count=visible_response.chunk_result_count,
        paper_results=visible_response.paper_results,
        chunk_results=visible_response.chunk_results,
        warnings=warnings or [],
        metrics=metrics if request.include_debug else None,
        debug=retrieval_debug(response, confidence, metrics) if request.include_debug else None,
    )


def build_agent_trace(state: ResearchAgentState) -> list[AgentTraceStep]:
    trace = [
        AgentTraceStep(
            step="Context rewrite",
            detail=f"Standalone query: {state.get('standalone_query') or state.get('original_query')}",
        )
    ]
    attempted_queries = state.get("attempted_queries", []) or []
    retry_count = state.get("retry_count", 0)
    for index, query in enumerate(attempted_queries, start=1):
        trace.append(AgentTraceStep(step=f"Retrieval attempt {index}", detail=query))
        if index <= retry_count:
            trace.append(
                AgentTraceStep(
                    step=f"CRAG retry {index}",
                    detail="Low confidence triggered reflection rewriting.",
                )
            )
    trace.append(
        AgentTraceStep(
            step="Confidence check",
            detail=f"Decision: {state.get('confidence_decision') or 'unknown'}",
        )
    )
    brief = state.get("brief")
    trace.append(
        AgentTraceStep(
            step="Synthesis",
            detail=f"Brief status: {brief.status if brief else 'not generated'}",
        )
    )
    return trace


def run_timed_optional_section(section: str, builder: Callable[[], Any]) -> tuple[str, Any | None, Exception | None, float]:
    started = time.perf_counter()
    try:
        result = builder()
        return section, result, None, round((time.perf_counter() - started) * 1000, 3)
    except (EvidenceMatrixError, ReadingPathError, OpenProblemsError) as exc:
        return section, None, exc, round((time.perf_counter() - started) * 1000, 3)


def build_agent_response(
    state: ResearchAgentState,
    *,
    request: ApiQueryRequest,
    warnings: list[str],
    metrics: RequestMetrics,
) -> ApiAgentResearchResponse:
    retrieval = state.get("retrieval_response")
    confidence_result = state.get("confidence")
    brief_result = state.get("brief")
    retrieval_response = None
    if retrieval is not None:
        retrieval_response = build_retrieval_response(
            retrieval,
            request=request,
            warnings=warnings,
            metrics=metrics,
            confidence=confidence_result,
        )
    debug = None
    if request.include_debug:
        debug = {
            "attempted_queries": state.get("attempted_queries", []) or [],
            "warnings": warnings,
        }
        if retrieval is not None:
            debug.update(retrieval_debug(retrieval, confidence_result, metrics))
    return ApiAgentResearchResponse(
        original_query=state["original_query"],
        standalone_query=state["standalone_query"],
        attempted_queries=state.get("attempted_queries", []) or [],
        retry_count=state["retry_count"],
        confidence_decision=state.get("confidence_decision"),
        retrieved_paper_count=len(state.get("retrieved_papers", []) or []),
        retrieved_chunk_count=len(state.get("retrieved_chunks", []) or []),
        retrieval=retrieval_response,
        confidence=confidence_result,
        brief=brief_result,
        warnings=list(dict.fromkeys(warnings)),
        trace=build_agent_trace(state),
        metrics=metrics if request.include_debug else None,
        debug=debug,
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


def publication_year_range(path: Path) -> tuple[int | None, int | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    years = [item.get("year") for item in payload if isinstance(item, dict) and isinstance(item.get("year"), int)]
    return (min(years), max(years)) if years else (None, None)


def corpus_stats() -> dict[str, Any]:
    year_min, year_max = publication_year_range(DATA_DIR / "enriched_papers_final.json")
    return {
        "topics": 5,
        "paper_count": count_json_records(DATA_DIR / "enriched_papers_final.json"),
        "full_text_paper_count": count_json_records(DATA_DIR / "full_text_papers.json"),
        "chunk_count": count_json_records(DATA_DIR / "full_text_chunks.json"),
        "raw_papers": count_json_records(DATA_DIR / "raw_papers.json"),
        "enriched_papers": count_json_records(DATA_DIR / "enriched_papers_final.json"),
        "embedded_papers": count_json_records(DATA_DIR / "embedded_papers.json"),
        "full_text_sources": count_json_records(DATA_DIR / "full_text_sources.json"),
        "full_text_papers": count_json_records(DATA_DIR / "full_text_papers.json"),
        "full_text_chunks": count_json_records(DATA_DIR / "full_text_chunks.json"),
        "embedded_full_text_chunks": count_json_records(DATA_DIR / "embedded_full_text_chunks.json"),
        "publication_year_min": year_min,
        "publication_year_max": year_max,
        "paper_collection": DEFAULT_PAPER_COLLECTION,
        "chunk_collection": DEFAULT_CHUNK_COLLECTION,
        "qdrant_collections": [DEFAULT_PAPER_COLLECTION, DEFAULT_CHUNK_COLLECTION],
        "supported_research_topics": SUPPORTED_RESEARCH_TOPICS,
        "limitations": [
            "The corpus is curated around five AI research topics.",
            "Full-text coverage is limited to legally available PDFs.",
        ],
    }


def dependency_status() -> dict[str, str]:
    load_env_file(PROJECT_ROOT / ".env")
    qdrant_url = os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
    dependencies = {
        "openai_configuration": "available" if os.getenv("OPENAI_API_KEY") else "missing",
        "qdrant": "unavailable",
        "paper_collection": "unavailable",
        "chunk_collection": "unavailable",
    }
    try:
        client = get_qdrant_client(url=qdrant_url)
        paper_available = bool(client.collection_exists(DEFAULT_PAPER_COLLECTION))
        chunk_available = bool(client.collection_exists(DEFAULT_CHUNK_COLLECTION))
        dependencies["qdrant"] = "available"
        dependencies["paper_collection"] = "available" if paper_available else "missing"
        dependencies["chunk_collection"] = "available" if chunk_available else "missing"
    except Exception:
        dependencies["qdrant"] = "unavailable"
    return dependencies


@app.get(
    "/health",
    tags=["Health"],
    summary="Check API and dependency health",
    responses={200: {"description": "Safe health summary without secrets"}},
)
def health() -> dict[str, Any]:
    dependencies = dependency_status()
    degraded = any(value != "available" for value in dependencies.values())
    return {
        "status": "degraded" if degraded else "healthy",
        "service": "research-synthesis-engine",
        "version": "0.1.0",
        "dependencies": dependencies,
    }


@app.get("/corpus/stats", tags=["Corpus"], summary="Return local corpus statistics")
def get_corpus_stats() -> dict[str, Any]:
    return corpus_stats()


@app.post("/route", response_model=ApiRoutePreviewResponse, tags=["Retrieval"], summary="Preview routing decision without retrieval")
def route_preview(request: ApiQueryRequest) -> ApiRoutePreviewResponse:
    route = route_query(request.question)
    return ApiRoutePreviewResponse(
        selected_route=route.route,
        route_confidence=route.confidence,
        reason=route.reason,
        matched_signals=route.matched_signals,
    )


@app.post("/retrieve", response_model=ApiRetrievalResponse, tags=["Retrieval"], summary="Run route-aware retrieval without synthesis")
def retrieve(request: ApiQueryRequest, fastapi_request: Request) -> ApiRetrievalResponse:
    timer = Timer()
    try:
        rewrite_result = rewrite_query(request.question, request.chat_history)

        started = time.perf_counter()
        retrieval = retrieve_for_request(request, query_override=rewrite_result.standalone_query)
        timer.record("retrieval_ms", started)
        retrieval, warnings = apply_request_filters(retrieval, request)
        metrics = timer.metrics()
        return build_retrieval_response(retrieval, request=request, warnings=warnings, metrics=metrics)
    except SERVICE_ERRORS as exc:
        raise service_error(exc, fastapi_request) from exc


@app.post("/confidence", response_model=ConfidenceAssessment, tags=["Research"], summary="Assess retrieval confidence")
def confidence(request: ApiQueryRequest, fastapi_request: Request) -> ConfidenceAssessment:
    try:
        retrieval = retrieve_for_request(request)
        retrieval, _ = apply_request_filters(retrieval, request)
        return assess_confidence(retrieval)
    except SERVICE_ERRORS as exc:
        raise service_error(exc, fastapi_request) from exc


@app.post("/brief", response_model=ResearchBrief, tags=["Research"], summary="Generate a confidence-gated research brief")
def brief(request: ApiQueryRequest, fastapi_request: Request) -> ResearchBrief:
    try:
        retrieval = retrieve_for_request(request)
        retrieval, _ = apply_request_filters(retrieval, request)
        confidence_result = assess_confidence(retrieval)
        return build_research_brief(retrieval, confidence=confidence_result)
    except SERVICE_ERRORS as exc:
        raise service_error(exc, fastapi_request) from exc


@app.post("/evidence-matrix", response_model=EvidenceMatrix, tags=["Research"], summary="Build an evidence matrix from retrieved sources")
def evidence_matrix(request: ApiQueryRequest, fastapi_request: Request) -> EvidenceMatrix:
    try:
        retrieval = retrieve_for_request(request)
        retrieval, _ = apply_request_filters(retrieval, request)
        return build_evidence_matrix(retrieval, max_rows=request.top_k)
    except SERVICE_ERRORS as exc:
        raise service_error(exc, fastapi_request) from exc


@app.post("/reading-path", response_model=ReadingPath, tags=["Research"], summary="Generate a grounded reading path")
def reading_path(request: ApiQueryRequest, fastapi_request: Request) -> ReadingPath:
    try:
        retrieval = retrieve_for_request(request)
        retrieval, _ = apply_request_filters(retrieval, request)
        confidence_result = assess_confidence(retrieval)
        return build_reading_path(retrieval, confidence=confidence_result, max_papers=request.max_papers)
    except SERVICE_ERRORS as exc:
        raise service_error(exc, fastapi_request) from exc


@app.post("/open-problems", response_model=OpenProblemsReport, tags=["Research"], summary="Generate grounded open research problems")
def open_problems(request: ApiQueryRequest, fastapi_request: Request) -> OpenProblemsReport:
    try:
        retrieval = retrieve_for_request(request)
        retrieval, _ = apply_request_filters(retrieval, request)
        confidence_result = assess_confidence(retrieval)
        return build_open_problems_report(retrieval, confidence=confidence_result, max_problems=request.max_problems)
    except SERVICE_ERRORS as exc:
        raise service_error(exc, fastapi_request) from exc


@app.post(
    "/agent/research",
    response_model=ApiAgentResearchResponse,
    tags=["Agent"],
    summary="Run the bounded research-agent loop",
    responses={
        200: {"description": "Agent state, retrieval summary, confidence decision, and grounded brief"},
        503: {"model": ApiErrorResponse, "description": "Retrieval or generation dependency failed"},
    },
)
def agent_research(request: ApiQueryRequest, fastapi_request: Request) -> ApiAgentResearchResponse:
    timer = Timer()
    filter_warnings: list[str] = []

    def add_elapsed(name: str, start: float) -> None:
        elapsed = round((time.perf_counter() - start) * 1000, 3)
        timer.values[name] = round(timer.values.get(name, 0.0) + elapsed, 3)

    def filtered_searcher(query: str) -> UnifiedSearchResponse:
        started = time.perf_counter()
        retrieval = retrieve_for_request(request, query_override=query)
        add_elapsed("retrieval_ms", started)
        filtered, warnings = apply_request_filters(retrieval, request)
        filter_warnings.extend(warnings)
        return filtered

    def timed_confidence(response: UnifiedSearchResponse) -> ConfidenceAssessment:
        started = time.perf_counter()
        result = assess_confidence(response)
        add_elapsed("confidence_ms", started)
        return result

    def timed_synthesis(response: UnifiedSearchResponse, confidence: ConfidenceAssessment) -> ResearchBrief:
        started = time.perf_counter()
        result = build_research_brief(response, confidence=confidence)
        timer.record("brief_ms", started)
        return result

    try:
        state = run_research_agent(
            request.question,
            chat_history=request.chat_history,
            max_retries=2,
            searcher=filtered_searcher,
            confidence_checker=timed_confidence,
            synthesizer=timed_synthesis,
        )
        warnings = list(filter_warnings) + list(state.get("warnings", []) or [])
        brief_result = state.get("brief")
        if brief_result and brief_result.warning:
            warnings.append(brief_result.warning)
        metrics = timer.metrics()
        return build_agent_response(state, request=request, warnings=warnings, metrics=metrics)
    except SERVICE_ERRORS as exc:
        raise service_error(exc, fastapi_request) from exc


@app.post("/guidance", response_model=ApiGuidanceResponse, tags=["Research"], summary="Generate the complete research analyst response")
def guidance(request: ApiQueryRequest, fastapi_request: Request) -> ApiGuidanceResponse:
    timer = Timer()
    try:
        rewrite_result = rewrite_query(request.question, request.chat_history)

        started = time.perf_counter()
        retrieval = retrieve_for_request(request, query_override=rewrite_result.standalone_query)
        timer.record("retrieval_ms", started)
        retrieval, filter_warnings = apply_request_filters(retrieval, request)

        started = time.perf_counter()
        confidence_result = assess_confidence(retrieval)
        timer.record("confidence_ms", started)

        started = time.perf_counter()
        brief_result = build_research_brief(retrieval, confidence=confidence_result)
        timer.record("brief_ms", started)

        warnings: list[str] = list(filter_warnings)
        matrix_result = None
        reading_path_result = None
        open_problems_result = None

        if confidence_result.decision == "sufficient_evidence":
            optional_builders: dict[str, Callable[[], Any]] = {
                "evidence_matrix": lambda: build_evidence_matrix(retrieval, brief=brief_result, max_rows=request.top_k),
                "reading_path": lambda: build_reading_path(
                    retrieval,
                    confidence=confidence_result,
                    max_papers=request.max_papers,
                ),
                "open_problems": lambda: build_open_problems_report(
                    retrieval,
                    confidence=confidence_result,
                    max_problems=request.max_problems,
                ),
            }
            display_names = {
                "evidence_matrix": "Evidence matrix",
                "reading_path": "Reading path",
                "open_problems": "Open problems",
            }
            metric_names = {
                "evidence_matrix": "evidence_matrix_ms",
                "reading_path": "reading_path_ms",
                "open_problems": "open_problems_ms",
            }
            with ThreadPoolExecutor(max_workers=len(optional_builders)) as executor:
                futures = [
                    executor.submit(run_timed_optional_section, section, builder)
                    for section, builder in optional_builders.items()
                ]
                for future in as_completed(futures):
                    section, result, exc, elapsed_ms = future.result()
                    timer.values[metric_names[section]] = elapsed_ms
                    if exc is not None:
                        LOGGER.warning("guidance_optional_generation_failed", extra={"section": section})
                        warnings.append(optional_generation_warning(display_names[section], exc))
                        continue
                    if section == "evidence_matrix":
                        matrix_result = result
                    elif section == "reading_path":
                        reading_path_result = result
                        warnings.extend(reading_path_result.limitations)
                    elif section == "open_problems":
                        open_problems_result = result
                        warnings.extend(open_problems_result.corpus_limitations)
                        warnings.extend(open_problems_result.evidence_gaps)
        else:
            warnings.extend([confidence_result.reason, confidence_result.recommended_action])

        if brief_result.warning:
            warnings.append(brief_result.warning)

        metrics = timer.metrics()
        retrieval_response = build_retrieval_response(
            retrieval,
            request=request,
            warnings=filter_warnings,
            metrics=metrics,
            confidence=confidence_result,
        )
        return ApiGuidanceResponse(
            question=request.question,
            standalone_query=rewrite_result.standalone_query,
            rewrite_used=rewrite_result.rewrite_used,
            rewrite_method=rewrite_result.method,
            rewrite_reason=rewrite_result.reason,
            retrieval=retrieval_response,
            confidence=confidence_result,
            brief=brief_result,
            evidence_matrix=matrix_result,
            reading_path=reading_path_result,
            open_problems=open_problems_result,
            warnings=list(dict.fromkeys(warnings)),
            metrics=metrics if request.include_debug else None,
            debug=retrieval_debug(retrieval, confidence_result, metrics) if request.include_debug else None,
        )
    except SERVICE_ERRORS as exc:
        raise service_error(exc, fastapi_request) from exc
