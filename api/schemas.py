"""Pydantic request/response models for the API."""

from __future__ import annotations

import os
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.query_rewriter import ChatTurn
from shared.schemas import (
    ConfidenceAssessment,
    EvidenceMatrix,
    OpenProblemsReport,
    QueryRoute,
    ReadingPath,
    ResearchBrief,
    RetrievedChunk,
    RetrievedPaper,
)


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


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_APPLY_RERANKING = env_bool("RSE_APPLY_RERANKING", True)


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
    apply_reranking: bool = DEFAULT_APPLY_RERANKING
    max_papers: int = Field(default=8, ge=1, le=20)
    max_problems: int = Field(default=DEFAULT_MAX_PROBLEMS, ge=1, le=20)
    research_areas: list[str] | None = None
    publication_year_min: int | None = Field(default=None, ge=1900, le=2100)
    publication_year_max: int | None = Field(default=None, ge=1900, le=2100)
    full_text_only: bool = False
    include_debug: bool = False
    include_evidence_matrix: bool = True
    include_reading_path: bool = True
    include_open_problems: bool = True
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
