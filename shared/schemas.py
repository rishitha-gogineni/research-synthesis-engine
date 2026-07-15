"""Pydantic schemas shared across ingestion, retrieval, and generation."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Paper(BaseModel):
    """Raw academic paper metadata fetched from an external source."""

    paper_id: Optional[str] = None
    title: str = Field(..., min_length=1)
    abstract: str = Field(..., min_length=1)
    authors: list[str] = Field(default_factory=list)
    citation_count: int = Field(default=0, ge=0)
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    topic: str = Field(..., min_length=1)


class EnrichedPaper(Paper):
    """Paper metadata plus structured fields extracted from the abstract."""

    main_contribution: str = Field(..., min_length=1)
    methodology: str = Field(..., min_length=1)
    dataset_used: str = Field(
        ...,
        min_length=1,
        description='Dataset or benchmark stated in the abstract; use "not stated in abstract" when absent.',
    )
    key_result: str = Field(
        ...,
        min_length=1,
        description='Key result stated in the abstract; use "not stated in abstract" when absent.',
    )
    limitations: str = Field(
        ...,
        min_length=1,
        description='Limitation stated in the abstract; use "not stated in abstract" when absent.',
    )


class RetrievalRequest(BaseModel):
    """Input schema for a research retrieval tool call."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    dense_top_k: int = Field(default=20, ge=1, le=100)
    sparse_top_k: int = Field(default=20, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped


class RetrievedPaper(BaseModel):
    """A candidate paper returned by the retrieval tool."""

    paper_id: Optional[str] = None
    title: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    year: Optional[int] = None
    citation_count: int = Field(default=0, ge=0)
    authors: list[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    main_contribution: Optional[str] = None
    methodology: Optional[str] = None
    dataset_used: Optional[str] = None
    key_result: Optional[str] = None
    limitations: Optional[str] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    hybrid_score: float = Field(default=0.0, ge=0.0)
    matched_by: list[str] = Field(default_factory=list)
    rerank_raw_score: Optional[float] = None
    rerank_score: Optional[float] = None
    citation_score: Optional[float] = None
    blended_score: Optional[float] = None
    score_breakdown: Optional[dict[str, Any]] = None


class RetrievalResponse(BaseModel):
    """Output schema for a research retrieval tool call."""

    query: str = Field(..., min_length=1)
    result_count: int = Field(..., ge=0)
    results: list[RetrievedPaper] = Field(default_factory=list)


QueryRouteName = Literal["paper_level", "chunk_level", "hybrid_both", "metadata_filter"]


class QueryRoute(BaseModel):
    """Routing decision for a user research question."""

    query: str = Field(..., min_length=1)
    route: QueryRouteName
    reason: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    matched_signals: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped


class RetrievedChunk(BaseModel):
    """A full-text chunk candidate returned by unified retrieval."""

    chunk_id: Optional[str] = None
    paper_id: Optional[str] = None
    title: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    year: Optional[int] = None
    citation_count: int = Field(default=0, ge=0)
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    section_hint: Optional[str] = None
    word_count: Optional[int] = None
    text: str = Field(..., min_length=1)
    pdf_url: Optional[str] = None
    source_type: Optional[str] = None
    page_count: Optional[int] = None
    dense_score: Optional[float] = None
    matched_by: list[str] = Field(default_factory=list)
    rerank_raw_score: Optional[float] = None
    rerank_score: Optional[float] = None
    citation_score: Optional[float] = None
    blended_score: Optional[float] = None
    score_breakdown: Optional[dict[str, Any]] = None


class UnifiedSearchRequest(BaseModel):
    """Input schema for route-aware unified retrieval."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    paper_top_k: int = Field(default=10, ge=1, le=50)
    chunk_top_k: int = Field(default=10, ge=1, le=50)
    dense_top_k: int = Field(default=20, ge=1, le=100)
    sparse_top_k: int = Field(default=20, ge=1, le=100)
    apply_reranking: bool = True

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped


class UnifiedSearchResponse(BaseModel):
    """Output schema for route-aware unified retrieval."""

    query: str = Field(..., min_length=1)
    route: QueryRoute
    paper_result_count: int = Field(..., ge=0)
    chunk_result_count: int = Field(..., ge=0)
    paper_results: list[RetrievedPaper] = Field(default_factory=list)
    chunk_results: list[RetrievedChunk] = Field(default_factory=list)

