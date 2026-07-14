"""Pydantic schemas shared across ingestion, retrieval, and generation."""

from typing import Literal, Optional

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

