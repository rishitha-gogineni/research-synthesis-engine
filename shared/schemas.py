"""Pydantic schemas shared across ingestion, retrieval, and generation."""

from typing import Optional

from pydantic import BaseModel, Field


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
