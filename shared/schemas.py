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
ConfidenceDecisionName = Literal["sufficient_evidence", "broaden_search", "ask_clarifying_question", "insufficient_evidence"]
BriefStatusName = Literal["generated", "skipped_low_confidence"]
ReadingStageName = Literal[
    "foundational",
    "core_methods",
    "evaluation_and_benchmarks",
    "recent_advances",
    "limitations_and_open_problems",
]
OpenProblemCategoryName = Literal[
    "data",
    "evaluation",
    "methodology",
    "generalization",
    "scalability",
    "efficiency",
    "safety",
    "interpretability",
    "reproducibility",
    "deployment",
]
EvidenceStrengthName = Literal["strong", "moderate", "weak"]


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


class ConfidenceAssessment(BaseModel):
    """CRAG-style assessment of whether retrieved evidence is strong enough for synthesis."""

    query: str = Field(..., min_length=1)
    route: QueryRouteName
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    decision: ConfidenceDecisionName
    reason: str = Field(..., min_length=1)
    recommended_action: str = Field(..., min_length=1)
    signals: list[str] = Field(default_factory=list)
    result_count: int = Field(..., ge=0)
    top_score: float = Field(default=0.0, ge=0.0, le=1.0)
    route_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationQuery(BaseModel):
    """One human-readable retrieval evaluation query."""

    query: str = Field(..., min_length=1)
    expected_route: QueryRouteName
    expected_topics: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    expected_relevant_ids: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped


class EvidenceSource(BaseModel):
    """One retrieved source made available to synthesis and evidence matrix generation."""

    source_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    paper_id: Optional[str] = None
    chunk_id: Optional[str] = None
    year: Optional[int] = None
    citation_count: int = Field(default=0, ge=0)
    evidence_text: str = Field(..., min_length=1)
    score: Optional[float] = None


class BriefTheme(BaseModel):
    """A theme in a generated research brief."""

    theme: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    supporting_source_ids: list[str] = Field(default_factory=list)


class ResearchBrief(BaseModel):
    """Grounded research brief generated from retrieved evidence."""

    query: str = Field(..., min_length=1)
    status: BriefStatusName
    confidence_decision: ConfidenceDecisionName
    direct_answer: str = Field(default="")
    themes: list[BriefTheme] = Field(default_factory=list)
    evidence_bullets: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    open_problems: list[str] = Field(default_factory=list)
    sources: list[EvidenceSource] = Field(default_factory=list)
    warning: Optional[str] = None


class EvidenceMatrixRow(BaseModel):
    """One claim-to-evidence row for inspecting a synthesized answer."""

    claim: str = Field(..., min_length=1)
    supporting_papers: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    methodology: str = Field(default="not stated in retrieved evidence")
    dataset: str = Field(default="not stated in retrieved evidence")
    key_result: str = Field(default="not stated in retrieved evidence")
    limitation: str = Field(default="not stated in retrieved evidence")
    evidence_strength: str = Field(default="medium")
    evidence_snippet: str = Field(default="")


class EvidenceMatrix(BaseModel):
    """Structured evidence matrix derived from retrieved papers and chunks."""

    query: str = Field(..., min_length=1)
    rows: list[EvidenceMatrixRow] = Field(default_factory=list)
    markdown: str = Field(default="")


class ReadingPathItem(BaseModel):
    """One recommended paper in a grounded reading sequence."""

    order: int = Field(..., ge=1)
    stage: ReadingStageName
    paper_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    authors: list[str] = Field(default_factory=list)
    publication_year: Optional[int] = None
    citation_count: Optional[int] = Field(default=None, ge=0)
    topic: Optional[str] = None
    reason_to_read: str = Field(..., min_length=1)
    focus_points: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    connection_to_next: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)
    evidence_snippet: Optional[str] = None
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ReadingPathStage(BaseModel):
    """A stage in the grounded reading path."""

    stage: ReadingStageName
    description: str = Field(..., min_length=1)
    papers: list[ReadingPathItem] = Field(default_factory=list)


class ReadingPath(BaseModel):
    """Grounded reading path built from retrieved papers and chunks."""

    question: str = Field(..., min_length=1)
    stages: list[ReadingPathStage] = Field(default_factory=list)
    total_papers: int = Field(..., ge=0)
    confidence_decision: ConfidenceDecisionName
    limitations: list[str] = Field(default_factory=list)


class OpenProblem(BaseModel):
    """One unresolved research challenge grounded in retrieved evidence."""

    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    category: OpenProblemCategoryName
    why_it_matters: str = Field(..., min_length=1)
    evidence_summary: str = Field(..., min_length=1)
    supporting_paper_ids: list[str] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrengthName
    suggested_research_directions: list[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class OpenProblemsReport(BaseModel):
    """Grounded report of open research problems found in retrieved evidence."""

    question: str = Field(..., min_length=1)
    problems: list[OpenProblem] = Field(default_factory=list)
    recurring_limitations: list[str] = Field(default_factory=list)
    conflicting_findings: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    corpus_limitations: list[str] = Field(default_factory=list)
    confidence_decision: ConfidenceDecisionName


class ResearchGuidanceResponse(BaseModel):
    """Combined Day 19 guidance output from one unified retrieval response."""

    question: str = Field(..., min_length=1)
    confidence: ConfidenceAssessment
    reading_path: Optional[ReadingPath] = None
    open_problems: Optional[OpenProblemsReport] = None
    warnings: list[str] = Field(default_factory=list)

