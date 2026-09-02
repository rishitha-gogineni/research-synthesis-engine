"""Pydantic schemas for validating LLM JSON responses.

The lead/judge/citation agents all parse raw JSON from an LLM response and,
until now, trusted its shape via `.get(...)` calls with defaults scattered
across the call sites. That silently tolerates a model drifting field names
or types. These schemas make the expected shape explicit and catch drift at
the boundary, right after `json.loads`, instead of downstream when a
missing/wrong-typed field causes a confusing error deeper in the pipeline.

Validation failures are handled the same way malformed JSON already is
(logged, safe default substituted) — never raised past the call site, since
a single LLM formatting slip should degrade gracefully, not crash a run.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field, ValidationError


class SubtaskSchema(BaseModel):
    id: str = ""
    objective: str = Field(default="", validation_alias=AliasChoices("objective", "task"))
    source: str = Field(default="", validation_alias=AliasChoices("source", "search_source"))
    queries: list[str] = Field(
        default_factory=list, validation_alias=AliasChoices("queries", "search_queries")
    )
    boundaries: str = Field(default="", validation_alias=AliasChoices("boundaries", "task_boundaries"))
    output_format: str = ""

    # The lead LLM occasionally drifts to alternate key names on the
    # looser-schema follow-up path (search_source/search_queries/task_boundaries).
    # subagent.py's run_subagent() tolerates both spellings via `.get(...) or
    # subtask["search_source"]` fallbacks — the aliases above normalize
    # whichever name the LLM used into the canonical field, so validated output
    # always has "source"/"queries"/"boundaries" populated either way.
    model_config = {"populate_by_name": True}


class PlanSchema(BaseModel):
    reasoning: str = ""
    subtasks: list[SubtaskSchema] = Field(default_factory=list)


class FollowUpPlanSchema(BaseModel):
    subtasks: list[SubtaskSchema] = Field(default_factory=list)


class SourceRefSchema(BaseModel):
    title: str = ""
    source: str = ""
    url: str = ""


class SynthesisSchema(BaseModel):
    synthesis: str = ""
    key_themes: list[str] = Field(default_factory=list)
    sources_used: list[SourceRefSchema] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    confidence: str = "low"
    needs_more_research: bool = False
    follow_up_subtasks: list[SubtaskSchema] = Field(default_factory=list)


class ReferenceSchema(BaseModel):
    id: int = 0
    title: str = ""
    source: str = ""
    url: str = ""


class CitationSchema(BaseModel):
    analysis: str = ""
    cited_report: str = ""
    references: list[ReferenceSchema] = Field(default_factory=list)
    uncited_claims: list[str] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)


class JudgeSchema(BaseModel):
    analysis: str = ""
    factual_accuracy: float = 0.0
    citation_accuracy: float = 0.0
    completeness: float = 0.0
    source_quality: float = 0.0
    tool_efficiency: float = 0.0
    overall: float = 0.0
    pass_: bool = Field(default=False, alias="pass")
    reasoning: str = ""

    model_config = {"populate_by_name": True}


def validate_or_raw(schema: type[BaseModel], data: dict) -> dict:
    """Validate `data` against `schema`; on failure, return `data` unchanged.

    Returns the schema-normalized dict (with defaults filled in for any
    missing/malformed fields) on success. On a ValidationError — which with
    every field defaulted above only happens for a genuine type mismatch,
    e.g. a string where a list was expected — falls back to the raw dict so
    a single malformed field doesn't take down an otherwise-usable response.
    """
    try:
        validated = schema.model_validate(data)
    except ValidationError:
        return data
    return validated.model_dump(by_alias=True)
