"""Shared helpers for Day 19 research guidance generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from shared.schemas import UnifiedSearchResponse


GuidanceGenerator = Callable[[str], str]
MISSING_VALUES = {"", "not specified", "not stated in abstract", "not stated in retrieved evidence", "none", "n/a"}
LIMITATION_KEYWORDS = (
    "limitation",
    "limited",
    "challenge",
    "future work",
    "open problem",
    "fails",
    "failure",
    "bias",
    "robust",
    "generalization",
    "scalability",
    "latency",
    "cost",
    "safety",
    "reliability",
)
EVALUATION_KEYWORDS = ("dataset", "benchmark", "evaluation", "metric", "accuracy", "bleu", "truthfulqa", "halu", "wmt")
METHOD_KEYWORDS = ("method", "architecture", "framework", "approach", "training", "retrieval", "attention", "adapter", "lora")
RECENT_YEAR = 2023


class GuidanceError(RuntimeError):
    """Raised when Day 19 guidance cannot be parsed or validated."""


@dataclass
class GuidanceCandidate:
    """Normalized paper-level candidate assembled from paper and chunk retrieval results."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    publication_year: int | None = None
    citation_count: int = 0
    topic: str | None = None
    methodology: str = ""
    dataset_used: str = ""
    key_result: str = ""
    limitations: str = ""
    source_ids: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    score: float = 0.0
    has_paper_support: bool = False
    has_chunk_support: bool = False
    section_hints: list[str] = field(default_factory=list)

    def all_text(self) -> str:
        parts = [
            self.title,
            self.topic or "",
            self.methodology,
            self.dataset_used,
            self.key_result,
            self.limitations,
            " ".join(self.evidence_snippets),
            " ".join(self.section_hints),
        ]
        return " ".join(part for part in parts if part).lower()

    def has_real_limitation(self) -> bool:
        if is_known(self.limitations):
            return True
        text = self.all_text()
        return any(keyword in text for keyword in LIMITATION_KEYWORDS)

    def has_evaluation_signal(self) -> bool:
        if is_known(self.dataset_used):
            return True
        text = self.all_text()
        return any(keyword in text for keyword in EVALUATION_KEYWORDS)

    def has_method_signal(self) -> bool:
        if is_known(self.methodology):
            return True
        text = self.all_text()
        return any(keyword in text for keyword in METHOD_KEYWORDS)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def is_known(value: Any) -> bool:
    return clean_text(value).lower() not in MISSING_VALUES


def truncate(value: str, max_chars: int = 420) -> str:
    cleaned = clean_text(value)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def source_score(result: Any) -> float:
    for attr in ("blended_score", "rerank_score", "hybrid_score", "dense_score", "sparse_score"):
        value = getattr(result, attr, None)
        if value is not None:
            return max(0.0, min(1.0, float(value)))
    return 0.0


def source_id_for_result(result: Any, fallback_index: int) -> str:
    if chunk_id := getattr(result, "chunk_id", None):
        return f"chunk:{chunk_id}"
    if paper_id := getattr(result, "paper_id", None):
        return f"paper:{paper_id}"
    title = clean_text(getattr(result, "title", "")) or f"result-{fallback_index}"
    slug = "-".join(title.lower().split())[:48]
    return f"result:{slug or fallback_index}"


def paper_key(result: Any, fallback_index: int) -> str:
    if paper_id := getattr(result, "paper_id", None):
        return str(paper_id)
    title = clean_text(getattr(result, "title", ""))
    if title:
        return "title:" + title.lower()
    return f"result:{fallback_index}"


def evidence_text_from_result(result: Any) -> str:
    for attr in ("text", "key_result", "main_contribution", "abstract"):
        if text := clean_text(getattr(result, attr, None)):
            return truncate(text)
    return ""


def merge_unique(existing: list[str], values: list[str], limit: int | None = None) -> list[str]:
    seen = set(existing)
    merged = list(existing)
    for value in values:
        cleaned = clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            merged.append(cleaned)
            if limit is not None and len(merged) >= limit:
                break
    return merged


def normalize_candidates(response: UnifiedSearchResponse) -> list[GuidanceCandidate]:
    candidates: dict[str, GuidanceCandidate] = {}
    all_results = list(response.paper_results) + list(response.chunk_results)

    for index, result in enumerate(all_results, start=1):
        key = paper_key(result, index)
        source_id = source_id_for_result(result, index)
        title = clean_text(getattr(result, "title", None)) or "Untitled source"
        score = source_score(result)
        candidate = candidates.get(key)
        if candidate is None:
            candidate = GuidanceCandidate(paper_id=key, title=title)
            candidates[key] = candidate

        candidate.title = candidate.title or title
        candidate.authors = merge_unique(candidate.authors, list(getattr(result, "authors", []) or []), limit=8)
        candidate.publication_year = candidate.publication_year or getattr(result, "year", None)
        candidate.citation_count = max(candidate.citation_count, int(getattr(result, "citation_count", 0) or 0))
        candidate.topic = candidate.topic or getattr(result, "topic", None)
        candidate.methodology = candidate.methodology or clean_text(getattr(result, "methodology", None))
        candidate.dataset_used = candidate.dataset_used or clean_text(getattr(result, "dataset_used", None))
        candidate.key_result = candidate.key_result or clean_text(getattr(result, "key_result", None))
        candidate.limitations = candidate.limitations or clean_text(getattr(result, "limitations", None))
        candidate.source_ids = merge_unique(candidate.source_ids, [source_id], limit=8)
        snippet = evidence_text_from_result(result)
        candidate.evidence_snippets = merge_unique(candidate.evidence_snippets, [snippet], limit=4)
        candidate.score = max(candidate.score, score)
        candidate.has_chunk_support = candidate.has_chunk_support or bool(getattr(result, "chunk_id", None))
        candidate.has_paper_support = candidate.has_paper_support or not bool(getattr(result, "chunk_id", None))
        candidate.section_hints = merge_unique(candidate.section_hints, [getattr(result, "section_hint", None)], limit=6)

    return sorted(candidates.values(), key=lambda item: (item.score, item.citation_count, item.publication_year or 0), reverse=True)


def citation_norm(candidates: list[GuidanceCandidate], candidate: GuidanceCandidate) -> float:
    max_citations = max((item.citation_count for item in candidates), default=0)
    if max_citations <= 0:
        return 0.0
    return candidate.citation_count / max_citations


def recency_norm(candidate: GuidanceCandidate) -> float:
    if candidate.publication_year is None:
        return 0.0
    return max(0.0, min(1.0, (candidate.publication_year - 2017) / 9))


def coverage_score(candidate: GuidanceCandidate) -> float:
    fields = [
        is_known(candidate.methodology),
        is_known(candidate.dataset_used),
        is_known(candidate.key_result),
        candidate.has_chunk_support,
        bool(candidate.evidence_snippets),
        candidate.has_real_limitation(),
    ]
    return sum(1 for field_value in fields if field_value) / len(fields)


def guidance_score(candidates: list[GuidanceCandidate], candidate: GuidanceCandidate) -> float:
    score = (
        0.45 * candidate.score
        + 0.20 * citation_norm(candidates, candidate)
        + 0.15 * recency_norm(candidate)
        + 0.20 * coverage_score(candidate)
    )
    return round(max(0.0, min(1.0, score)), 6)


def parse_json_payload(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise GuidanceError("generator did not return valid JSON") from None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise GuidanceError(f"generator returned malformed JSON: {exc}") from exc


def call_with_json_retry(generator: GuidanceGenerator, prompt: str, *, repair_label: str) -> dict[str, Any]:
    first = generator(prompt)
    try:
        return parse_json_payload(first)
    except GuidanceError:
        repair_prompt = (
            f"The previous {repair_label} response was not valid JSON. "
            "Return only corrected JSON, preserve the same allowed IDs, and do not add new evidence.\n\n"
            f"Original prompt:\n{prompt}\n\nInvalid response:\n{first}"
        )
        second = generator(repair_prompt)
        return parse_json_payload(second)


def candidate_payload(candidate: GuidanceCandidate, candidates: list[GuidanceCandidate]) -> dict[str, Any]:
    return {
        "paper_id": candidate.paper_id,
        "title": candidate.title,
        "authors": candidate.authors,
        "publication_year": candidate.publication_year,
        "citation_count": candidate.citation_count,
        "topic": candidate.topic,
        "methodology": candidate.methodology,
        "dataset_used": candidate.dataset_used,
        "key_result": candidate.key_result,
        "limitations": candidate.limitations,
        "source_ids": candidate.source_ids,
        "evidence_snippets": candidate.evidence_snippets,
        "section_hints": candidate.section_hints,
        "relevance_score": guidance_score(candidates, candidate),
        "has_paper_support": candidate.has_paper_support,
        "has_chunk_support": candidate.has_chunk_support,
    }


def all_source_ids(candidates: list[GuidanceCandidate]) -> set[str]:
    return {source_id for candidate in candidates for source_id in candidate.source_ids}


def candidate_by_id(candidates: list[GuidanceCandidate]) -> dict[str, GuidanceCandidate]:
    return {candidate.paper_id: candidate for candidate in candidates}
