"""Generate a grounded reading path from unified retrieval results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from agent.guidance_common import (
    GuidanceCandidate,
    GuidanceError,
    GuidanceGenerator,
    all_source_ids,
    call_with_json_retry,
    candidate_by_id,
    candidate_payload,
    clean_text,
    guidance_score,
    is_known,
    normalize_candidates,
)
from retrieval.confidence import assess_confidence, load_response
from retrieval.index_qdrant import load_env_file
from retrieval.unified_search import run_unified_search
from shared.schemas import ConfidenceAssessment, ReadingPath, ReadingPathItem, ReadingPathStage, UnifiedSearchResponse


DEFAULT_READING_MODEL = "gpt-4o-mini"
DEFAULT_MAX_PAPERS = 8
STAGE_ORDER = [
    "foundational",
    "core_methods",
    "evaluation_and_benchmarks",
    "recent_advances",
    "limitations_and_open_problems",
]
STAGE_DESCRIPTIONS = {
    "foundational": "Start with high-signal papers that define the area or establish core ideas.",
    "core_methods": "Then read method papers that explain the main technical approaches.",
    "evaluation_and_benchmarks": "Next inspect datasets, metrics, and benchmark evidence.",
    "recent_advances": "Then move to newer work that extends or challenges earlier methods.",
    "limitations_and_open_problems": "Finish with papers that surface limitations, gaps, or unresolved issues.",
}


class ReadingPathError(GuidanceError):
    """Raised when a reading path cannot be generated or validated."""


def candidate_method_key(candidate: GuidanceCandidate) -> str:
    if is_known(candidate.methodology):
        return clean_text(candidate.methodology).lower()[:80]
    return ""


def citation_threshold(candidates: list[GuidanceCandidate]) -> int:
    citations = sorted((candidate.citation_count for candidate in candidates), reverse=True)
    if not citations:
        return 0
    index = min(len(citations) - 1, max(0, len(citations) // 3))
    return citations[index]


def stage_pool(stage: str, candidates: list[GuidanceCandidate]) -> list[GuidanceCandidate]:
    threshold = citation_threshold(candidates)
    if stage == "foundational":
        pool = [item for item in candidates if item.citation_count >= threshold or (item.publication_year or 9999) <= 2020]
        return sorted(pool, key=lambda item: (item.citation_count, item.score), reverse=True)
    if stage == "core_methods":
        has_recent_pool = any(item.publication_year is not None and item.publication_year >= 2023 for item in candidates)
        pool = [
            item
            for item in candidates
            if item.has_method_signal() and not (has_recent_pool and item.publication_year is not None and item.publication_year >= 2023)
        ]
        return sorted(pool, key=lambda item: (guidance_score(candidates, item), item.citation_count), reverse=True)
    if stage == "evaluation_and_benchmarks":
        has_recent_pool = any(item.publication_year is not None and item.publication_year >= 2023 for item in candidates)
        pool = [
            item
            for item in candidates
            if item.has_evaluation_signal() and not (has_recent_pool and item.publication_year is not None and item.publication_year >= 2023)
        ]
        return sorted(pool, key=lambda item: (guidance_score(candidates, item), item.has_chunk_support), reverse=True)
    if stage == "recent_advances":
        pool = [item for item in candidates if item.publication_year is not None and item.publication_year >= 2023]
        return sorted(pool, key=lambda item: (item.publication_year or 0, guidance_score(candidates, item)), reverse=True)
    if stage == "limitations_and_open_problems":
        pool = [item for item in candidates if item.has_real_limitation()]
        return sorted(pool, key=lambda item: (item.has_chunk_support, guidance_score(candidates, item)), reverse=True)
    return []


def pick_diverse(
    pool: list[GuidanceCandidate],
    *,
    selected_ids: set[str],
    used_methods: set[str],
    limit: int,
) -> list[GuidanceCandidate]:
    picked: list[GuidanceCandidate] = []
    for candidate in pool:
        method_key = candidate_method_key(candidate)
        if candidate.paper_id in selected_ids:
            continue
        if method_key and method_key in used_methods:
            continue
        picked.append(candidate)
        selected_ids.add(candidate.paper_id)
        if method_key:
            used_methods.add(method_key)
        if len(picked) >= limit:
            return picked

    for candidate in pool:
        if candidate.paper_id in selected_ids:
            continue
        picked.append(candidate)
        selected_ids.add(candidate.paper_id)
        method_key = candidate_method_key(candidate)
        if method_key:
            used_methods.add(method_key)
        if len(picked) >= limit:
            break
    return picked


def select_reading_candidates(
    response: UnifiedSearchResponse,
    *,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> dict[str, list[GuidanceCandidate]]:
    candidates = normalize_candidates(response)
    if max_papers <= 0:
        raise ReadingPathError("max_papers must be greater than 0")
    if not candidates:
        return {}

    quotas = {
        "foundational": 2,
        "core_methods": 2,
        "evaluation_and_benchmarks": 2,
        "recent_advances": 2,
        "limitations_and_open_problems": 1,
    }
    selected_ids: set[str] = set()
    used_methods: set[str] = set()
    selected: dict[str, list[GuidanceCandidate]] = {}

    for stage in STAGE_ORDER:
        remaining = max_papers - sum(len(items) for items in selected.values())
        if remaining <= 0:
            break
        target = min(quotas[stage], remaining)
        picked = pick_diverse(stage_pool(stage, candidates), selected_ids=selected_ids, used_methods=used_methods, limit=target)
        if picked:
            selected[stage] = picked

    remaining = max_papers - sum(len(items) for items in selected.values())
    if remaining > 0:
        fallback_pool = sorted(candidates, key=lambda item: guidance_score(candidates, item), reverse=True)
        extra = pick_diverse(fallback_pool, selected_ids=selected_ids, used_methods=used_methods, limit=remaining)
        if extra:
            selected.setdefault("core_methods", []).extend(extra)

    return {stage: selected[stage] for stage in STAGE_ORDER if selected.get(stage)}


def selected_candidates_flat(selected: dict[str, list[GuidanceCandidate]]) -> list[GuidanceCandidate]:
    return [candidate for stage in STAGE_ORDER for candidate in selected.get(stage, [])]


READING_PATH_PROMPT_TEMPLATE = """You are generating a grounded reading path for a research synthesis engine.

User question:
{question}

Allowed stages, in order:
{stage_order}

Use only the candidate papers below. Preserve exact paper_id and source_ids. Do not introduce outside papers or unsupported claims. Explain why the user should read each paper in this learning sequence. Mention prerequisites only when supported by the candidate metadata or as very lightweight reading guidance.

Candidate papers grouped by deterministic stage:
{candidate_json}

Return only valid JSON with this exact shape:
{{
  "stages": [
    {{
      "stage": "foundational",
      "description": "why this stage matters for the question",
      "papers": [
        {{
          "paper_id": "exact candidate paper_id",
          "reason_to_read": "grounded reason",
          "focus_points": ["what to focus on"],
          "prerequisites": ["optional prerequisite"],
          "connection_to_next": "how this prepares the next paper or stage",
          "source_ids": ["source:id"]
        }}
      ]
    }}
  ],
  "limitations": ["corpus or evidence limitation"]
}}
"""


def build_reading_path_prompt(question: str, selected: dict[str, list[GuidanceCandidate]]) -> str:
    all_candidates = selected_candidates_flat(selected)
    payload = {
        stage: [candidate_payload(candidate, all_candidates) for candidate in candidates]
        for stage, candidates in selected.items()
    }
    return READING_PATH_PROMPT_TEMPLATE.format(
        question=question,
        stage_order=", ".join(STAGE_ORDER),
        candidate_json=json.dumps(payload, indent=2),
    )


def build_guarded_reading_path(question: str, confidence: ConfidenceAssessment, reason: str | None = None) -> ReadingPath:
    message = reason or confidence.reason
    return ReadingPath(
        question=question,
        stages=[],
        total_papers=0,
        confidence_decision=confidence.decision,
        limitations=[message, confidence.recommended_action],
    )


def validate_reading_path_payload(
    question: str,
    selected: dict[str, list[GuidanceCandidate]],
    payload: dict[str, Any],
    confidence: ConfidenceAssessment,
    *,
    max_papers: int,
) -> ReadingPath:
    candidates = selected_candidates_flat(selected)
    candidate_map = candidate_by_id(candidates)
    valid_source_ids = all_source_ids(candidates)
    seen_papers: set[str] = set()
    stages: list[ReadingPathStage] = []
    order = 1

    for stage_payload in payload.get("stages", []):
        stage = stage_payload.get("stage")
        if stage not in STAGE_ORDER:
            raise ReadingPathError(f"invalid reading path stage: {stage}")
        papers: list[ReadingPathItem] = []
        for item_payload in stage_payload.get("papers", []):
            if order > max_papers:
                break
            paper_id = item_payload.get("paper_id")
            if paper_id not in candidate_map:
                raise ReadingPathError(f"reading path referenced unknown paper_id: {paper_id}")
            if paper_id in seen_papers:
                raise ReadingPathError(f"reading path duplicated paper_id: {paper_id}")
            candidate = candidate_map[paper_id]
            source_ids = item_payload.get("source_ids") or candidate.source_ids
            invalid_sources = [source_id for source_id in source_ids if source_id not in valid_source_ids]
            if invalid_sources:
                raise ReadingPathError(f"reading path referenced unknown source_ids: {invalid_sources}")
            seen_papers.add(paper_id)
            papers.append(
                ReadingPathItem(
                    order=order,
                    stage=stage,
                    paper_id=paper_id,
                    title=candidate.title,
                    authors=candidate.authors,
                    publication_year=candidate.publication_year,
                    citation_count=candidate.citation_count,
                    topic=candidate.topic,
                    reason_to_read=item_payload.get("reason_to_read") or f"Retrieved as relevant evidence for {question}.",
                    focus_points=list(item_payload.get("focus_points") or []),
                    prerequisites=list(item_payload.get("prerequisites") or []),
                    connection_to_next=item_payload.get("connection_to_next"),
                    source_ids=list(source_ids),
                    evidence_snippet=candidate.evidence_snippets[0] if candidate.evidence_snippets else None,
                    relevance_score=guidance_score(candidates, candidate),
                )
            )
            order += 1
        if papers:
            stages.append(
                ReadingPathStage(
                    stage=stage,
                    description=stage_payload.get("description") or STAGE_DESCRIPTIONS[stage],
                    papers=papers,
                )
            )
        if order > max_papers:
            break

    limitations = list(payload.get("limitations") or [])
    if len(seen_papers) < sum(len(items) for items in selected.values()):
        limitations.append("Some retrieved candidates were omitted to respect the maximum reading-path length.")
    if not stages:
        limitations.append("No valid reading path items were returned after grounding validation.")
    return ReadingPath(
        question=question,
        stages=stages,
        total_papers=sum(len(stage.papers) for stage in stages),
        confidence_decision=confidence.decision,
        limitations=limitations,
    )


def call_openai_generator(prompt: str, *, model: str = DEFAULT_READING_MODEL, env_file: Path = Path(".env")) -> str:
    load_env_file(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ReadingPathError("OPENAI_API_KEY is missing. Add it to .env before live reading path generation.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON for a grounded research reading path."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ReadingPathError("OpenAI returned an empty reading path")
    return content


def build_reading_path(
    response: UnifiedSearchResponse,
    *,
    confidence: ConfidenceAssessment | None = None,
    generator: GuidanceGenerator | None = None,
    model: str = DEFAULT_READING_MODEL,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> ReadingPath:
    confidence = confidence or assess_confidence(response)
    if confidence.decision != "sufficient_evidence":
        return build_guarded_reading_path(response.query, confidence)

    selected = select_reading_candidates(response, max_papers=max_papers)
    if not selected:
        return build_guarded_reading_path(response.query, confidence, "No retrieved papers or chunks were available for a reading path.")

    prompt = build_reading_path_prompt(response.query, selected)
    active_generator = generator or (lambda text: call_openai_generator(text, model=model))
    payload = call_with_json_retry(active_generator, prompt, repair_label="reading path")
    try:
        return validate_reading_path_payload(response.query, selected, payload, confidence, max_papers=max_papers)
    except (TypeError, ValidationError) as exc:
        raise ReadingPathError(f"reading path failed schema validation: {exc}") from exc


def reading_path_to_text(path: ReadingPath) -> str:
    if not path.stages:
        return "No confident reading path generated. " + " ".join(path.limitations)
    lines = [f"Reading path for: {path.question}", f"Total papers: {path.total_papers}"]
    for stage in path.stages:
        lines.append(f"\n{stage.stage}: {stage.description}")
        for item in stage.papers:
            year = item.publication_year or "unknown year"
            lines.append(f"{item.order}. {item.title} ({year})")
            lines.append(f"   Why: {item.reason_to_read}")
            if item.source_ids:
                lines.append(f"   Sources: {', '.join(item.source_ids)}")
    if path.limitations:
        lines.append("\nLimitations:")
        lines.extend(f"- {limitation}" for limitation in path.limitations)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run unified retrieval for this question before building the path.")
    parser.add_argument("--query", dest="query_option", help="Run unified retrieval for this question before building the path.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a saved UnifiedSearchResponse JSON file.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_MAX_PAPERS)
    parser.add_argument("--model", default=DEFAULT_READING_MODEL)
    parser.add_argument("--readable", action="store_true", help="Print a readable text version instead of JSON.")
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = args.query_option or args.query
    if args.input:
        response = load_response(args.input)
    elif query:
        response = run_unified_search(query, top_k=args.top_k, apply_reranking=not args.no_rerank)
    else:
        raise ReadingPathError("provide either --query, a positional query, or --input path")

    path = build_reading_path(response, max_papers=args.top_k, model=args.model)
    print(reading_path_to_text(path) if args.readable else path.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except ReadingPathError as exc:
        raise SystemExit(f"Error: {exc}") from None
