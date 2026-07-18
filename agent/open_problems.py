"""Generate grounded open-problems reports from unified retrieval results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from agent.guidance_common import (
    EVALUATION_KEYWORDS,
    LIMITATION_KEYWORDS,
    GuidanceCandidate,
    GuidanceError,
    GuidanceGenerator,
    all_source_ids,
    call_with_json_retry,
    candidate_by_id,
    candidate_payload,
    clean_text,
    normalize_candidates,
    truncate,
)
from retrieval.confidence import assess_confidence, load_response
from retrieval.index_qdrant import load_env_file
from retrieval.unified_search import run_unified_search
from shared.schemas import ConfidenceAssessment, OpenProblem, OpenProblemsReport, UnifiedSearchResponse


DEFAULT_OPEN_PROBLEMS_MODEL = "gpt-4o-mini"
DEFAULT_MAX_PROBLEMS = 6
VALID_CATEGORIES = {
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
}
CATEGORY_KEYWORDS = {
    "evaluation": ("evaluation", "metric", "benchmark", "benchmark coverage", "measure"),
    "data": ("dataset", "data", "annotation"),
    "methodology": ("method", "architecture", "approach", "training"),
    "generalization": ("generalization", "domain", "robust", "transfer"),
    "scalability": ("scalability", "scale", "large-scale"),
    "efficiency": ("efficiency", "cost", "latency", "compute"),
    "safety": ("safety", "reliability", "hallucination", "risk"),
    "interpretability": ("interpretability", "explain", "transparent"),
    "reproducibility": ("reproducibility", "replicate", "open source"),
    "deployment": ("deployment", "real-world", "production", "system"),
}
CONFLICT_KEYWORDS = ("however", "whereas", "contrary", "conflict", "inconsistent", "trade-off", "tradeoff")


class OpenProblemsError(GuidanceError):
    """Raised when an open-problems report cannot be generated or validated."""


def infer_category(text: str) -> str:
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "methodology"


def limitation_text(candidate: GuidanceCandidate) -> str:
    if clean_text(candidate.limitations).lower() not in {"", "not specified", "not stated in abstract", "not stated in retrieved evidence"}:
        return truncate(candidate.limitations)
    snippets = []
    for snippet in candidate.evidence_snippets:
        lowered = snippet.lower()
        if any(keyword in lowered for keyword in LIMITATION_KEYWORDS + EVALUATION_KEYWORDS):
            snippets.append(snippet)
    return truncate(" ".join(snippets), 520)


def extract_problem_evidence(candidates: list[GuidanceCandidate]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for candidate in candidates:
        text = limitation_text(candidate)
        if not text:
            continue
        key = text.lower()[:180]
        if key in seen_texts:
            continue
        seen_texts.add(key)
        evidence.append(
            {
                "paper_id": candidate.paper_id,
                "title": candidate.title,
                "topic": candidate.topic,
                "publication_year": candidate.publication_year,
                "citation_count": candidate.citation_count,
                "category_hint": infer_category(text),
                "source_ids": candidate.source_ids,
                "evidence_snippets": [text, *candidate.evidence_snippets[:2]],
                "limitation": text,
                "has_chunk_support": candidate.has_chunk_support,
            }
        )
    return evidence


def detect_conflicting_findings(candidates: list[GuidanceCandidate]) -> list[str]:
    conflicts: list[str] = []
    for candidate in candidates:
        for snippet in candidate.evidence_snippets:
            lowered = snippet.lower()
            if any(keyword in lowered for keyword in CONFLICT_KEYWORDS):
                conflicts.append(f"{candidate.title}: {truncate(snippet, 220)}")
                break
    return conflicts[:5]


def recurring_limitations_from_evidence(evidence: list[dict[str, Any]]) -> list[str]:
    buckets: dict[str, set[str]] = {}
    examples: dict[str, str] = {}
    for item in evidence:
        category = item["category_hint"]
        buckets.setdefault(category, set()).add(item["paper_id"])
        examples.setdefault(category, item["limitation"])
    recurring = []
    for category, paper_ids in buckets.items():
        if len(paper_ids) >= 2:
            recurring.append(f"{category}: {examples[category]}")
    return recurring[:6]


def evidence_strength(paper_ids: list[str], source_ids: list[str]) -> str:
    distinct_papers = len(set(paper_ids))
    distinct_sources = len(set(source_ids))
    if distinct_papers >= 2 and distinct_sources >= 2:
        return "strong"
    if distinct_papers >= 2 or distinct_sources >= 2:
        return "moderate"
    return "weak"


OPEN_PROBLEMS_PROMPT_TEMPLATE = """You are generating grounded open research problems for a research synthesis engine.

User question:
{question}

Use only the limitation and evidence records below. Preserve exact paper_id and source_ids. Do not introduce outside problems. Do not claim a problem is universally unresolved; describe it as unresolved within the retrieved evidence. Merge repeated limitations into one problem.

Valid categories:
{categories}

Evidence records:
{evidence_json}

Return only valid JSON with this exact shape:
{{
  "problems": [
    {{
      "title": "specific open problem",
      "description": "grounded description",
      "category": "evaluation",
      "why_it_matters": "why this matters for the user question",
      "evidence_summary": "short summary of the retrieved evidence",
      "supporting_paper_ids": ["exact paper_id"],
      "supporting_source_ids": ["source:id"],
      "evidence_snippets": ["short copied or paraphrased snippet from evidence records"],
      "suggested_research_directions": ["grounded next direction"],
      "confidence": 0.0
    }}
  ],
  "recurring_limitations": ["repeated limitation across records"],
  "conflicting_findings": ["conflict supported by evidence"],
  "evidence_gaps": ["what retrieved evidence is missing"],
  "corpus_limitations": ["limitations of the current corpus"]
}}
"""


def build_open_problems_prompt(question: str, evidence: list[dict[str, Any]]) -> str:
    return OPEN_PROBLEMS_PROMPT_TEMPLATE.format(
        question=question,
        categories=", ".join(sorted(VALID_CATEGORIES)),
        evidence_json=json.dumps(evidence, indent=2),
    )


def build_guarded_open_problems(question: str, confidence: ConfidenceAssessment, reason: str | None = None) -> OpenProblemsReport:
    message = reason or confidence.reason
    return OpenProblemsReport(
        question=question,
        problems=[],
        recurring_limitations=[],
        conflicting_findings=[],
        evidence_gaps=[confidence.recommended_action],
        corpus_limitations=[message],
        confidence_decision=confidence.decision,
    )


def validate_problem_payload(
    question: str,
    candidates: list[GuidanceCandidate],
    payload: dict[str, Any],
    confidence: ConfidenceAssessment,
    *,
    max_problems: int,
) -> OpenProblemsReport:
    candidate_map = candidate_by_id(candidates)
    valid_source_ids = all_source_ids(candidates)
    problems: list[OpenProblem] = []
    seen_titles: set[str] = set()

    for problem_payload in payload.get("problems", []):
        paper_ids = list(problem_payload.get("supporting_paper_ids") or [])
        source_ids = list(problem_payload.get("supporting_source_ids") or [])
        if not paper_ids or not source_ids:
            continue
        unknown_papers = [paper_id for paper_id in paper_ids if paper_id not in candidate_map]
        unknown_sources = [source_id for source_id in source_ids if source_id not in valid_source_ids]
        if unknown_papers:
            raise OpenProblemsError(f"open problem referenced unknown paper_ids: {unknown_papers}")
        if unknown_sources:
            raise OpenProblemsError(f"open problem referenced unknown source_ids: {unknown_sources}")
        title = clean_text(problem_payload.get("title"))
        if not title or title.lower() in seen_titles:
            continue
        category = problem_payload.get("category") or infer_category(title)
        if category not in VALID_CATEGORIES:
            category = infer_category(" ".join([title, problem_payload.get("description", "")]))
        snippets = [truncate(snippet, 260) for snippet in problem_payload.get("evidence_snippets", []) if clean_text(snippet)]
        if not snippets:
            snippets = [snippet for paper_id in paper_ids for snippet in candidate_map[paper_id].evidence_snippets[:1]]
        seen_titles.add(title.lower())
        problems.append(
            OpenProblem(
                title=title,
                description=problem_payload.get("description") or title,
                category=category,
                why_it_matters=problem_payload.get("why_it_matters") or "This matters because it affects how confidently the retrieved findings can be used.",
                evidence_summary=problem_payload.get("evidence_summary") or snippets[0],
                supporting_paper_ids=paper_ids,
                supporting_source_ids=source_ids,
                evidence_snippets=snippets[:4],
                evidence_strength=evidence_strength(paper_ids, source_ids),
                suggested_research_directions=list(problem_payload.get("suggested_research_directions") or []),
                confidence=problem_payload.get("confidence"),
            )
        )
        if len(problems) >= max_problems:
            break

    evidence = extract_problem_evidence(candidates)
    recurring = list(payload.get("recurring_limitations") or recurring_limitations_from_evidence(evidence))
    conflicts = list(payload.get("conflicting_findings") or detect_conflicting_findings(candidates))
    gaps = list(payload.get("evidence_gaps") or [])
    corpus_limits = list(payload.get("corpus_limitations") or [])
    if not problems:
        corpus_limits.append("No validated open problems were supported by the retrieved evidence.")
    return OpenProblemsReport(
        question=question,
        problems=problems,
        recurring_limitations=recurring,
        conflicting_findings=conflicts,
        evidence_gaps=gaps,
        corpus_limitations=corpus_limits,
        confidence_decision=confidence.decision,
    )


def call_openai_generator(prompt: str, *, model: str = DEFAULT_OPEN_PROBLEMS_MODEL, env_file: Path = Path(".env")) -> str:
    load_env_file(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenProblemsError("OPENAI_API_KEY is missing. Add it to .env before live open-problems generation.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON for grounded research open problems."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise OpenProblemsError("OpenAI returned empty open-problems output")
    return content


def build_open_problems_report(
    response: UnifiedSearchResponse,
    *,
    confidence: ConfidenceAssessment | None = None,
    generator: GuidanceGenerator | None = None,
    model: str = DEFAULT_OPEN_PROBLEMS_MODEL,
    max_problems: int = DEFAULT_MAX_PROBLEMS,
) -> OpenProblemsReport:
    confidence = confidence or assess_confidence(response)
    if confidence.decision != "sufficient_evidence":
        return build_guarded_open_problems(response.query, confidence)

    candidates = normalize_candidates(response)
    evidence = extract_problem_evidence(candidates)
    if not evidence:
        return build_guarded_open_problems(
            response.query,
            confidence,
            "Retrieved evidence did not contain clear limitations, future-work, or evaluation-gap signals.",
        )

    prompt = build_open_problems_prompt(response.query, evidence)
    active_generator = generator or (lambda text: call_openai_generator(text, model=model))
    payload = call_with_json_retry(active_generator, prompt, repair_label="open problems")
    try:
        return validate_problem_payload(response.query, candidates, payload, confidence, max_problems=max_problems)
    except (TypeError, ValidationError) as exc:
        raise OpenProblemsError(f"open-problems payload failed schema validation: {exc}") from exc


def open_problems_to_text(report: OpenProblemsReport) -> str:
    if not report.problems:
        parts = ["No confident open-problems report generated."]
        parts.extend(report.corpus_limitations)
        parts.extend(report.evidence_gaps)
        return " ".join(parts)
    lines = [f"Open problems for: {report.question}"]
    for index, problem in enumerate(report.problems, start=1):
        lines.append(f"\n{index}. {problem.title} [{problem.category}, {problem.evidence_strength}]")
        lines.append(f"   Why it matters: {problem.why_it_matters}")
        lines.append(f"   Evidence: {problem.evidence_summary}")
        lines.append(f"   Sources: {', '.join(problem.supporting_source_ids)}")
    if report.recurring_limitations:
        lines.append("\nRecurring limitations:")
        lines.extend(f"- {item}" for item in report.recurring_limitations)
    if report.evidence_gaps:
        lines.append("\nEvidence gaps:")
        lines.extend(f"- {item}" for item in report.evidence_gaps)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run unified retrieval for this question before finding open problems.")
    parser.add_argument("--query", dest="query_option", help="Run unified retrieval for this question before finding open problems.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a saved UnifiedSearchResponse JSON file.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_MAX_PROBLEMS)
    parser.add_argument("--model", default=DEFAULT_OPEN_PROBLEMS_MODEL)
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
        raise OpenProblemsError("provide either --query, a positional query, or --input path")

    report = build_open_problems_report(response, max_problems=args.top_k, model=args.model)
    print(open_problems_to_text(report) if args.readable else report.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except OpenProblemsError as exc:
        raise SystemExit(f"Error: {exc}") from None
