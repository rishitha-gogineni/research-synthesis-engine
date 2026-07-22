"""Generate grounded research briefs from unified retrieval results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from pydantic import ValidationError

from retrieval.confidence import assess_confidence, load_response
from retrieval.index_qdrant import load_env_file
from retrieval.unified_search import run_unified_search
from shared.schemas import (
    BriefTheme,
    ConfidenceAssessment,
    EvidenceSource,
    ResearchBrief,
    UnifiedSearchResponse,
)


DEFAULT_SYNTHESIS_MODEL = "gpt-4o-mini"
MAX_SOURCE_TEXT_CHARS = 900
MAX_SOURCES = 12
BriefGenerator = Callable[[str], str]


class SynthesisError(RuntimeError):
    """Raised when a grounded brief cannot be generated or parsed."""


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())


def source_score(result: Any) -> float:
    for attr in ("blended_score", "rerank_score", "hybrid_score", "dense_score", "sparse_score"):
        value = getattr(result, attr, None)
        if value is not None:
            return max(0.0, min(1.0, float(value)))
    return 0.0


def source_text_from_result(result: Any) -> str:
    if text := clean_text(getattr(result, "text", None)):
        return text[:MAX_SOURCE_TEXT_CHARS]

    parts = [
        clean_text(getattr(result, "main_contribution", None)),
        clean_text(getattr(result, "methodology", None)),
        clean_text(getattr(result, "dataset_used", None)),
        clean_text(getattr(result, "key_result", None)),
        clean_text(getattr(result, "limitations", None)),
        clean_text(getattr(result, "abstract", None)),
    ]
    joined = " ".join(part for part in parts if part)
    return joined[:MAX_SOURCE_TEXT_CHARS]


def evidence_source_from_result(result: Any, fallback_index: int) -> EvidenceSource | None:
    evidence_text = source_text_from_result(result)
    if not evidence_text:
        return None

    paper_id = getattr(result, "paper_id", None)
    chunk_id = getattr(result, "chunk_id", None)
    if chunk_id:
        source_id = f"chunk:{chunk_id}"
    elif paper_id:
        source_id = f"paper:{paper_id}"
    else:
        source_id = f"result:{fallback_index}"

    return EvidenceSource(
        source_id=source_id,
        title=getattr(result, "title", "Untitled source"),
        topic=getattr(result, "topic", "Unknown"),
        paper_id=paper_id,
        chunk_id=chunk_id,
        year=getattr(result, "year", None),
        citation_count=getattr(result, "citation_count", 0) or 0,
        evidence_text=evidence_text,
        score=round(source_score(result), 6),
    )


def collect_evidence_sources(response: UnifiedSearchResponse, max_sources: int = MAX_SOURCES) -> list[EvidenceSource]:
    candidates = list(response.paper_results) + list(response.chunk_results)
    ranked = sorted(candidates, key=source_score, reverse=True)

    sources: list[EvidenceSource] = []
    seen: set[str] = set()
    for index, result in enumerate(ranked, start=1):
        source = evidence_source_from_result(result, index)
        if source is None or source.source_id in seen:
            continue
        seen.add(source.source_id)
        sources.append(source)
        if len(sources) >= max_sources:
            break
    return sources


def build_synthesis_prompt(query: str, sources: list[EvidenceSource]) -> str:
    source_blocks = []
    for source in sources:
        source_blocks.append(
            "\n".join(
                [
                    f"SOURCE_ID: {source.source_id}",
                    f"TITLE: {source.title}",
                    f"TOPIC: {source.topic}",
                    f"YEAR: {source.year or 'unknown'}",
                    f"CITATIONS: {source.citation_count}",
                    f"EVIDENCE: {source.evidence_text}",
                ]
            )
        )

    joined_sources = "\n\n".join(source_blocks)
    return f"""You are generating a grounded research brief for a research synthesis engine.

User question:
{query}

Use only the retrieved sources below. Do not add outside facts. If the retrieved evidence does not support a point, say so in limitations or open_problems. Cite source IDs exactly as provided.

Retrieved sources:
{joined_sources}

Write for a student or research analyst who needs a useful synthesis, not just a list of papers.

Direct-answer requirements:
- Answer the exact user question in 2-3 concise paragraphs.
- First paragraph: give the plain-language conceptual answer before naming papers or methods.
- Second paragraph: connect that concept to the strongest retrieved evidence using source IDs.
- Every direct-answer paragraph should include at least one exact SOURCE_ID when evidence supports the claim.
- Optional third paragraph: add nuance, boundary conditions, or what the evidence does not establish.
- For comparison or contrast questions, define both sides, state the key difference, and give one concrete example.
- For agent/task questions, explicitly address planning, tool/API use, action execution, observation/feedback, and workflow completion when supported by evidence.
- When contrasting with chatbots, say "a plain chatbot without tool access mainly generates responses" rather than claiming all ChatGPT-like systems only answer.
- Prefer broad survey evidence and high-citation sources when multiple sources support the same point.
- Ground every substantive claim in the retrieved evidence.
- Do not mention unsupported statistics, datasets, or paper findings.
- If the evidence is partial, say what is and is not established.

Theme requirements:
- Return 3-5 named research themes when the evidence supports them.
- Each theme summary must be one sentence and cite supporting source IDs.

Evidence requirements:
- Evidence bullets should be specific claim/source statements, not generic summaries.
- Keep source IDs exactly as provided.

Return only valid JSON with this exact shape:
{{
  "direct_answer": "2-3 paragraph answer grounded in the sources",
  "themes": [
    {{
      "theme": "short theme name",
      "summary": "one-sentence explanation of what the sources collectively say about this theme",
      "supporting_source_ids": ["source:id"]
    }}
  ],
  "evidence_bullets": ["specific evidence point with source IDs"],
  "limitations": ["what the retrieved evidence does not establish"],
  "open_problems": ["research gap or follow-up question grounded in the evidence"]
}}
"""


def parse_brief_payload(raw_text: str) -> dict[str, Any]:
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
            raise SynthesisError("brief generator did not return valid JSON") from None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SynthesisError(f"brief generator returned malformed JSON: {exc}") from exc


def ensure_direct_answer_citations(answer: str, sources: list[EvidenceSource]) -> str:
    cleaned = clean_text(answer)
    if not cleaned or not sources:
        return answer
    known_ids = [source.source_id for source in sources if source.source_id]
    if any(source_id in cleaned for source_id in known_ids):
        return answer
    cited = "; ".join(known_ids[:2])
    return f"{answer.rstrip()} Sources: {cited}."


def build_guarded_brief(
    query: str,
    confidence: ConfidenceAssessment,
    sources: list[EvidenceSource],
) -> ResearchBrief:
    direct_answer = (
        "I cannot answer this question reliably from the indexed research corpus yet. "
        f"The evidence gate returned `{confidence.decision}` because {confidence.reason.lower()} "
        "No synthesis was generated, so the response does not invent claims beyond the retrieved evidence."
    )
    return ResearchBrief(
        query=query,
        status="skipped_low_confidence",
        confidence_decision=confidence.decision,
        direct_answer=direct_answer,
        themes=[],
        evidence_bullets=[],
        limitations=[confidence.reason],
        open_problems=[confidence.recommended_action],
        sources=sources,
        warning="Answer generation skipped because retrieved evidence did not pass the synthesis confidence threshold.",
    )


def build_research_brief(
    response: UnifiedSearchResponse,
    *,
    confidence: ConfidenceAssessment | None = None,
    generator: BriefGenerator | None = None,
    model: str = DEFAULT_SYNTHESIS_MODEL,
) -> ResearchBrief:
    confidence = confidence or assess_confidence(response)
    sources = collect_evidence_sources(response)

    if confidence.decision != "sufficient_evidence":
        return build_guarded_brief(response.query, confidence, sources)

    if not sources:
        raise SynthesisError("cannot generate a research brief without evidence sources")

    prompt = build_synthesis_prompt(response.query, sources)
    raw_text = generator(prompt) if generator else call_openai_generator(prompt, model=model)
    payload = parse_brief_payload(raw_text)

    try:
        themes = [BriefTheme(**item) for item in payload.get("themes", [])]
        return ResearchBrief(
            query=response.query,
            status="generated",
            confidence_decision=confidence.decision,
            direct_answer=ensure_direct_answer_citations(payload.get("direct_answer", ""), sources),
            themes=themes,
            evidence_bullets=list(payload.get("evidence_bullets", [])),
            limitations=list(payload.get("limitations", [])),
            open_problems=list(payload.get("open_problems", [])),
            sources=sources,
        )
    except (TypeError, ValidationError) as exc:
        raise SynthesisError(f"brief payload failed schema validation: {exc}") from exc


def call_openai_generator(
    prompt: str,
    *,
    model: str = DEFAULT_SYNTHESIS_MODEL,
    env_file: Path = Path(".env"),
) -> str:
    load_env_file(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SynthesisError("OPENAI_API_KEY is missing. Add it to .env before live brief generation.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON for a grounded research brief."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise SynthesisError("OpenAI returned an empty brief")
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Run unified retrieval for this query before generating a brief.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a saved UnifiedSearchResponse JSON file.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_SYNTHESIS_MODEL)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        response = load_response(args.input)
    elif args.query:
        response = run_unified_search(args.query, top_k=args.top_k, apply_reranking=not args.no_rerank)
    else:
        raise SynthesisError("provide either a query or --input path")

    if args.env_file != Path(".env"):
        generator = lambda prompt: call_openai_generator(prompt, model=args.model, env_file=args.env_file)
        brief = build_research_brief(response, generator=generator, model=args.model)
    else:
        brief = build_research_brief(response, model=args.model)
    print(brief.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except SynthesisError as exc:
        raise SystemExit(f"Error: {exc}") from None
