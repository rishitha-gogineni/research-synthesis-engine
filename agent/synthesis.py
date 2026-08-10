"""Generate grounded research briefs from unified retrieval results."""

from __future__ import annotations

import argparse
import json
import os
import re
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
MMR_LAMBDA = 0.72
MMR_CANDIDATE_POOL_MULTIPLIER = 3
SOURCE_REFERENCE_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])(?:paper|chunk|result):[A-Za-z0-9_:/-]+(?:\.[A-Za-z0-9_:/-]+)*")
BriefGenerator = Callable[[str], str]


class SynthesisError(RuntimeError):
    """Raised when a grounded brief cannot be generated or parsed."""


def normalized_terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def compress_text_for_query(text: str, query: str, *, budget_chars: int) -> str:
    """Keep query-relevant sentences instead of blindly truncating long evidence."""
    if budget_chars <= 0:
        return ""
    if len(text) <= budget_chars:
        return text

    sentences = split_into_sentences(text)
    query_terms = normalized_terms(query)
    if len(sentences) <= 1 or not query_terms:
        return text[:budget_chars]

    scored = [
        (len(normalized_terms(sentence) & query_terms), index)
        for index, sentence in enumerate(sentences)
    ]
    if not any(score > 0 for score, _ in scored):
        return text[:budget_chars]

    selected: set[int] = set()
    used_chars = 0
    for score, index in sorted(scored, key=lambda item: item[0], reverse=True):
        if score <= 0:
            break
        sentence = sentences[index]
        added_chars = len(sentence) + (1 if selected else 0)
        if selected and used_chars + added_chars > budget_chars:
            continue
        selected.add(index)
        used_chars += added_chars
        if used_chars >= budget_chars:
            break

    if not selected:
        return text[:budget_chars]
    return " ".join(sentences[index] for index in sorted(selected))[:budget_chars]


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


def source_text_from_result(result: Any, query: str | None = None) -> str:
    if text := clean_text(getattr(result, "text", None)):
        if query and query.strip() and len(text) > MAX_SOURCE_TEXT_CHARS:
            return compress_text_for_query(text, query, budget_chars=MAX_SOURCE_TEXT_CHARS)
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


def evidence_source_from_result(result: Any, fallback_index: int, query: str | None = None) -> EvidenceSource | None:
    evidence_text = source_text_from_result(result, query=query)
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


def source_tokens(source: EvidenceSource) -> set[str]:
    text = " ".join([source.title, source.topic, source.evidence_text]).lower()
    return {token for token in re.findall(r"[a-z][a-z0-9-]{2,}", text) if len(token) > 3}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def select_mmr_sources(
    sources: list[EvidenceSource],
    *,
    max_sources: int,
    lambda_weight: float = MMR_LAMBDA,
) -> list[EvidenceSource]:
    """Select relevant but non-redundant sources for LLM context assembly."""
    if max_sources <= 0:
        return []
    if len(sources) <= max_sources:
        return sources

    selected: list[EvidenceSource] = []
    remaining = list(sources)
    token_cache = {source.source_id: source_tokens(source) for source in remaining}

    while remaining and len(selected) < max_sources:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        def mmr_score(source: EvidenceSource) -> tuple[float, float]:
            diversity_penalty = max(
                jaccard_similarity(token_cache[source.source_id], token_cache[selected_source.source_id])
                for selected_source in selected
            )
            score = (lambda_weight * source.score) - ((1.0 - lambda_weight) * diversity_penalty)
            return score, source.score

        best = max(remaining, key=mmr_score)
        remaining.remove(best)
        selected.append(best)

    return selected


def collect_evidence_sources(response: UnifiedSearchResponse, max_sources: int = MAX_SOURCES) -> list[EvidenceSource]:
    candidates = list(response.paper_results) + list(response.chunk_results)
    ranked = sorted(candidates, key=source_score, reverse=True)

    sources: list[EvidenceSource] = []
    seen: set[str] = set()
    pool_size = max(max_sources, max_sources * MMR_CANDIDATE_POOL_MULTIPLIER)
    for index, result in enumerate(ranked, start=1):
        source = evidence_source_from_result(result, index, query=response.query)
        if source is None or source.source_id in seen:
            continue
        seen.add(source.source_id)
        sources.append(source)
        if len(sources) >= pool_size:
            break
    return select_mmr_sources(sources, max_sources=max_sources)


def synthesis_question_type(query: str) -> str:
    lowered = query.lower()
    if any(token in lowered for token in ("explain the", "explain this", "explain that", "summarize the", "summarize this")) and "paper" in lowered:
        return "paper_explanation"
    if any(token in lowered for token in ("highly cited", "most cited", "top cited", "published after", "published before", "survey papers", "top papers")):
        return "metadata_listing"
    if any(token in lowered for token in ("read", "reading", "start", "first", "path", "papers should")):
        return "reading_path"
    if any(token in lowered for token in ("limitation", "limitations", "open problem", "future work", "unsolved", "challenge")):
        return "open_problem"
    if any(token in lowered for token in ("dataset", "benchmark", "metric", "evaluate", "evaluation", "experiment", "experiments", "methodology", "method")):
        return "evidence_detail"
    if any(token in lowered for token in ("compare", "versus", " vs ", "difference", "tradeoff")):
        return "comparison"
    return "overview"


def answer_template_instructions(query: str) -> str:
    question_type = synthesis_question_type(query)
    templates = {
        "paper_explanation": """Question-type template: specific paper explanation
- Direct answer paragraph 1: state what the paper proposes and why it matters.
- Direct answer paragraph 2: explain how the method or study works, including experiments or evaluation only when the retrieved evidence states them.
- Optional paragraph 3: summarize the main result or limitation if the evidence supports it.
- Do not turn this into a broad literature survey unless the user asks for comparison or context.""",
        "comparison": """Question-type template: comparison
- Direct answer paragraph 1: define both items and state the most important difference.
- Direct answer paragraph 2: compare mechanism, evidence, and practical tradeoff side by side.
- Name any missing direct head-to-head evidence explicitly instead of forcing a conclusion.""",
        "evidence_detail": """Question-type template: methods, datasets, experiments, or evaluation
- Direct answer paragraph 1: answer the requested method, dataset, benchmark, metric, or experiment question directly.
- Direct answer paragraph 2: connect the answer to the strongest paper or chunk evidence.
- If datasets or metrics are not stated in the retrieved evidence, say that clearly.""",
        "reading_path": """Question-type template: reading guidance
- Direct answer paragraph 1: explain the recommended learning order in plain language.
- Direct answer paragraph 2: identify foundation papers first, then method or benchmark papers, then limitations or survey papers.
- Avoid claiming one paper is universally best unless the retrieved evidence supports that ranking.""",
        "open_problem": """Question-type template: limitations and open problems
- Direct answer paragraph 1: name the main unresolved issues found in the retrieved evidence.
- Direct answer paragraph 2: explain why those gaps matter for future research or evaluation.
- Do not list generic limitations that are not grounded in the provided sources.""",
        "metadata_listing": """Question-type template: ranked bibliography
- Direct answer paragraph 1: summarize the ranked set of papers, including year/citation patterns when present.
- Direct answer paragraph 2: explain why the top papers are relevant to the user's filter.
- Do not invent detailed experimental claims from metadata-only evidence.""",
        "overview": """Question-type template: conceptual overview
- Direct answer paragraph 1: give the main conceptual answer in plain language.
- Direct answer paragraph 2: connect the concept to 2-3 strongest retrieved sources.
- Keep extra caveats short unless the user explicitly asks for limitations.""",
    }
    return templates[question_type]


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

{answer_template_instructions(query)}

Direct-answer requirements:
- Answer the exact user question in 2-3 concise paragraphs.
- First paragraph: give the plain-language conceptual answer before naming papers or methods.
- Second paragraph: connect that concept to the strongest retrieved evidence using source IDs.
- Every direct-answer paragraph should include at least one exact SOURCE_ID when evidence supports the claim.
- Cite every factual sentence immediately with one or more exact SOURCE_IDs; a paragraph-level citation is not enough.
- If a sentence combines claims from multiple sources, include every supporting SOURCE_ID after that sentence.
- Omit any metric, dataset, method name, or result not explicitly supported by its cited evidence; state that the evidence does not specify it instead.
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


def sanitize_source_references(
    payload: dict[str, Any],
    sources: list[EvidenceSource],
) -> tuple[dict[str, Any], list[str]]:
    """Constrain generated source references to the retrieved evidence set."""

    known_ids = {source.source_id for source in sources}
    warnings: list[str] = []

    def clean_text_references(value: Any, field_name: str) -> str:
        text = clean_text(value if isinstance(value, str) else "")
        unknown: set[str] = set()
        known = False

        def replace(match: re.Match[str]) -> str:
            nonlocal known
            reference = match.group(0)
            if reference in known_ids:
                known = True
                return reference
            unknown.add(reference)
            return ""

        cleaned = SOURCE_REFERENCE_PATTERN.sub(replace, text)
        if unknown:
            warnings.append(f"Removed unknown source references from {field_name}: {', '.join(sorted(unknown))}.")
        if field_name.startswith("evidence_bullets") and unknown and not known:
            return ""
        return clean_text(cleaned)

    sanitized = dict(payload)
    sanitized["direct_answer"] = clean_text_references(payload.get("direct_answer"), "direct_answer")

    for field_name in ("evidence_bullets", "limitations", "open_problems"):
        values = payload.get(field_name) or []
        cleaned_values = []
        for index, value in enumerate(values):
            cleaned = clean_text_references(value, f"{field_name}[{index}]")
            if cleaned:
                cleaned_values.append(cleaned)
        sanitized[field_name] = cleaned_values

    clean_themes = []
    for index, raw_theme in enumerate(payload.get("themes") or []):
        if not isinstance(raw_theme, dict):
            warnings.append(f"Dropped malformed theme at index {index}.")
            continue
        theme = dict(raw_theme)
        theme["summary"] = clean_text_references(theme.get("summary"), f"themes[{index}].summary")
        source_ids = [str(source_id) for source_id in (theme.get("supporting_source_ids") or [])]
        valid_ids = [source_id for source_id in source_ids if source_id in known_ids]
        unknown_ids = sorted(set(source_ids) - known_ids)
        if unknown_ids:
            warnings.append(
                f"Removed unknown source references from themes[{index}].supporting_source_ids: {', '.join(unknown_ids)}."
            )
        if valid_ids:
            theme["supporting_source_ids"] = valid_ids
            clean_themes.append(theme)
        else:
            warnings.append(f"Dropped theme at index {index} because it had no retrieved supporting source.")
    sanitized["themes"] = clean_themes
    return sanitized, warnings


def ensure_direct_answer_citations(answer: str, sources: list[EvidenceSource]) -> str:
    cleaned = clean_text(answer)
    if not cleaned or not sources:
        return answer
    known_ids = [source.source_id for source in sources if source.source_id]
    if any(source_id in cleaned for source_id in known_ids):
        return answer
    cited = "; ".join(known_ids[:2])
    return f"{answer.rstrip()} Sources: {cited}."


def build_metadata_listing_brief(
    query: str,
    confidence: ConfidenceAssessment,
    sources: list[EvidenceSource],
) -> ResearchBrief:
    ranked_sources = sources[:8]
    if ranked_sources:
        lead = "The corpus returned a ranked bibliography for this metadata request."
        paper_summaries = []
        for index, source in enumerate(ranked_sources[:5], start=1):
            year = source.year or "unknown year"
            citations = f"{source.citation_count} citations"
            paper_summaries.append(f"{index}. {source.title} ({year}, {citations}; {source.source_id})")
        direct_answer = lead + " Top matches: " + " ".join(paper_summaries)
    else:
        direct_answer = "No papers matched this metadata request in the indexed corpus."

    return ResearchBrief(
        query=query,
        status="generated",
        confidence_decision=confidence.decision,
        direct_answer=direct_answer,
        themes=[],
        evidence_bullets=[
            "This response is a ranked bibliography generated from retrieved paper metadata, not a synthesized claim about paper contents."
        ],
        limitations=[
            "Metadata listing results rank papers by available corpus metadata; they do not by themselves establish experimental findings."
        ],
        open_problems=[],
        sources=ranked_sources,
    )


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

    if response.route.route == "metadata_filter" and response.paper_results:
        return build_metadata_listing_brief(response.query, confidence, sources)

    if confidence.decision != "sufficient_evidence":
        return build_guarded_brief(response.query, confidence, sources)

    if not sources:
        raise SynthesisError("cannot generate a research brief without evidence sources")

    prompt = build_synthesis_prompt(response.query, sources)
    raw_text = generator(prompt) if generator else call_openai_generator(prompt, model=model)
    payload, reference_warnings = sanitize_source_references(parse_brief_payload(raw_text), sources)

    try:
        themes = [BriefTheme(**item) for item in payload.get("themes", [])]
        warning = "; ".join(reference_warnings) if reference_warnings else None
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
            warning=warning,
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
