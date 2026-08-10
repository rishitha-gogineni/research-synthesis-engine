"""LLM-judge faithfulness and answer-relevancy scoring for generated research briefs.

Retrieval evaluation (retrieval/evaluate.py) answers "did we find the right
evidence?" via Recall@K, MRR, and route accuracy. It says nothing about
whether the *generated* direct_answer actually sticks to what that evidence
supports. A brief can cite exactly the right sources and still assert
something they don't say -- that's a faithfulness failure, not a retrieval
failure, and none of the IR metrics can catch it.

This module asks a second LLM call to act as a judge: given a brief's
direct_answer and the evidence_text of every source it cites, score how much
of the answer is actually grounded (faithfulness) and how directly it
addresses the original query (answer_relevancy), independent of grounding.
This mirrors the RAGAS faithfulness/answer_relevancy metrics referenced in
retrieval-at-scale write-ups, implemented directly against this project's
own ResearchBrief schema rather than pulling in the ragas package.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from pathlib import Path
from typing import Callable

from openai import OpenAI

from retrieval.index_qdrant import load_env_file
from shared.schemas import FaithfulnessAssessment, ResearchBrief


DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
JudgeGenerator = Callable[[str], str]


class FaithfulnessEvalError(RuntimeError):
    """Raised when a faithfulness assessment cannot be generated or parsed."""


def build_faithfulness_prompt(brief: ResearchBrief) -> str:
    if not brief.direct_answer.strip():
        raise FaithfulnessEvalError("cannot judge a brief with an empty direct_answer")

    source_pattern = re.compile(r"(?<![A-Za-z0-9_-])(?:paper|chunk|result):[A-Za-z0-9_.:/-]+")
    cited_ids = {match.group(0) for match in source_pattern.finditer(brief.direct_answer)}
    if not cited_ids:
        raise FaithfulnessEvalError("cannot judge a brief with no source citations")
    evidence_blocks = "\n".join(
        f"[{source.source_id}] {source.evidence_text}"
        for source in brief.sources
        if source.source_id in cited_ids and source.evidence_text.strip()
    )
    if not evidence_blocks:
        raise FaithfulnessEvalError("cannot judge a brief with no cited evidence text")

    return f"""You are a strict fact-checking judge for a research-synthesis system.

QUERY:
{brief.query}

GENERATED ANSWER (to be judged):
{brief.direct_answer}

CITED EVIDENCE (the only material the answer is allowed to rely on):
{evidence_blocks}

Judge two things independently:

1. faithfulness_score (0.0 to 1.0): the fraction of factual claims in the
   generated answer that are directly supported by the cited evidence above.
   A claim not stated or implied by the evidence is unsupported, even if it
   sounds plausible or is generally true. 1.0 means every claim is grounded.
   0.0 means the answer is not grounded in the evidence at all.
2. answer_relevancy_score (0.0 to 1.0): how directly the generated answer
   addresses the query above, regardless of whether it's grounded. A
   well-grounded but off-topic answer should still score low here.

List any specific unsupported_claims as short quoted phrases from the answer.

Respond with strict JSON only, matching exactly this shape:
{{
  "faithfulness_score": <float 0.0-1.0>,
  "answer_relevancy_score": <float 0.0-1.0>,
  "unsupported_claims": [<string>, ...],
  "judge_notes": "<one or two sentence explanation>"
}}"""


def parse_faithfulness_payload(raw_text: str) -> dict:
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
            raise FaithfulnessEvalError("judge did not return valid JSON") from None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise FaithfulnessEvalError(f"judge returned malformed JSON: {exc}") from exc


def assess_faithfulness(
    brief: ResearchBrief,
    *,
    generator: JudgeGenerator | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
) -> FaithfulnessAssessment:
    """Score a generated brief's faithfulness and answer relevancy via an LLM judge."""

    prompt = build_faithfulness_prompt(brief)
    source_pattern = re.compile(r"(?<![A-Za-z0-9_-])(?:paper|chunk|result):[A-Za-z0-9_.:/-]+")
    cited_ids = {match.group(0) for match in source_pattern.finditer(brief.direct_answer)}
    raw_text = generator(prompt) if generator else call_openai_judge(prompt, model=model)
    payload = parse_faithfulness_payload(raw_text)

    try:
        return FaithfulnessAssessment(
            query=brief.query,
            faithfulness_score=float(payload["faithfulness_score"]),
            answer_relevancy_score=float(payload["answer_relevancy_score"]),
            unsupported_claims=list(payload.get("unsupported_claims", [])),
            judge_notes=str(payload.get("judge_notes", "")),
            source_ids_checked=[source.source_id for source in brief.sources if source.source_id in cited_ids],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FaithfulnessEvalError(f"judge response missing required fields: {exc}") from exc


def call_openai_judge(
    prompt: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    env_file: Path = Path(".env"),
) -> str:
    load_env_file(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise FaithfulnessEvalError("OPENAI_API_KEY is missing. Add it to .env before live faithfulness judging.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON for a faithfulness/answer-relevancy judgment."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise FaithfulnessEvalError("OpenAI returned an empty judgment")
    return content


def load_brief(path: Path) -> ResearchBrief:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ResearchBrief(**payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise FaithfulnessEvalError(f"failed to load research brief from {path}: {exc}") from exc


def summarize_faithfulness_assessments(assessments: list[FaithfulnessAssessment]) -> dict:
    """Aggregate per-brief judgments into corpus-level faithfulness/relevancy stats.

    Mirrors the summary shape of retrieval/evaluate.py's summarize_evaluations
    so both eval loops read the same way in a report or resume bullet.
    """

    if not assessments:
        return {
            "evaluated_count": 0,
            "mean_faithfulness": None,
            "mean_answer_relevancy": None,
            "briefs_with_unsupported_claims": 0,
        }

    faithfulness_scores = [assessment.faithfulness_score for assessment in assessments]
    relevancy_scores = [assessment.answer_relevancy_score for assessment in assessments]
    with_unsupported = sum(1 for assessment in assessments if assessment.unsupported_claims)

    return {
        "evaluated_count": len(assessments),
        "mean_faithfulness": round(statistics.mean(faithfulness_scores), 4),
        "mean_answer_relevancy": round(statistics.mean(relevancy_scores), 4),
        "min_faithfulness": round(min(faithfulness_scores), 4),
        "briefs_with_unsupported_claims": with_unsupported,
    }


def run_faithfulness_eval(
    briefs: list[ResearchBrief],
    *,
    generator: JudgeGenerator | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
) -> tuple[dict, list[FaithfulnessAssessment]]:
    """Judge a batch of generated briefs and return a corpus-level summary plus per-brief detail."""

    assessments = []
    for brief in briefs:
        if brief.status != "generated" or not brief.direct_answer.strip():
            # Guarded/low-confidence briefs have no synthesized answer to judge.
            continue
        assessments.append(assess_faithfulness(brief, generator=generator, model=model))
    return summarize_faithfulness_assessments(assessments), assessments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to a saved ResearchBrief JSON file.")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    brief = load_brief(args.input)
    generator = None
    if args.env_file != Path(".env"):
        generator = lambda prompt: call_openai_judge(prompt, model=args.model, env_file=args.env_file)
    assessment = assess_faithfulness(brief, generator=generator, model=args.model)
    print(assessment.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except FaithfulnessEvalError as exc:
        raise SystemExit(f"Error: {exc}") from None
