import json

import pytest

from agent.faithfulness_eval import (
    FaithfulnessEvalError,
    assess_faithfulness,
    build_faithfulness_prompt,
    load_brief,
    parse_faithfulness_payload,
    run_faithfulness_eval,
    summarize_faithfulness_assessments,
)
from shared.schemas import EvidenceSource, FaithfulnessAssessment, ResearchBrief


def make_source(source_id="chunk:c1", text="Retrieved passages provide external evidence that reduces hallucination."):
    return EvidenceSource(
        source_id=source_id,
        title="A Paper",
        topic="Retrieval-Augmented Generation (RAG)",
        paper_id="p1",
        year=2023,
        citation_count=10,
        evidence_text=text,
        score=0.9,
    )


def make_brief(
    query="How do RAG systems reduce hallucinations?",
    status="generated",
    direct_answer="RAG systems reduce hallucinations by grounding generation in retrieved evidence [chunk:c1].",
    sources=None,
):
    return ResearchBrief(
        query=query,
        status=status,
        confidence_decision="sufficient_evidence",
        direct_answer=direct_answer,
        sources=sources if sources is not None else [make_source()],
    )


def fake_generator(payload: dict):
    def _generator(prompt: str) -> str:
        assert "How do RAG systems reduce hallucinations" in prompt
        return json.dumps(payload)

    return _generator


def test_build_faithfulness_prompt_includes_query_answer_and_evidence():
    brief = make_brief()

    prompt = build_faithfulness_prompt(brief)

    assert brief.query in prompt
    assert brief.direct_answer in prompt
    assert "[chunk:c1]" in prompt
    assert "Retrieved passages provide external evidence" in prompt


def test_build_faithfulness_prompt_rejects_empty_direct_answer():
    brief = make_brief(direct_answer="")

    with pytest.raises(FaithfulnessEvalError, match="empty direct_answer"):
        build_faithfulness_prompt(brief)


def test_build_faithfulness_prompt_rejects_brief_with_no_evidence_text():
    source = make_source(text="   ")  # min_length=1 passes, but it's blank after stripping
    brief = make_brief(sources=[source])

    with pytest.raises(FaithfulnessEvalError, match="no cited evidence"):
        build_faithfulness_prompt(brief)


def test_parse_faithfulness_payload_handles_code_fenced_json():
    raw = '```json\n{"faithfulness_score": 0.8, "answer_relevancy_score": 0.9}\n```'

    payload = parse_faithfulness_payload(raw)

    assert payload["faithfulness_score"] == 0.8
    assert payload["answer_relevancy_score"] == 0.9


def test_parse_faithfulness_payload_extracts_json_from_surrounding_prose():
    raw = 'Here is my judgment: {"faithfulness_score": 0.5, "answer_relevancy_score": 0.6} Thanks!'

    payload = parse_faithfulness_payload(raw)

    assert payload["faithfulness_score"] == 0.5


def test_parse_faithfulness_payload_raises_on_garbage():
    with pytest.raises(FaithfulnessEvalError, match="valid JSON"):
        parse_faithfulness_payload("not json at all, no braces here")


def test_assess_faithfulness_returns_populated_assessment():
    brief = make_brief()
    generator = fake_generator(
        {
            "faithfulness_score": 0.9,
            "answer_relevancy_score": 1.0,
            "unsupported_claims": [],
            "judge_notes": "Answer is fully grounded in the cited chunk.",
        }
    )

    assessment = assess_faithfulness(brief, generator=generator)

    assert isinstance(assessment, FaithfulnessAssessment)
    assert assessment.query == brief.query
    assert assessment.faithfulness_score == 0.9
    assert assessment.answer_relevancy_score == 1.0
    assert assessment.unsupported_claims == []
    assert assessment.source_ids_checked == ["chunk:c1"]


def test_assess_faithfulness_surfaces_unsupported_claims():
    brief = make_brief(
        direct_answer="RAG reduces hallucinations and also cures the common cold [chunk:c1].",
    )
    generator = fake_generator(
        {
            "faithfulness_score": 0.5,
            "answer_relevancy_score": 0.8,
            "unsupported_claims": ["also cures the common cold"],
            "judge_notes": "The cold-curing claim has no support in the evidence.",
        }
    )

    assessment = assess_faithfulness(brief, generator=generator)

    assert assessment.unsupported_claims == ["also cures the common cold"]
    assert assessment.faithfulness_score < 1.0


def test_assess_faithfulness_raises_on_missing_required_fields():
    brief = make_brief()
    generator = fake_generator({"faithfulness_score": 0.9})  # missing answer_relevancy_score

    with pytest.raises(FaithfulnessEvalError, match="missing required fields"):
        assess_faithfulness(brief, generator=generator)


def test_assess_faithfulness_clamps_are_enforced_by_schema():
    brief = make_brief()
    generator = fake_generator({"faithfulness_score": 1.5, "answer_relevancy_score": 0.9})

    with pytest.raises(FaithfulnessEvalError):
        assess_faithfulness(brief, generator=generator)


def test_summarize_faithfulness_assessments_computes_means_and_flags():
    assessments = [
        FaithfulnessAssessment(query="q1", faithfulness_score=1.0, answer_relevancy_score=1.0, unsupported_claims=[]),
        FaithfulnessAssessment(
            query="q2",
            faithfulness_score=0.5,
            answer_relevancy_score=0.7,
            unsupported_claims=["x cures y"],
        ),
    ]

    summary = summarize_faithfulness_assessments(assessments)

    assert summary["evaluated_count"] == 2
    assert summary["mean_faithfulness"] == 0.75
    assert summary["mean_answer_relevancy"] == 0.85
    assert summary["min_faithfulness"] == 0.5
    assert summary["briefs_with_unsupported_claims"] == 1


def test_summarize_faithfulness_assessments_handles_empty_list():
    summary = summarize_faithfulness_assessments([])

    assert summary["evaluated_count"] == 0
    assert summary["mean_faithfulness"] is None


def test_run_faithfulness_eval_skips_guarded_and_empty_answer_briefs():
    generated_brief = make_brief()
    guarded_brief = make_brief(status="skipped_low_confidence", direct_answer="")
    empty_answer_brief = make_brief(direct_answer="")

    generator = fake_generator(
        {"faithfulness_score": 0.9, "answer_relevancy_score": 0.95, "unsupported_claims": []}
    )

    summary, assessments = run_faithfulness_eval(
        [generated_brief, guarded_brief, empty_answer_brief],
        generator=generator,
    )

    assert len(assessments) == 1
    assert summary["evaluated_count"] == 1


def test_load_brief_round_trips_a_saved_research_brief(tmp_path):
    brief = make_brief()
    path = tmp_path / "brief.json"
    path.write_text(brief.model_dump_json(), encoding="utf-8")

    loaded = load_brief(path)

    assert loaded.query == brief.query
    assert loaded.sources[0].source_id == "chunk:c1"


def test_load_brief_raises_for_missing_file(tmp_path):
    with pytest.raises(FaithfulnessEvalError, match="failed to load research brief"):
        load_brief(tmp_path / "does-not-exist.json")
