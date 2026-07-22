import json
import subprocess
import sys

import pytest

from agent.synthesis import (
    SynthesisError,
    build_research_brief,
    build_synthesis_prompt,
    collect_evidence_sources,
    ensure_direct_answer_citations,
    parse_brief_payload,
)
from shared.schemas import ConfidenceAssessment, QueryRoute, RetrievedChunk, RetrievedPaper, UnifiedSearchResponse


def make_route(route="hybrid_both", confidence=0.92):
    return QueryRoute(query="How do RAG systems reduce hallucinations?", route=route, reason="test route", confidence=confidence)


def make_paper(paper_id="p1", score=0.88):
    return RetrievedPaper(
        paper_id=paper_id,
        title="Retrieval-Augmented Generation for Large Language Models: A Survey",
        topic="Retrieval-Augmented Generation (RAG)",
        year=2023,
        citation_count=670,
        abstract="RAG systems augment generation with retrieved evidence to improve factuality.",
        main_contribution="Surveys RAG architectures and their use for grounding language model outputs.",
        methodology="Taxonomy and comparative survey of RAG frameworks.",
        dataset_used="not stated in abstract",
        key_result="RAG can improve factual grounding when retrieval quality is strong.",
        limitations="not stated in abstract",
        blended_score=score,
        rerank_score=score,
    )


def make_chunk(chunk_id="c1", score=0.94):
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id="p1",
        title="Retrieval-Augmented Generation for Large Language Models: A Survey",
        topic="Retrieval-Augmented Generation (RAG)",
        year=2023,
        citation_count=670,
        text="Retrieved passages provide external evidence that can reduce unsupported generations.",
        section_hint="Grounding",
        blended_score=score,
        rerank_score=score,
        dense_score=score,
    )


def make_response(papers=None, chunks=None):
    papers = papers or []
    chunks = chunks or []
    return UnifiedSearchResponse(
        query="How do RAG systems reduce hallucinations?",
        route=make_route(),
        paper_result_count=len(papers),
        chunk_result_count=len(chunks),
        paper_results=papers,
        chunk_results=chunks,
    )


def make_confidence(decision="sufficient_evidence"):
    return ConfidenceAssessment(
        query="How do RAG systems reduce hallucinations?",
        route="hybrid_both",
        confidence_score=0.9,
        decision=decision,
        reason="test confidence reason",
        recommended_action="test action",
        signals=["top_score=0.94"],
        result_count=2,
        top_score=0.94,
        route_confidence=0.92,
    )


def test_collect_evidence_sources_sorts_and_assigns_stable_ids():
    response = make_response(papers=[make_paper(score=0.7)], chunks=[make_chunk(score=0.95)])

    sources = collect_evidence_sources(response)

    assert [source.source_id for source in sources] == ["chunk:c1", "paper:p1"]
    assert sources[0].score == 0.95
    assert sources[1].evidence_text.startswith("Surveys RAG architectures")


def test_build_synthesis_prompt_restricts_model_to_retrieved_sources():
    sources = collect_evidence_sources(make_response(papers=[make_paper()], chunks=[make_chunk()]))

    prompt = build_synthesis_prompt("What reduces hallucinations?", sources)

    assert "Use only the retrieved sources" in prompt
    assert "chunk:c1" in prompt
    assert "Return only valid JSON" in prompt
    assert "2-3 concise paragraphs" in prompt
    assert "3-5 named research themes" in prompt


def test_synthesis_prompt_guides_agent_vs_chatbot_contrast_answers():
    sources = collect_evidence_sources(
        make_response(
            papers=[make_paper(paper_id="agent-survey", score=0.82)],
            chunks=[
                RetrievedChunk(
                    chunk_id="agent-tools",
                    paper_id="agent-survey",
                    title="A Survey on Large Language Model based Autonomous Agents",
                    topic="AI Agents & Tool Use",
                    year=2024,
                    citation_count=1205,
                    text="Agents use planning, tool APIs, observation feedback, and execution loops to complete tasks.",
                    section_hint="Tool use",
                    blended_score=0.96,
                    rerank_score=0.96,
                    dense_score=0.96,
                )
            ],
        )
    )

    prompt = build_synthesis_prompt(
        "How does an agent perform tasks while a normal chatbot just gives answers?",
        sources,
    )

    assert "First paragraph: give the plain-language conceptual answer" in prompt
    assert "For comparison or contrast questions, define both sides" in prompt
    assert "planning, tool/API use, action execution, observation/feedback" in prompt
    assert "a plain chatbot without tool access mainly generates responses" in prompt
    assert "Prefer broad survey evidence and high-citation sources" in prompt


def test_build_research_brief_uses_mocked_generator_and_validates_schema():
    response = make_response(papers=[make_paper()], chunks=[make_chunk()])
    captured = {}

    def fake_generator(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps(
            {
                "direct_answer": "RAG reduces hallucinations by grounding generation in retrieved evidence.",
                "themes": [
                    {
                        "theme": "Evidence grounding",
                        "summary": "Retrieved context gives the model external support for factual claims.",
                        "supporting_source_ids": ["chunk:c1", "paper:p1"],
                    }
                ],
                "evidence_bullets": ["Retrieved passages provide evidence for unsupported claims [chunk:c1]."],
                "limitations": ["The retrieved evidence does not quantify a single universal reduction."],
                "open_problems": ["Compare grounding quality across retrieval methods."],
            }
        )

    brief = build_research_brief(response, confidence=make_confidence(), generator=fake_generator)

    assert brief.status == "generated"
    assert brief.confidence_decision == "sufficient_evidence"
    assert brief.themes[0].supporting_source_ids == ["chunk:c1", "paper:p1"]
    assert "SOURCE_ID: chunk:c1" in captured["prompt"]


def test_direct_answer_citation_guard_adds_source_ids_when_missing():
    sources = collect_evidence_sources(make_response(papers=[make_paper()], chunks=[make_chunk()]))

    answer = ensure_direct_answer_citations("RAG grounds answers using retrieved evidence.", sources)

    assert "chunk:c1" in answer
    assert "paper:p1" in answer


def test_direct_answer_citation_guard_preserves_existing_citations():
    sources = collect_evidence_sources(make_response(papers=[make_paper()], chunks=[make_chunk()]))

    answer = ensure_direct_answer_citations("RAG grounds answers using retrieved evidence (SOURCE_ID: chunk:c1).", sources)

    assert answer == "RAG grounds answers using retrieved evidence (SOURCE_ID: chunk:c1)."


def test_low_confidence_skips_generation():
    response = make_response(papers=[make_paper()])

    def should_not_run(_: str) -> str:
        raise AssertionError("generator should not run when CRAG confidence is low")

    brief = build_research_brief(response, confidence=make_confidence(decision="broaden_search"), generator=should_not_run)

    assert brief.status == "skipped_low_confidence"
    assert brief.warning
    assert "cannot answer" in brief.direct_answer.lower()
    assert "does not invent claims" in brief.direct_answer
    assert brief.open_problems == ["test action"]


def test_parse_brief_payload_accepts_json_fences():
    payload = parse_brief_payload('```json\n{"direct_answer": "ok"}\n```')

    assert payload["direct_answer"] == "ok"


def test_malformed_brief_payload_raises_synthesis_error():
    with pytest.raises(SynthesisError):
        parse_brief_payload("not json")


def test_synthesis_cli_accepts_low_confidence_input_without_openai(tmp_path):
    response = make_response()
    path = tmp_path / "response.json"
    path.write_text(response.model_dump_json())

    completed = subprocess.run(
        [sys.executable, "-m", "agent.synthesis", "--input", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "skipped_low_confidence"
    assert payload["confidence_decision"] in {"insufficient_evidence", "broaden_search", "ask_clarifying_question"}
