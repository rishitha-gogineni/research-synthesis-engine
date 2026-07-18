import json
import subprocess
import sys

import pytest

from agent.reading_path import ReadingPathError, build_reading_path, select_reading_candidates
from shared.schemas import ConfidenceAssessment, QueryRoute, RetrievedChunk, RetrievedPaper, UnifiedSearchResponse


def make_route(route="hybrid_both", confidence=0.95):
    return QueryRoute(query="Which RAG papers should I read first?", route=route, reason="test", confidence=confidence)


def make_confidence(decision="sufficient_evidence"):
    return ConfidenceAssessment(
        query="Which RAG papers should I read first?",
        route="hybrid_both",
        confidence_score=0.91,
        decision=decision,
        reason="retrieval is strong",
        recommended_action="proceed",
        signals=["top_score=0.95"],
        result_count=5,
        top_score=0.95,
        route_confidence=0.95,
    )


def make_paper(paper_id, title, year, citations, score, methodology="method", dataset="not stated in abstract", limitations="not stated in abstract"):
    return RetrievedPaper(
        paper_id=paper_id,
        title=title,
        authors=["Author A"],
        topic="Retrieval-Augmented Generation (RAG)",
        year=year,
        citation_count=citations,
        abstract=f"{title} discusses retrieval grounding and evaluation.",
        main_contribution=f"Contribution from {title}.",
        methodology=methodology,
        dataset_used=dataset,
        key_result=f"Key result from {title}.",
        limitations=limitations,
        blended_score=score,
        rerank_score=score,
        hybrid_score=score,
    )


def make_chunk(chunk_id, paper_id, title, score, text="Limitations include benchmark coverage and generalization gaps."):
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title=title,
        topic="Retrieval-Augmented Generation (RAG)",
        year=2024,
        citation_count=20,
        section_hint="Limitations",
        text=text,
        blended_score=score,
        rerank_score=score,
        dense_score=score,
    )


def make_response(papers=None, chunks=None):
    papers = papers or []
    chunks = chunks or []
    return UnifiedSearchResponse(
        query="Which RAG papers should I read first?",
        route=make_route(),
        paper_result_count=len(papers),
        chunk_result_count=len(chunks),
        paper_results=papers,
        chunk_results=chunks,
    )


def generator_for(ids_by_stage):
    def fake_generator(prompt):
        stages = []
        for stage, paper_ids in ids_by_stage:
            stages.append(
                {
                    "stage": stage,
                    "description": f"Read {stage} papers in this part of the sequence.",
                    "papers": [
                        {
                            "paper_id": paper_id,
                            "reason_to_read": f"{paper_id} is supported by retrieved evidence.",
                            "focus_points": ["retrieval grounding", "evaluation"],
                            "prerequisites": [],
                            "connection_to_next": "This prepares the next step.",
                            "source_ids": [f"paper:{paper_id}"],
                        }
                        for paper_id in paper_ids
                    ],
                }
            )
        return json.dumps({"stages": stages, "limitations": ["Limited to retrieved corpus."]})

    return fake_generator


def test_foundational_papers_appear_before_recent_advances():
    response = make_response(
        papers=[
            make_paper("p-old", "Foundational RAG", 2020, 900, 0.8),
            make_paper("p-recent", "Recent RAG Agent", 2025, 30, 0.96),
        ]
    )

    path = build_reading_path(
        response,
        confidence=make_confidence(),
        generator=generator_for([("foundational", ["p-old"]), ("recent_advances", ["p-recent"])]),
    )

    assert path.stages[0].stage == "foundational"
    assert path.stages[0].papers[0].paper_id == "p-old"
    assert path.stages[1].stage == "recent_advances"
    assert path.stages[1].papers[0].order == 2


def test_duplicate_papers_are_removed_when_paper_and_chunk_match():
    response = make_response(
        papers=[make_paper("p1", "Shared Paper", 2021, 100, 0.8)],
        chunks=[make_chunk("c1", "p1", "Shared Paper", 0.95)],
    )

    selected = select_reading_candidates(response, max_papers=5)
    flattened = [candidate for candidates in selected.values() for candidate in candidates]

    assert len([candidate for candidate in flattened if candidate.paper_id == "p1"]) == 1
    assert flattened[0].source_ids == ["paper:p1", "chunk:c1"]


def test_max_paper_count_is_respected_even_if_llm_returns_more_items():
    response = make_response(
        papers=[
            make_paper("p1", "Paper 1", 2020, 100, 0.9),
            make_paper("p2", "Paper 2", 2021, 90, 0.88),
            make_paper("p3", "Paper 3", 2025, 1, 0.99),
        ]
    )

    path = build_reading_path(
        response,
        confidence=make_confidence(),
        generator=generator_for([("foundational", ["p1", "p2", "p3"])]),
        max_papers=2,
    )

    assert path.total_papers == 2
    assert [item.order for stage in path.stages for item in stage.papers] == [1, 2]


def test_low_citation_high_relevance_paper_is_not_excluded():
    response = make_response(
        papers=[
            make_paper("p-cited", "Cited Overview", 2020, 1000, 0.65),
            make_paper("p-relevant", "Focused Recent Method", 2025, 2, 0.99, methodology="new focused method"),
        ]
    )

    selected = select_reading_candidates(response, max_papers=2)
    selected_ids = {candidate.paper_id for candidates in selected.values() for candidate in candidates}

    assert "p-relevant" in selected_ids


def test_recent_papers_are_represented_when_available():
    response = make_response(
        papers=[
            make_paper("p-old", "Older Paper", 2019, 1000, 0.8),
            make_paper("p-new", "New Paper", 2025, 10, 0.85),
        ]
    )

    selected = select_reading_candidates(response, max_papers=3)

    assert any(candidate.paper_id == "p-new" for candidate in selected.get("recent_advances", []))


def test_invalid_llm_returned_paper_ids_are_rejected():
    response = make_response(papers=[make_paper("p1", "Known Paper", 2020, 100, 0.9)])

    with pytest.raises(ReadingPathError):
        build_reading_path(
            response,
            confidence=make_confidence(),
            generator=generator_for([("foundational", ["unknown-paper"])]),
        )


def test_fewer_than_five_available_papers_are_handled():
    response = make_response(papers=[make_paper("p1", "Only Paper", 2022, 40, 0.9)])

    path = build_reading_path(response, confidence=make_confidence(), generator=generator_for([("core_methods", ["p1"])]))

    assert path.total_papers == 1
    assert path.stages[0].papers[0].title == "Only Paper"


def test_no_result_input_returns_guarded_response():
    response = make_response()

    path = build_reading_path(response, confidence=make_confidence(decision="insufficient_evidence"), generator=lambda _: "{}")

    assert path.total_papers == 0
    assert path.confidence_decision == "insufficient_evidence"


def test_low_confidence_retrieval_skips_generation():
    response = make_response(papers=[make_paper("p1", "Known Paper", 2020, 100, 0.9)])

    def should_not_run(_):
        raise AssertionError("generator should not run")

    path = build_reading_path(response, confidence=make_confidence(decision="broaden_search"), generator=should_not_run)

    assert path.stages == []
    assert path.confidence_decision == "broaden_search"


def test_malformed_json_receives_one_retry():
    response = make_response(papers=[make_paper("p1", "Known Paper", 2020, 100, 0.9)])
    calls = []

    def flaky_generator(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "not json"
        return generator_for([("foundational", ["p1"])] )(prompt)

    path = build_reading_path(response, confidence=make_confidence(), generator=flaky_generator)

    assert len(calls) == 2
    assert path.total_papers == 1


def test_reading_path_cli_accepts_low_confidence_input_without_openai(tmp_path):
    response = make_response()
    path = tmp_path / "response.json"
    path.write_text(response.model_dump_json())

    completed = subprocess.run(
        [sys.executable, "-m", "agent.reading_path", "--input", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["total_papers"] == 0
    assert payload["confidence_decision"] in {"insufficient_evidence", "broaden_search", "ask_clarifying_question"}
