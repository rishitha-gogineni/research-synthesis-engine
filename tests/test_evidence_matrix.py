import json
import subprocess
import sys

from agent.evidence_matrix import (
    MISSING_VALUE,
    build_evidence_matrix,
    matrix_to_markdown,
    normalized_value,
    row_from_result,
    strength_from_score,
)
from shared.schemas import EvidenceMatrix, QueryRoute, RetrievedChunk, RetrievedPaper, UnifiedSearchResponse


def make_route(route="hybrid_both", confidence=0.9):
    return QueryRoute(query="Which evidence supports RAG for hallucination reduction?", route=route, reason="test", confidence=confidence)


def make_paper(score=0.82):
    return RetrievedPaper(
        paper_id="p1",
        title="Retrieval-Augmented Generation for Large Language Models: A Survey",
        topic="Retrieval-Augmented Generation (RAG)",
        year=2023,
        citation_count=670,
        abstract="RAG connects generation with retrieved evidence.",
        main_contribution="Comprehensive review of RAG paradigms and grounding strategies.",
        methodology="Survey and taxonomy.",
        dataset_used="not stated in abstract",
        key_result="RAG improves factual grounding when retrieved evidence is relevant.",
        limitations="Survey abstracts do not report one controlled experiment.",
        blended_score=score,
        rerank_score=score,
    )


def make_chunk(score=0.9):
    return RetrievedChunk(
        chunk_id="c1",
        paper_id="p1",
        title="Retrieval-Augmented Generation for Large Language Models: A Survey",
        topic="Retrieval-Augmented Generation (RAG)",
        text="Grounding generation in retrieved passages reduces unsupported claims.",
        section_hint="Grounding",
        blended_score=score,
        rerank_score=score,
    )


def make_response(papers=None, chunks=None):
    papers = papers or []
    chunks = chunks or []
    return UnifiedSearchResponse(
        query="Which evidence supports RAG for hallucination reduction?",
        route=make_route(),
        paper_result_count=len(papers),
        chunk_result_count=len(chunks),
        paper_results=papers,
        chunk_results=chunks,
    )


def test_normalized_value_keeps_real_values_and_relabels_missing_values():
    assert normalized_value("TruthfulQA") == "TruthfulQA"
    assert normalized_value("not stated in abstract") == MISSING_VALUE
    assert normalized_value(None) == MISSING_VALUE


def test_strength_from_score_buckets_scores():
    assert strength_from_score(0.9) == "high"
    assert strength_from_score(0.5) == "medium"
    assert strength_from_score(0.2) == "low"


def test_row_from_paper_preserves_structured_metadata():
    row = row_from_result(make_paper(), 1)

    assert row.claim.startswith("Comprehensive review")
    assert row.source_ids == ["paper:p1"]
    assert row.methodology == "Survey and taxonomy."
    assert row.dataset == MISSING_VALUE
    assert row.key_result.startswith("RAG improves factual grounding")
    assert row.evidence_strength == "high"


def test_row_from_chunk_uses_chunk_source_id_and_text_snippet():
    row = row_from_result(make_chunk(), 1)

    assert row.claim.endswith("(Grounding)")
    assert row.source_ids == ["chunk:c1"]
    assert row.evidence_snippet.startswith("Grounding generation")


def test_build_evidence_matrix_sorts_results_and_renders_markdown():
    response = make_response(papers=[make_paper(score=0.7)], chunks=[make_chunk(score=0.95)])

    matrix = build_evidence_matrix(response, max_rows=2)

    assert len(matrix.rows) == 2
    assert matrix.rows[0].source_ids == ["chunk:c1"]
    assert "| Claim | Sources | Methodology | Dataset | Key Result | Limitation | Strength |" in matrix.markdown


def test_matrix_to_markdown_handles_empty_rows():
    matrix = EvidenceMatrix(query="empty", rows=[])

    assert matrix_to_markdown(matrix) == "No retrieved evidence rows available."


def test_evidence_matrix_cli_prints_markdown(tmp_path):
    response = make_response(papers=[make_paper()], chunks=[make_chunk()])
    path = tmp_path / "response.json"
    path.write_text(response.model_dump_json())

    completed = subprocess.run(
        [sys.executable, "-m", "agent.evidence_matrix", "--input", str(path), "--markdown"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "| Claim | Sources |" in completed.stdout
    assert "chunk:c1" in completed.stdout
