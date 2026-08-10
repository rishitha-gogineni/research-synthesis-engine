from retrieval.context_expansion import aggregate_paper_evidence, expand_chunk_context
from retrieval.corpus_index import CorpusIndex


def sample_index():
    papers = [
        {"paper_id": "p1", "title": "Paper One", "topic": "RAG", "year": 2024},
        {"paper_id": "p2", "title": "Paper Two", "topic": "RAG", "year": 2023},
    ]
    chunks = [
        {"chunk_id": "c1", "paper_id": "p1", "title": "Paper One", "chunk_index": 0, "text": "Introduction context."},
        {"chunk_id": "c2", "paper_id": "p1", "title": "Paper One", "chunk_index": 1, "text": "The key experiment compares methods."},
        {"chunk_id": "c3", "paper_id": "p1", "title": "Paper One", "chunk_index": 2, "text": "The result is reported in this section."},
        {"chunk_id": "c4", "paper_id": "p2", "title": "Paper Two", "chunk_index": 0, "text": "A second paper result."},
    ]
    return CorpusIndex.from_records(papers, chunks)


def test_expand_chunk_context_keeps_anchor_id_and_adds_neighbors():
    candidate = {
        "chunk_id": "c2",
        "paper_id": "p1",
        "chunk_index": 1,
        "text": "The key experiment compares methods.",
        "matched_by": ["chunk_dense"],
    }

    expanded = expand_chunk_context([candidate], sample_index(), top_n=1)

    assert expanded[0]["chunk_id"] == "c2"
    assert "Introduction context." in expanded[0]["text"]
    assert "The result is reported" in expanded[0]["text"]
    assert "parent_context" in expanded[0]["matched_by"]
    assert expanded[0]["score_breakdown"]["parent_context"]["context_chunk_ids"] == ["c1", "c2", "c3"]


def test_aggregate_paper_evidence_backfills_parent_paper():
    papers = [{"paper_id": "p1", "title": "Paper One", "topic": "RAG", "year": 2024}]
    chunks = [{"chunk_id": "c1", "paper_id": "p1", "chunk_index": 0, "text": "evidence"}]
    index = CorpusIndex.from_records(papers, chunks)

    results = aggregate_paper_evidence(
        [],
        [{"chunk_id": "c1", "paper_id": "p1", "hybrid_score": 0.8}],
        index,
        top_k=1,
    )

    assert results[0]["paper_id"] == "p1"
    assert "chunk_evidence_backfill" in results[0]["matched_by"]
    assert results[0]["score_breakdown"]["chunk_evidence"]["chunk_count"] == 1

