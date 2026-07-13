import pytest

from ingestion.embed import build_embedding_record, build_embedding_text, truncate_embedding
from shared.schemas import EnrichedPaper


def make_enriched_paper() -> EnrichedPaper:
    return EnrichedPaper(
        paper_id="W123",
        title="A test paper",
        abstract="This paper studies retrieval augmented generation.",
        authors=["Ada Lovelace"],
        citation_count=42,
        arxiv_id="2401.12345",
        url="https://example.com",
        year=2024,
        topic="Retrieval-Augmented Generation (RAG)",
        main_contribution="Introduces a retrieval method.",
        methodology="Evaluates retrieval over question answering.",
        dataset_used="QA benchmark",
        key_result="Improves answer grounding.",
        limitations="not stated in abstract",
    )


def test_build_embedding_text_contains_core_fields():
    text = build_embedding_text(make_enriched_paper())

    assert "Title: A test paper" in text
    assert "Topic: Retrieval-Augmented Generation (RAG)" in text
    assert "Main contribution: Introduces a retrieval method." in text
    assert "Limitations: not stated in abstract" in text


def test_truncate_embedding_keeps_prefix_dimensions():
    embedding = [float(index) for index in range(3072)]

    truncated = truncate_embedding(embedding, dimensions=1024)

    assert len(truncated) == 1024
    assert truncated[0] == 0.0
    assert truncated[-1] == 1023.0


def test_truncate_embedding_rejects_short_vectors():
    with pytest.raises(ValueError):
        truncate_embedding([0.1, 0.2], dimensions=3)


def test_build_embedding_record_stores_metadata_and_truncated_embedding():
    paper = make_enriched_paper()
    full_embedding = [float(index) for index in range(3072)]

    record = build_embedding_record(
        paper=paper,
        full_embedding=full_embedding,
        embedding_text=build_embedding_text(paper),
        model="text-embedding-3-large",
        dimensions=1024,
    )

    assert record["paper_id"] == "W123"
    assert record["embedding_model"] == "text-embedding-3-large"
    assert record["full_embedding_dimensions"] == 3072
    assert record["embedding_dimensions"] == 1024
    assert len(record["embedding"]) == 1024
    assert record["metadata"]["main_contribution"] == "Introduces a retrieval method."

