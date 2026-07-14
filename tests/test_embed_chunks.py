from full_text.embed_chunks import build_chunk_embedding_text, build_embedding_record, trim_for_embedding


def make_chunk():
    return {
        "chunk_id": "chunk-1",
        "paper_id": "paper-1",
        "title": "A Paper",
        "topic": "RAG",
        "year": 2024,
        "citation_count": 10,
        "chunk_index": 0,
        "total_chunks": 2,
        "section_hint": "methodology",
        "word_count": 5,
        "text": "This chunk describes the method.",
    }


def test_build_chunk_embedding_text_includes_retrieval_context():
    text = build_chunk_embedding_text(make_chunk())

    assert "Title: A Paper" in text
    assert "Section hint: methodology" in text
    assert "This chunk describes the method." in text


def test_build_embedding_record_truncates_and_stores_metadata():
    chunk = make_chunk()
    record = build_embedding_record(chunk, [float(i) for i in range(3072)], "embedding text", "model", 1024)

    assert record["chunk_id"] == "chunk-1"
    assert len(record["embedding"]) == 1024
    assert record["embedding_dimensions"] == 1024
    assert record["metadata"] == chunk



def test_trim_for_embedding_caps_long_text():
    text = "word " * 10000
    trimmed = trim_for_embedding(text, max_chars=100)

    assert len(trimmed) <= 100
    assert trimmed.endswith("word")
