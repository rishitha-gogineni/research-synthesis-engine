from full_text.index_chunks_qdrant import build_chunk_payload, point_id_for_chunk


def make_record():
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
        "word_count": 100,
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 1024,
        "embedding": [0.1] * 1024,
        "metadata": {
            "text": "Chunk text",
            "pdf_url": "https://example.org/p.pdf",
            "source_type": "arxiv",
            "page_count": 10,
        },
    }


def test_point_id_for_chunk_is_stable_uuid():
    record = make_record()
    assert point_id_for_chunk(record) == point_id_for_chunk(record)
    assert len(point_id_for_chunk(record)) == 36


def test_build_chunk_payload_flattens_metadata():
    payload = build_chunk_payload(make_record())

    assert payload["chunk_id"] == "chunk-1"
    assert payload["text"] == "Chunk text"
    assert payload["section_hint"] == "methodology"
    assert payload["embedding_dimensions"] == 1024
