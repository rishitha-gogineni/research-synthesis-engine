from retrieval.index_qdrant import build_payload, point_id_for


def make_record():
    return {
        "paper_id": "https://openalex.org/W123",
        "title": "A test paper",
        "topic": "Retrieval-Augmented Generation (RAG)",
        "year": 2024,
        "citation_count": 42,
        "embedding": [0.1] * 1024,
        "embedding_text": "Title: A test paper",
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 1024,
        "metadata": {
            "authors": ["Ada Lovelace"],
            "abstract": "A useful abstract.",
            "arxiv_id": "2401.12345",
            "url": "https://example.com",
            "main_contribution": "Introduces a method.",
            "methodology": "Evaluates it.",
            "dataset_used": "not specified",
            "key_result": "not specified",
            "limitations": "not specified",
        },
    }


def test_point_id_for_is_stable_uuid():
    record = make_record()

    assert point_id_for(record) == point_id_for(record)
    assert len(point_id_for(record)) == 36


def test_build_payload_flattens_metadata_for_qdrant():
    payload = build_payload(make_record())

    assert payload["paper_id"] == "https://openalex.org/W123"
    assert payload["title"] == "A test paper"
    assert payload["authors"] == ["Ada Lovelace"]
    assert payload["main_contribution"] == "Introduces a method."
    assert payload["embedding_dimensions"] == 1024

