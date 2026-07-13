from retrieval.build_bm25 import build_corpus_text, normalize_title, search_bm25, tokenize


class FakeBm25:
    def get_scores(self, query_tokens):
        return [3.0 if "hallucination" in query_tokens else 0.0, 1.0]


def make_record(title, topic):
    return {
        "paper_id": title,
        "title": title,
        "topic": topic,
        "year": 2024,
        "citation_count": 10,
        "embedding_text": f"Title: {title}\nKey result: hallucination detection",
        "metadata": {
            "title": title,
            "abstract": "A paper about hallucination detection.",
            "main_contribution": "Detects hallucinations.",
            "methodology": "not specified",
            "dataset_used": "not specified",
            "key_result": "not specified",
            "limitations": "not specified",
        },
    }


def test_tokenize_lowercases_and_removes_punctuation():
    assert tokenize("RAG-based Hallucination Detection!") == ["rag", "based", "hallucination", "detection"]


def test_build_corpus_text_includes_enriched_fields():
    text = build_corpus_text(make_record("A test paper", "LLM Evaluation"))

    assert "A paper about hallucination detection." in text
    assert "Detects hallucinations." in text


def test_normalize_title_collapses_punctuation_and_case():
    assert normalize_title("A Survey: On Hallucination!") == "a survey on hallucination"


def test_search_bm25_returns_ranked_results():
    artifact = {
        "bm25": FakeBm25(),
        "papers": [
            {"paper_id": "1", "title": "Hallucination paper", "topic": "LLM Evaluation"},
            {"paper_id": "2", "title": "Other paper", "topic": "RAG"},
        ],
    }

    results = search_bm25(artifact, "hallucination detection", top_k=1)

    assert results[0]["title"] == "Hallucination paper"
    assert results[0]["score"] == 3.0


def test_search_bm25_deduplicates_titles():
    artifact = {
        "bm25": FakeBm25(),
        "papers": [
            {"paper_id": "1", "title": "Hallucination paper", "topic": "LLM Evaluation"},
            {"paper_id": "2", "title": "Hallucination Paper!", "topic": "LLM Evaluation"},
        ],
    }

    results = search_bm25(artifact, "hallucination detection", top_k=2)

    assert len(results) == 1
