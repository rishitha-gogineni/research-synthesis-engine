import pytest

from ingestion.extract import enrich_paper
from shared.schemas import EnrichedPaper, Paper


def make_paper() -> Paper:
    return Paper(
        paper_id="W123",
        title="A test paper",
        abstract="We propose a retrieval method and evaluate it on a QA benchmark.",
        authors=["Ada Lovelace"],
        citation_count=10,
        topic="Retrieval-Augmented Generation (RAG)",
    )


def test_enrich_paper_returns_schema_compliant_output():
    def fake_extractor(_paper):
        return {
            "main_contribution": "Proposes a retrieval method.",
            "methodology": "Evaluates the method on a QA benchmark.",
            "dataset_used": "QA benchmark",
            "key_result": "not specified",
            "limitations": "not specified",
        }

    enriched = enrich_paper(make_paper(), extractor=fake_extractor)

    assert isinstance(enriched, EnrichedPaper)
    assert enriched.main_contribution == "Proposes a retrieval method."
    assert enriched.dataset_used == "QA benchmark"
    assert enriched.limitations == "not specified"


def test_enrich_paper_retries_malformed_json_once():
    attempts = {"count": 0}

    def flaky_extractor(_paper):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return "{not valid json"
        return {
            "main_contribution": "Introduces a benchmark.",
            "methodology": "Compares model answers against references.",
            "dataset_used": "not specified",
            "key_result": "not specified",
            "limitations": "not specified",
        }

    enriched = enrich_paper(make_paper(), extractor=flaky_extractor, retries=1)

    assert attempts["count"] == 2
    assert enriched.main_contribution == "Introduces a benchmark."


def test_enrich_paper_fails_after_retry_when_required_field_missing():
    def invalid_extractor(_paper):
        return {
            "main_contribution": "Something useful.",
            "methodology": "not specified",
            "dataset_used": "not specified",
            "key_result": "not specified",
        }

    with pytest.raises(ValueError):
        enrich_paper(make_paper(), extractor=invalid_extractor, retries=1)

