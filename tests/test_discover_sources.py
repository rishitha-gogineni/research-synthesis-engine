from full_text.discover_sources import (
    arxiv_pdf_url,
    clean_arxiv_id,
    discover_existing_arxiv,
    discover_openalex_source,
    summarize_sources,
)


def make_record(**overrides):
    record = {
        "paper_id": "https://openalex.org/W123",
        "title": "A Paper",
        "topic": "RAG",
        "year": 2024,
        "citation_count": 42,
        "url": "https://doi.org/10.1000/example",
        "arxiv_id": None,
    }
    record.update(overrides)
    return record


def test_clean_arxiv_id_accepts_abs_and_pdf_urls():
    assert clean_arxiv_id("http://arxiv.org/abs/2106.09685") == "2106.09685"
    assert clean_arxiv_id("https://arxiv.org/pdf/2309.01219.pdf") == "2309.01219"
    assert clean_arxiv_id("not arxiv") is None


def test_arxiv_pdf_url_builds_pdf_link():
    assert arxiv_pdf_url("2106.09685") == "https://arxiv.org/pdf/2106.09685.pdf"


def test_discover_existing_arxiv_uses_local_url():
    source = discover_existing_arxiv(make_record(url="http://arxiv.org/abs/2106.09685"))

    assert source is not None
    assert source["full_text_available"] is True
    assert source["source_type"] == "arxiv"
    assert source["pdf_url"] == "https://arxiv.org/pdf/2106.09685.pdf"


def test_discover_openalex_source_prefers_arxiv_ids():
    source = discover_openalex_source(
        make_record(),
        {"ids": {"arxiv": "https://arxiv.org/abs/2401.12345"}},
    )

    assert source is not None
    assert source["source_type"] == "arxiv"
    assert source["pdf_url"] == "https://arxiv.org/pdf/2401.12345.pdf"


def test_discover_openalex_source_uses_best_oa_pdf():
    source = discover_openalex_source(
        make_record(),
        {
            "ids": {},
            "best_oa_location": {
                "pdf_url": "https://example.org/paper.pdf",
                "landing_page_url": "https://example.org/paper",
                "license": "cc-by",
                "is_oa": True,
                "source": {"host_organization_name": "Example"},
            },
        },
    )

    assert source is not None
    assert source["source_type"] == "openalex_oa"
    assert source["pdf_url"] == "https://example.org/paper.pdf"
    assert source["license"] == "cc-by"


def test_summarize_sources_counts_topics_and_source_types():
    summary = summarize_sources(
        [
            {"topic": "RAG", "full_text_available": True, "source_type": "arxiv"},
            {"topic": "RAG", "full_text_available": False, "source_type": "unavailable"},
            {"topic": "Agents", "full_text_available": True, "source_type": "openalex_oa"},
        ]
    )

    assert summary["total"] == 3
    assert summary["available"] == 2
    assert summary["by_topic"]["RAG"] == {"total": 2, "available": 1}
    assert summary["by_source_type"]["arxiv"] == 1
