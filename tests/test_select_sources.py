import pytest

from full_text.select_sources import (
    available_sources,
    group_by_topic,
    select_full_text_sources,
    summarize_selection,
)


def make_source(title, topic, citations, available=True, source_type="arxiv", pdf_url="https://example.org/p.pdf"):
    return {
        "paper_id": f"id-{title}",
        "title": title,
        "topic": topic,
        "citation_count": citations,
        "full_text_available": available,
        "source_type": source_type,
        "pdf_url": pdf_url,
    }


def test_available_sources_filters_unavailable_and_missing_pdf():
    sources = [
        make_source("A", "RAG", 10),
        make_source("B", "RAG", 20, available=False),
        make_source("C", "RAG", 30, pdf_url=None),
    ]

    available = available_sources(sources)

    assert [source["title"] for source in available] == ["A"]


def test_available_sources_deduplicates_by_paper_id():
    first = make_source("A", "RAG", 10)
    duplicate = {**first, "citation_count": 20}

    available = available_sources([first, duplicate])

    assert len(available) == 1


def test_group_by_topic_groups_sources():
    grouped = group_by_topic([make_source("A", "RAG", 10), make_source("B", "Agents", 20)])

    assert set(grouped) == {"RAG", "Agents"}
    assert grouped["RAG"][0]["title"] == "A"


def test_select_full_text_sources_keeps_top_cited_per_topic():
    sources = [
        make_source("RAG low", "RAG", 10),
        make_source("RAG high", "RAG", 100, source_type="openalex_oa"),
        make_source("Agent high", "Agents", 80),
        make_source("Agent low", "Agents", 5),
    ]

    selected = select_full_text_sources(sources, per_topic=1)

    assert [source["title"] for source in selected] == ["Agent high", "RAG high"]
    assert all(source["selected_for_full_text"] for source in selected)
    assert {source["topic_selection_rank"] for source in selected} == {1}


def test_select_full_text_sources_rejects_invalid_limits():
    with pytest.raises(ValueError, match="per_topic"):
        select_full_text_sources([], per_topic=0)
    with pytest.raises(ValueError, match="max_total"):
        select_full_text_sources([], max_total=0)


def test_summarize_selection_counts_topics_sources_and_citations():
    selected = [make_source("A", "RAG", 10), make_source("B", "Agents", 20, source_type="openalex_oa")]

    summary = summarize_selection(selected)

    assert summary["selected"] == 2
    assert summary["by_topic"] == {"RAG": 1, "Agents": 1}
    assert summary["by_source_type"] == {"arxiv": 1, "openalex_oa": 1}
    assert summary["max_citation_count"] == 20
    assert summary["min_citation_count"] == 10
