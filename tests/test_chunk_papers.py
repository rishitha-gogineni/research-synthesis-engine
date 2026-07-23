import pytest

from full_text.chunk_papers import (
    chunk_paper,
    detect_section_hint,
    normalize_text,
    split_words,
    stable_chunk_id,
    summarize_chunks,
    word_windows,
)


def make_paper(text, status="success"):
    return {
        "paper_id": "paper-1",
        "title": "A Paper",
        "topic": "RAG",
        "year": 2024,
        "citation_count": 10,
        "source_type": "arxiv",
        "pdf_url": "https://example.org/p.pdf",
        "page_count": 2,
        "extraction_status": status,
        "text": text,
    }


def test_normalize_text_collapses_whitespace():
    assert normalize_text("A\n\n  B\tC") == "A B C"


def test_split_words_returns_normalized_words():
    assert split_words("A  B\nC") == ["A", "B", "C"]


def test_word_windows_uses_overlap():
    windows = word_windows(["a", "b", "c", "d", "e"], max_words=3, overlap_words=1)

    assert windows == [["a", "b", "c"], ["c", "d", "e"]]


def test_word_windows_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="smaller"):
        word_windows(["a"], max_words=3, overlap_words=3)


def test_detect_section_hint_finds_methodology():
    assert detect_section_hint("Our method uses a new training algorithm.") == "methodology"


def test_stable_chunk_id_is_stable():
    paper = make_paper("text")
    assert stable_chunk_id(paper, 0) == stable_chunk_id(paper, 0)


def test_chunk_paper_skips_failed_extractions():
    assert chunk_paper(make_paper("text", status="failed")) == []


def test_chunk_paper_builds_metadata():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_paper(make_paper(text), max_words=5, overlap_words=1)

    assert len(chunks) == 3
    assert chunks[0]["title"] == "A Paper"
    assert chunks[0]["word_count"] == 5
    assert chunks[0]["total_chunks"] == 3


def test_summarize_chunks_counts_topics_and_sections():
    summary = summarize_chunks([
        {"paper_id": "1", "topic": "RAG", "section_hint": "methodology"},
        {"paper_id": "1", "topic": "RAG", "section_hint": "results"},
    ])

    assert summary["chunks"] == 2
    assert summary["papers"] == 1
    assert summary["by_topic"] == {"RAG": 2}


def test_write_chunks_replaces_invalid_surrogates(tmp_path):
    from full_text.chunk_papers import write_chunks

    output = tmp_path / "chunks.json"
    write_chunks(output, [{"chunk_id": "c1", "text": "bad\ud835text"}])

    assert "bad?text" in output.read_text(encoding="utf-8")
