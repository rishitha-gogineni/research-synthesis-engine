from pathlib import Path
from types import SimpleNamespace

from full_text.download_extract import (
    build_failure_record,
    build_success_record,
    is_probably_pdf,
    safe_pdf_filename,
    source_key,
    summarize_records,
)


def make_source(**overrides):
    source = {
        "paper_id": "https://openalex.org/W123",
        "title": "A Useful Paper: Test/Case",
        "topic": "RAG",
        "citation_count": 42,
        "pdf_url": "https://example.org/paper.pdf",
    }
    source.update(overrides)
    return source


def test_source_key_prefers_paper_id():
    assert source_key(make_source()) == "https://openalex.org/W123"
    assert source_key({"title": "Fallback"}) == "Fallback"


def test_safe_pdf_filename_is_stable_and_pdf_suffix():
    first = safe_pdf_filename(make_source())
    second = safe_pdf_filename(make_source())

    assert first == second
    assert first.endswith(".pdf")
    assert "/" not in first


def test_is_probably_pdf_accepts_pdf_signature():
    assert is_probably_pdf(b"%PDF-1.7 content", "application/octet-stream") is True
    assert is_probably_pdf(b"<html></html>", "text/html") is False


def test_build_success_record_marks_low_text_for_short_extraction(tmp_path):
    source = make_source()
    record = build_success_record(source, tmp_path / "paper.pdf", "short text", page_count=2)

    assert record["extraction_status"] == "low_text"
    assert record["page_count"] == 2
    assert record["text_char_count"] == len("short text")
    assert record["error"] is None


def test_build_failure_record_preserves_source_metadata():
    record = build_failure_record(make_source(), ValueError("bad pdf"))

    assert record["title"] == "A Useful Paper: Test/Case"
    assert record["extraction_status"] == "failed"
    assert record["error"] == "bad pdf"


def test_summarize_records_counts_status_topics_pages_and_chars():
    records = [
        {"topic": "RAG", "extraction_status": "success", "page_count": 10, "text_char_count": 1000},
        {"topic": "RAG", "extraction_status": "failed", "page_count": 0, "text_char_count": 0},
        {"topic": "Agents", "extraction_status": "low_text", "page_count": 2, "text_char_count": 200},
    ]

    summary = summarize_records(records)

    assert summary["processed"] == 3
    assert summary["by_status"] == {"success": 1, "failed": 1, "low_text": 1}
    assert summary["by_topic"]["RAG"] == {"total": 2, "success_or_low_text": 1, "failed": 1}
    assert summary["total_pages"] == 12
    assert summary["total_text_chars"] == 1200



def test_run_download_extract_can_append_existing_records(monkeypatch, tmp_path):
    from full_text import download_extract

    output = tmp_path / "records.json"
    existing = [{"paper_id": "old", "title": "Old", "extraction_status": "success"}]
    download_extract.write_records(output, existing)

    def fake_process_source(source, **kwargs):
        return {**source, "extraction_status": "success", "page_count": 1, "text_char_count": 600, "text": "x" * 600}

    monkeypatch.setattr(download_extract, "process_source", fake_process_source)
    records = download_extract.run_download_extract(
        [{"paper_id": "old", "title": "Old"}, {"paper_id": "new", "title": "New"}],
        output_path=output,
        pdf_dir=tmp_path,
        append_existing=True,
        delay_seconds=0,
    )

    assert [record["paper_id"] for record in records] == ["old", "new"]
