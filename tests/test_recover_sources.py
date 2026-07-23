from full_text.recover_sources import (
    arxiv_pdf_from_url,
    merge_success,
    normalize_title,
    remaining_records,
    summarize,
)


def test_normalize_title_removes_case_and_punctuation():
    assert normalize_title("Attention Is All You Need!") == "attention is all you need"


def test_arxiv_pdf_from_url_handles_abs_and_pdf_links():
    assert arxiv_pdf_from_url("https://arxiv.org/abs/1706.03762") == "https://arxiv.org/pdf/1706.03762.pdf"
    assert arxiv_pdf_from_url("https://arxiv.org/pdf/2301.12345v2.pdf") == "https://arxiv.org/pdf/2301.12345.pdf"


def test_remaining_records_excludes_already_chunked_papers():
    enriched = [{"paper_id": "p1"}, {"paper_id": "p2"}]
    chunks = [{"paper_id": "p1"}]

    assert remaining_records(enriched, chunks) == [{"paper_id": "p2"}]


def test_merge_success_replaces_matching_record():
    existing = [
        {"paper_id": "p1", "title": "Old", "extraction_status": "failed"},
        {"paper_id": "p2", "title": "Keep", "extraction_status": "success"},
    ]
    recovered = {"paper_id": "p1", "title": "New", "extraction_status": "success"}

    merged = merge_success(existing, recovered)

    assert merged[0] == recovered
    assert merged[1]["paper_id"] == "p2"


def test_summarize_counts_candidate_papers_and_attempts():
    records = [{"paper_id": "p1", "extraction_status": "success"}]
    candidates = [
        {"paper_id": "p1", "pdf_url": "u1"},
        {"paper_id": "p1", "pdf_url": "u2"},
        {"paper_id": "p2", "pdf_url": "u3"},
    ]
    attempts = [{"status": "failed"}, {"status": "success"}]

    summary = summarize(records, candidates, attempts)

    assert summary["records"] == 1
    assert summary["successful_full_text_records"] == 1
    assert summary["candidate_papers"] == 2
    assert summary["candidate_urls"] == 3
    assert summary["attempt_statuses"] == {"failed": 1, "success": 1}
