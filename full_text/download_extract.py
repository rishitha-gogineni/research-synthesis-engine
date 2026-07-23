"""Download selected open PDFs and extract local full text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, BinaryIO

import requests

from ingestion.fetch_papers import build_headers


DEFAULT_INPUT = Path("data/full_text_selected.json")
DEFAULT_OUTPUT = Path("data/full_text_papers.json")
DEFAULT_PDF_DIR = Path("data/pdfs")
MIN_TEXT_CHARS = 500
PDF_SIGNATURE = b"%PDF"
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def load_selected(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_json_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [clean_json_text(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json_text(item) for key, item in value.items()}
    return value


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json_text(records), indent=2, ensure_ascii=False), encoding="utf-8")


def source_key(source: dict[str, Any]) -> str:
    value = source.get("paper_id") or source.get("title") or source.get("pdf_url") or "paper"
    return str(value)


def safe_pdf_filename(source: dict[str, Any]) -> str:
    raw_id = source_key(source).rstrip("/").split("/")[-1]
    title = source.get("title") or "paper"
    slug = SAFE_NAME_PATTERN.sub("_", title.lower()).strip("_")[:60] or "paper"
    digest = hashlib.sha1(source_key(source).encode("utf-8")).hexdigest()[:10]
    raw_id = SAFE_NAME_PATTERN.sub("_", raw_id).strip("_")[:32] or digest
    return f"{raw_id}_{digest}_{slug}.pdf"


def is_probably_pdf(content: bytes, content_type: str | None = None) -> bool:
    if content.startswith(PDF_SIGNATURE):
        return True
    return bool(content_type and "pdf" in content_type.lower() and PDF_SIGNATURE in content[:2048])


def download_pdf(
    source: dict[str, Any],
    *,
    pdf_dir: Path = DEFAULT_PDF_DIR,
    session: requests.Session | None = None,
    overwrite: bool = False,
    timeout: int = 45,
) -> Path:
    pdf_url = source.get("pdf_url")
    if not pdf_url:
        raise ValueError("source does not have a pdf_url")

    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / safe_pdf_filename(source)
    if pdf_path.exists() and not overwrite:
        return pdf_path

    active_session = session or requests.Session()
    try:
        response = active_session.get(pdf_url, headers=build_headers(), timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        content = response.content
        content_type = response.headers.get("Content-Type")
        if not is_probably_pdf(content, content_type):
            raise ValueError(f"downloaded content is not a PDF: content_type={content_type}")
        pdf_path.write_bytes(content)
        return pdf_path
    finally:
        if session is None:
            active_session.close()


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF text extraction. Install project dependencies first.") from exc

    with pdf_path.open("rb") as handle:
        return extract_text_from_pdf_file(handle)


def extract_text_from_pdf_file(handle: BinaryIO) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(handle)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.strip())
    full_text = "\n\n".join(page for page in pages if page).strip()
    return full_text, len(reader.pages)


def build_success_record(source: dict[str, Any], pdf_path: Path, text: str, page_count: int) -> dict[str, Any]:
    status = "success" if len(text) >= MIN_TEXT_CHARS else "low_text"
    return {
        **source,
        "pdf_path": str(pdf_path),
        "extraction_status": status,
        "page_count": page_count,
        "text_char_count": len(text),
        "text": text,
        "error": None,
    }


def build_failure_record(source: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {
        **source,
        "pdf_path": None,
        "extraction_status": "failed",
        "page_count": 0,
        "text_char_count": 0,
        "text": "",
        "error": str(error),
    }


def process_source(
    source: dict[str, Any],
    *,
    pdf_dir: Path = DEFAULT_PDF_DIR,
    session: requests.Session | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    try:
        pdf_path = download_pdf(source, pdf_dir=pdf_dir, session=session, overwrite=overwrite)
        text, page_count = extract_text_from_pdf(pdf_path)
        return build_success_record(source, pdf_path, text, page_count)
    except Exception as exc:  # noqa: BLE001 - keep batch extraction running and record failures.
        return build_failure_record(source, exc)


def load_existing_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {source_key(record): record for record in records}


def run_download_extract(
    sources: list[dict[str, Any]],
    *,
    output_path: Path = DEFAULT_OUTPUT,
    pdf_dir: Path = DEFAULT_PDF_DIR,
    limit: int | None = None,
    overwrite: bool = False,
    delay_seconds: float = 0.2,
    append_existing: bool = False,
) -> list[dict[str, Any]]:
    selected = sources[:limit] if limit else sources
    existing_by_key = {} if overwrite else load_existing_records(output_path)
    output_records: list[dict[str, Any]] = list(existing_by_key.values()) if append_existing else []
    active_session = requests.Session()

    try:
        for index, source in enumerate(selected, start=1):
            key = source_key(source)
            if key in existing_by_key:
                if not append_existing:
                    output_records.append(existing_by_key[key])
            else:
                record = process_source(source, pdf_dir=pdf_dir, session=active_session, overwrite=overwrite)
                output_records.append(record)
            write_records(output_path, output_records)
            if index % 10 == 0 or index == len(selected):
                successes = sum(1 for item in output_records if item["extraction_status"] in {"success", "low_text"})
                failures = sum(1 for item in output_records if item["extraction_status"] == "failed")
                print(f"processed {index}/{len(selected)} PDFs; extracted={successes}; failed={failures}")
            if delay_seconds > 0 and key not in existing_by_key:
                time.sleep(delay_seconds)
    finally:
        active_session.close()

    return output_records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_topic: dict[str, dict[str, int]] = {}
    for record in records:
        status = record.get("extraction_status") or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
        topic = record.get("topic") or "Unknown"
        topic_counts = by_topic.setdefault(topic, {"total": 0, "success_or_low_text": 0, "failed": 0})
        topic_counts["total"] += 1
        if status in {"success", "low_text"}:
            topic_counts["success_or_low_text"] += 1
        elif status == "failed":
            topic_counts["failed"] += 1
    return {
        "processed": len(records),
        "by_status": by_status,
        "by_topic": by_topic,
        "total_pages": sum(int(record.get("page_count") or 0) for record in records),
        "total_text_chars": sum(int(record.get("text_char_count") or 0) for record in records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = load_selected(args.input)
    records = run_download_extract(
        sources,
        output_path=args.output,
        pdf_dir=args.pdf_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        delay_seconds=args.delay_seconds,
        append_existing=args.append_existing,
    )
    print(json.dumps(summarize_records(records), indent=2, ensure_ascii=False))
    print(f"Wrote {len(records)} full-text records to {args.output}")


if __name__ == "__main__":
    main()
