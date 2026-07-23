"""Recover additional legal full-text PDFs for abstract-only papers."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from full_text.download_extract import process_source, source_key, write_records


DEFAULT_ENRICHED = Path("data/enriched_papers_final.json")
DEFAULT_EXISTING_PAPERS = Path("data/full_text_papers.json")
DEFAULT_EXISTING_CHUNKS = Path("data/full_text_chunks.json")
DEFAULT_EXISTING_SOURCES = Path("data/full_text_sources.json")
DEFAULT_OUTPUT = Path("data/full_text_papers.json")
DEFAULT_CANDIDATES_OUTPUT = Path("data/full_text_recovery_candidates.json")
DEFAULT_PDF_DIR = Path("data/pdfs")
OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_title(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def clean_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip() or None


def openalex_work_id(paper_id: str | None) -> str | None:
    if not paper_id:
        return None
    return str(paper_id).rstrip("/").split("/")[-1]


def arxiv_pdf_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", url)
    if match:
        return f"https://arxiv.org/pdf/{match.group(1)}.pdf"
    return None


def candidate_source(record: dict[str, Any], *, source_type: str, pdf_url: str, reason: str, license_value: str | None = None) -> dict[str, Any]:
    return {
        "paper_id": record.get("paper_id"),
        "title": record.get("title"),
        "topic": record.get("topic"),
        "year": record.get("year"),
        "citation_count": record.get("citation_count", 0),
        "url": record.get("url"),
        "arxiv_id": record.get("arxiv_id"),
        "full_text_available": True,
        "source_type": source_type,
        "pdf_url": pdf_url,
        "landing_page_url": None,
        "license": license_value,
        "is_oa": True,
        "reason": reason,
    }


def fetch_json(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        response = session.get(url, params=params, headers={"User-Agent": "ResearchSynthesisEngine/0.1 public-full-text-recovery"}, timeout=20)
        if response.status_code in {403, 404, 429}:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def openalex_candidates(session: requests.Session, record: dict[str, Any]) -> list[dict[str, Any]]:
    work_id = openalex_work_id(record.get("paper_id"))
    if not work_id:
        return []
    work = fetch_json(
        session,
        f"{OPENALEX_WORKS_URL}/{work_id}",
        params={"select": "id,doi,display_name,best_oa_location,primary_location,locations,ids,open_access"},
    )
    if not work:
        return []
    candidates: list[dict[str, Any]] = []
    ids = work.get("ids") or {}
    arxiv_pdf = arxiv_pdf_from_url(ids.get("arxiv"))
    if arxiv_pdf:
        candidates.append(candidate_source(record, source_type="arxiv", pdf_url=arxiv_pdf, reason="arxiv identifier from OpenAlex ids"))
    locations = []
    for key in ("best_oa_location", "primary_location"):
        if work.get(key):
            locations.append(work[key])
    locations.extend(work.get("locations") or [])
    seen = {item["pdf_url"] for item in candidates}
    for location in locations:
        pdf_url = location.get("pdf_url")
        if not pdf_url or pdf_url in seen:
            continue
        arxiv_pdf = arxiv_pdf_from_url(pdf_url) or arxiv_pdf_from_url(location.get("landing_page_url"))
        if arxiv_pdf:
            pdf_url = arxiv_pdf
            source_type = "arxiv"
        else:
            source_type = "openalex_oa"
        seen.add(pdf_url)
        candidates.append(
            candidate_source(
                record,
                source_type=source_type,
                pdf_url=pdf_url,
                reason="OpenAlex recovery pdf_url",
                license_value=location.get("license"),
            )
        )
    return candidates


def arxiv_title_candidates(session: requests.Session, record: dict[str, Any]) -> list[dict[str, Any]]:
    local_pdf = arxiv_pdf_from_url(record.get("url"))
    if local_pdf:
        return [candidate_source(record, source_type="arxiv", pdf_url=local_pdf, reason="arxiv identifier from local URL")]
    title = record.get("title") or ""
    try:
        response = session.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f'ti:"{title}"', "start": 0, "max_results": 3},
            headers={"User-Agent": "ResearchSynthesisEngine/0.1 public-full-text-recovery"},
            timeout=20,
        )
    except Exception:
        return []
    if response.status_code != 200 or "<entry>" not in response.text:
        return []
    expected = normalize_title(title)
    for entry in response.text.split("<entry>")[1:]:
        title_match = re.search(r"<title>(.*?)</title>", entry, re.S)
        id_match = re.search(r"<id>https?://arxiv.org/abs/([^<]+)</id>", entry)
        found_title = normalize_title(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else ""
        if id_match and (found_title == expected or expected in found_title or found_title in expected):
            arxiv_id = id_match.group(1).split("v")[0]
            return [candidate_source(record, source_type="arxiv", pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf", reason="arxiv title match")]
    return []


def semantic_scholar_candidates(session: requests.Session, record: dict[str, Any]) -> list[dict[str, Any]]:
    title = record.get("title") or ""
    data = fetch_json(
        session,
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": title, "limit": 3, "fields": "title,openAccessPdf,isOpenAccess,externalIds,url"},
    )
    if not data:
        return []
    expected = normalize_title(title)
    for item in data.get("data") or []:
        found_title = normalize_title(item.get("title"))
        if not found_title or not (found_title == expected or expected in found_title or found_title in expected):
            continue
        pdf_url = (item.get("openAccessPdf") or {}).get("url")
        if pdf_url:
            return [candidate_source(record, source_type="semantic_scholar_oa", pdf_url=pdf_url, reason="Semantic Scholar openAccessPdf")]
    return []


def remaining_records(enriched: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunked_ids = {chunk.get("paper_id") for chunk in chunks if chunk.get("paper_id")}
    return [record for record in enriched if record.get("paper_id") not in chunked_ids]


def build_recovery_candidates(
    records: list[dict[str, Any]],
    existing_sources: list[dict[str, Any]],
    existing_papers: list[dict[str, Any]],
    *,
    session: requests.Session,
    delay_seconds: float,
) -> list[dict[str, Any]]:
    sources_by_id = {source.get("paper_id"): source for source in existing_sources}
    papers_by_id = {paper.get("paper_id"): paper for paper in existing_papers}
    all_candidates: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        candidates: list[dict[str, Any]] = []
        existing_source = sources_by_id.get(record.get("paper_id")) or {}
        existing_paper = papers_by_id.get(record.get("paper_id")) or {}
        if existing_source.get("pdf_url") and existing_paper.get("extraction_status") == "failed":
            candidates.append({**existing_source, "reason": f"retry previous failed source: {existing_source.get('reason')}"})
        candidates.extend(arxiv_title_candidates(session, record))
        candidates.extend(openalex_candidates(session, record))
        candidates.extend(semantic_scholar_candidates(session, record))
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            pdf_url = candidate.get("pdf_url")
            if not pdf_url or pdf_url in seen:
                continue
            seen.add(pdf_url)
            candidate["recovery_candidate_rank"] = len(deduped) + 1
            deduped.append(candidate)
        all_candidates.extend(deduped)
        if index % 20 == 0 or index == len(records):
            print(f"audited {index}/{len(records)} abstract-only papers; candidates={len(all_candidates)}")
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return all_candidates


def merge_success(existing_records: list[dict[str, Any]], recovered_record: dict[str, Any]) -> list[dict[str, Any]]:
    key = source_key(recovered_record)
    merged: list[dict[str, Any]] = []
    replaced = False
    for record in existing_records:
        if source_key(record) == key:
            merged.append(recovered_record)
            replaced = True
        else:
            merged.append(record)
    if not replaced:
        merged.append(recovered_record)
    return merged


def run_recovery(
    candidates: list[dict[str, Any]],
    existing_records: list[dict[str, Any]],
    *,
    output_path: Path,
    pdf_dir: Path,
    limit_papers: int | None,
    delay_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(source_key(candidate), []).append(candidate)
    ordered_keys = list(grouped)[:limit_papers] if limit_papers else list(grouped)
    records = list(existing_records)
    attempts: list[dict[str, Any]] = []
    session = requests.Session()
    try:
        for index, key in enumerate(ordered_keys, start=1):
            success = None
            for candidate in grouped[key]:
                result = process_source(candidate, pdf_dir=pdf_dir, session=session, overwrite=True)
                attempts.append({
                    "paper_id": candidate.get("paper_id"),
                    "title": candidate.get("title"),
                    "pdf_url": candidate.get("pdf_url"),
                    "source_type": candidate.get("source_type"),
                    "status": result.get("extraction_status"),
                    "text_char_count": result.get("text_char_count"),
                    "error": result.get("error"),
                })
                if result.get("extraction_status") == "success":
                    success = result
                    break
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
            if success:
                records = merge_success(records, success)
                write_records(output_path, records)
            if index % 10 == 0 or index == len(ordered_keys):
                successes = sum(1 for item in attempts if item["status"] == "success")
                print(f"retried {index}/{len(ordered_keys)} papers; successful_attempts={successes}")
    finally:
        session.close()
    return records, attempts


def summarize(records: list[dict[str, Any]], candidates: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "successful_full_text_records": sum(1 for item in records if item.get("extraction_status") == "success"),
        "candidate_papers": len({source_key(item) for item in candidates}),
        "candidate_urls": len(candidates),
        "attempted_urls": len(attempts),
        "successful_attempts": sum(1 for item in attempts if item.get("status") == "success"),
        "attempt_statuses": dict(Counter(item.get("status") for item in attempts)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--existing-papers", type=Path, default=DEFAULT_EXISTING_PAPERS)
    parser.add_argument("--existing-chunks", type=Path, default=DEFAULT_EXISTING_CHUNKS)
    parser.add_argument("--existing-sources", type=Path, default=DEFAULT_EXISTING_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES_OUTPUT)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--limit-papers", type=int, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enriched = load_json(args.enriched)
    existing_papers = load_json(args.existing_papers)
    existing_chunks = load_json(args.existing_chunks)
    existing_sources = load_json(args.existing_sources)
    abstract_only = remaining_records(enriched, existing_chunks)
    session = requests.Session()
    try:
        candidates = build_recovery_candidates(
            abstract_only,
            existing_sources,
            existing_papers,
            session=session,
            delay_seconds=args.delay_seconds,
        )
    finally:
        session.close()
    write_json(args.candidates_output, candidates)
    if args.audit_only:
        print(json.dumps(summarize(existing_papers, candidates, []), indent=2, ensure_ascii=False))
        print(f"Wrote {len(candidates)} recovery candidate URLs to {args.candidates_output}")
        return
    records, attempts = run_recovery(
        candidates,
        existing_papers,
        output_path=args.output,
        pdf_dir=args.pdf_dir,
        limit_papers=args.limit_papers,
        delay_seconds=args.delay_seconds,
    )
    attempts_path = args.candidates_output.with_name("full_text_recovery_attempts.json")
    write_json(attempts_path, attempts)
    print(json.dumps(summarize(records, candidates, attempts), indent=2, ensure_ascii=False))
    print(f"Wrote recovered full-text records to {args.output}")
    print(f"Wrote recovery candidates to {args.candidates_output}")
    print(f"Wrote recovery attempts to {attempts_path}")


if __name__ == "__main__":
    main()
