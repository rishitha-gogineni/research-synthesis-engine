"""Discover legal open full-text PDF sources for the paper corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from ingestion.fetch_papers import build_headers, load_env_file


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "ids",
        "title",
        "display_name",
        "open_access",
        "best_oa_location",
        "primary_location",
        "locations",
    ]
)
ARXIV_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", re.IGNORECASE)
ARXIV_ID_PATTERN = re.compile(r"^(?P<id>\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$|^(?P<legacy>[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?(?:\.pdf)?$", re.IGNORECASE)


def load_papers(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_sources(path: Path, sources: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    raw = str(value).strip()
    match = ARXIV_PATTERN.search(raw)
    if match:
        raw = match.group(1)
    raw = raw.removeprefix("abs/").removeprefix("pdf/").removesuffix(".pdf")
    raw = raw.strip("/")
    if ARXIV_ID_PATTERN.match(raw):
        return raw
    return None


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def arxiv_abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def paper_base(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": record.get("paper_id"),
        "title": record.get("title"),
        "topic": record.get("topic"),
        "year": record.get("year"),
        "citation_count": record.get("citation_count", 0),
        "url": record.get("url"),
        "arxiv_id": record.get("arxiv_id"),
    }


def unavailable_source(record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **paper_base(record),
        "full_text_available": False,
        "source_type": "unavailable",
        "pdf_url": None,
        "landing_page_url": None,
        "license": None,
        "is_oa": False,
        "reason": reason,
    }


def arxiv_source(record: dict[str, Any], arxiv_id: str, reason: str) -> dict[str, Any]:
    return {
        **paper_base(record),
        "arxiv_id": arxiv_id,
        "full_text_available": True,
        "source_type": "arxiv",
        "pdf_url": arxiv_pdf_url(arxiv_id),
        "landing_page_url": arxiv_abs_url(arxiv_id),
        "license": None,
        "is_oa": True,
        "reason": reason,
    }


def discover_existing_arxiv(record: dict[str, Any]) -> dict[str, Any] | None:
    arxiv_id = clean_arxiv_id(record.get("arxiv_id")) or clean_arxiv_id(record.get("url"))
    if arxiv_id:
        return arxiv_source(record, arxiv_id, "existing arxiv identifier in local corpus")
    return None


def extract_openalex_work_id(paper_id: str | None) -> str | None:
    if not paper_id:
        return None
    return str(paper_id).rstrip("/").split("/")[-1]


def fetch_openalex_work(
    session: requests.Session,
    paper_id: str,
    *,
    api_key: str | None = None,
    mailto: str | None = None,
    retries: int = 2,
    retry_backoff_seconds: float = 5.0,
) -> dict[str, Any] | None:
    work_id = extract_openalex_work_id(paper_id)
    if not work_id:
        return None

    params: dict[str, Any] = {"select": OPENALEX_SELECT_FIELDS}
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto

    url = f"{OPENALEX_WORKS_URL}/{work_id}"
    for attempt in range(retries + 1):
        response = session.get(url, params=params, headers=build_headers(), timeout=30)
        if response.status_code != 429 or attempt == retries:
            break
        retry_after = response.headers.get("Retry-After")
        wait_seconds = float(retry_after) if retry_after else retry_backoff_seconds * (attempt + 1)
        time.sleep(wait_seconds)

    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def location_to_source(record: dict[str, Any], location: dict[str, Any] | None, reason: str) -> dict[str, Any] | None:
    if not location:
        return None

    pdf_url = location.get("pdf_url")
    landing_page_url = location.get("landing_page_url")
    source = location.get("source") or {}
    license_value = location.get("license")
    is_oa = bool(location.get("is_oa") or pdf_url)

    arxiv_id = clean_arxiv_id(pdf_url) or clean_arxiv_id(landing_page_url)
    if arxiv_id:
        return arxiv_source(record, arxiv_id, reason)

    if pdf_url and str(pdf_url).startswith(("http://", "https://")):
        return {
            **paper_base(record),
            "full_text_available": True,
            "source_type": "openalex_oa",
            "pdf_url": pdf_url,
            "landing_page_url": landing_page_url,
            "license": license_value,
            "is_oa": is_oa,
            "reason": reason,
            "host_organization_name": source.get("host_organization_name"),
        }

    return None


def discover_openalex_source(record: dict[str, Any], raw_work: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw_work:
        return None

    ids = raw_work.get("ids") or {}
    arxiv_id = clean_arxiv_id(ids.get("arxiv"))
    if arxiv_id:
        return arxiv_source(record, arxiv_id, "arxiv identifier from OpenAlex ids")

    for field, reason in [
        ("best_oa_location", "OpenAlex best_oa_location pdf_url"),
        ("primary_location", "OpenAlex primary_location pdf_url"),
    ]:
        source = location_to_source(record, raw_work.get(field), reason)
        if source:
            return source

    for location in raw_work.get("locations") or []:
        source = location_to_source(record, location, "OpenAlex locations pdf_url")
        if source:
            return source

    open_access = raw_work.get("open_access") or {}
    oa_url = open_access.get("oa_url")
    arxiv_id = clean_arxiv_id(oa_url)
    if arxiv_id:
        return arxiv_source(record, arxiv_id, "OpenAlex open_access oa_url")

    return None


def discover_sources(
    records: list[dict[str, Any]],
    *,
    session: requests.Session | None = None,
    api_key: str | None = None,
    mailto: str | None = None,
    delay_seconds: float = 0.05,
    use_openalex: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = records[:limit] if limit else records
    sources: list[dict[str, Any]] = []
    session_owner = session is None
    active_session = session or requests.Session()

    try:
        for index, record in enumerate(selected, start=1):
            source = discover_existing_arxiv(record)
            if source is None and use_openalex:
                raw_work = fetch_openalex_work(active_session, record.get("paper_id"), api_key=api_key, mailto=mailto)
                source = discover_openalex_source(record, raw_work)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
            if source is None:
                source = unavailable_source(record, "no open full-text PDF source found")
            sources.append(source)
            if index % 25 == 0:
                available = sum(1 for item in sources if item["full_text_available"])
                print(f"checked {index}/{len(selected)} papers; available={available}")
    finally:
        if session_owner:
            active_session.close()

    return sorted(sources, key=lambda item: (item["full_text_available"], item.get("citation_count") or 0), reverse=True)


def summarize_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    by_topic: dict[str, dict[str, int]] = {}
    by_type: dict[str, int] = {}
    for source in sources:
        topic = source.get("topic") or "Unknown"
        topic_counts = by_topic.setdefault(topic, {"total": 0, "available": 0})
        topic_counts["total"] += 1
        if source.get("full_text_available"):
            topic_counts["available"] += 1
        source_type = source.get("source_type") or "unknown"
        by_type[source_type] = by_type.get(source_type, 0) + 1
    return {
        "total": len(sources),
        "available": sum(1 for source in sources if source.get("full_text_available")),
        "unavailable": sum(1 for source in sources if not source.get("full_text_available")),
        "by_topic": by_topic,
        "by_source_type": by_type,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/enriched_papers_final.json"))
    parser.add_argument("--output", type=Path, default=Path("data/full_text_sources.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key", default=None, help="OpenAlex API key. Defaults to OPENALEX_API_KEY.")
    parser.add_argument("--mailto", default=None, help="Email for OpenAlex polite usage. Defaults to OPENALEX_EMAIL.")
    parser.add_argument("--delay-seconds", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--existing-only", action="store_true", help="Only use arXiv identifiers already present locally.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    records = load_papers(args.input)
    sources = discover_sources(
        records,
        api_key=args.api_key or os.getenv("OPENALEX_API_KEY"),
        mailto=args.mailto or os.getenv("OPENALEX_EMAIL"),
        delay_seconds=args.delay_seconds,
        use_openalex=not args.existing_only,
        limit=args.limit,
    )
    write_sources(args.output, sources)
    summary = summarize_sources(sources)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {len(sources)} source records to {args.output}")


if __name__ == "__main__":
    main()
