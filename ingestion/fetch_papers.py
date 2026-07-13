"""Fetch raw academic paper metadata from the OpenAlex Works API."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests

from shared.schemas import Paper


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "title",
        "display_name",
        "abstract_inverted_index",
        "authorships",
        "cited_by_count",
        "publication_year",
        "ids",
        "primary_location",
    ]
)

DEFAULT_TOPICS = [
    "Retrieval-Augmented Generation (RAG)",
    "Transformers / Attention Mechanisms",
    "LLM Evaluation & Hallucination Detection",
    "AI Agents & Tool Use",
    "Fine-tuning (LoRA / PEFT)",
]

DEFAULT_TOPIC_QUERIES = {
    "Retrieval-Augmented Generation (RAG)": [
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "RAG large language models",
    ],
    "Transformers / Attention Mechanisms": [
        "transformer attention",
        "attention is all you need",
        "self attention neural networks",
        "vision transformer attention",
    ],
    "LLM Evaluation & Hallucination Detection": [
        "large language model evaluation",
        "hallucination large language models",
        "hallucination detection large language models",
        "factuality large language models",
        "TruthfulQA",
        "HaluEval",
    ],
    "AI Agents & Tool Use": [
        "large language model based autonomous agents",
        "autonomous agents large language models",
        "LLM based multi agent systems",
        "large language model agents",
        "LLM agents",
        "tool learning with large language models",
        "tool learning large language models",
        "tool use large language models",
        "automatic multi-step reasoning tool-use language models",
        "generative agents interactive simulacra",
    ],
    "Fine-tuning (LoRA / PEFT)": [
        "low rank adaptation large language models",
        "LoRA low rank adaptation large language models",
        "parameter efficient fine tuning language models",
        "parameter-efficient fine-tuning language models",
        "adapter tuning language models",
        "prompt tuning language models",
    ],
}


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines from a local .env file without an extra dependency."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_headers() -> dict[str, str]:
    return {"User-Agent": "research-synthesis-engine/0.1"}


def reconstruct_abstract(abstract_inverted_index: Optional[dict[str, list[int]]]) -> Optional[str]:
    """Turn OpenAlex's abstract inverted index into regular paragraph text."""

    if not abstract_inverted_index:
        return None

    positioned_words: list[tuple[int, str]] = []
    for word, positions in abstract_inverted_index.items():
        for position in positions:
            positioned_words.append((position, word))

    if not positioned_words:
        return None

    return " ".join(word for _, word in sorted(positioned_words)).strip()


def extract_authors(authorships: list[dict[str, Any]]) -> list[str]:
    authors: list[str] = []
    for authorship in authorships:
        author = authorship.get("author") or {}
        name = (author.get("display_name") or "").strip()
        if name:
            authors.append(name)
    return authors


def extract_arxiv_id(ids: dict[str, Any]) -> Optional[str]:
    raw_arxiv = ids.get("arxiv") or ids.get("arXiv")
    if not raw_arxiv:
        return None
    return str(raw_arxiv).removeprefix("https://arxiv.org/abs/")


def extract_url(raw_work: dict[str, Any]) -> Optional[str]:
    primary_location = raw_work.get("primary_location") or {}
    landing_page_url = primary_location.get("landing_page_url")
    ids = raw_work.get("ids") or {}
    return landing_page_url or ids.get("doi") or raw_work.get("doi") or raw_work.get("id")


def normalize_paper(raw_work: dict[str, Any], topic: str) -> Optional[Paper]:
    """Convert an OpenAlex work payload into the project's raw schema.

    Works without a title or reconstructable abstract are skipped so later
    retrieval and extraction steps always have usable text.
    """

    title = (raw_work.get("title") or raw_work.get("display_name") or "").strip()
    abstract = reconstruct_abstract(raw_work.get("abstract_inverted_index"))
    if not title or not abstract:
        return None

    ids = raw_work.get("ids") or {}

    return Paper(
        paper_id=raw_work.get("id"),
        title=title,
        abstract=abstract,
        authors=extract_authors(raw_work.get("authorships") or []),
        citation_count=raw_work.get("cited_by_count") or 0,
        arxiv_id=extract_arxiv_id(ids),
        url=extract_url(raw_work),
        year=raw_work.get("publication_year"),
        topic=topic,
    )


def fetch_topic(
    topic: str,
    per_topic: int,
    session: requests.Session,
    api_key: Optional[str] = None,
    mailto: Optional[str] = None,
    existing_ids: Optional[set[str]] = None,
    sort: str = "cited_by_count:desc",
    delay_seconds: float = 0.2,
    search_limit: int = 100,
    max_pages: int = 10,
    retries: int = 2,
    retry_backoff_seconds: float = 5.0,
) -> tuple[list[Paper], int]:
    """Fetch works for one topic, returning normalized papers and skipped count."""

    candidate_by_key: dict[str, Paper] = {}
    existing_seen: set[str] = set(existing_ids or set())
    skipped_missing_text = 0
    per_page = min(search_limit, 100)
    queries = DEFAULT_TOPIC_QUERIES.get(topic, [topic])

    for query in queries:
        page = 1
        while page <= max_pages:
            params: dict[str, Any] = {
                "filter": f"title.search:{query}",
                "per_page": per_page,
                "page": page,
                "select": OPENALEX_SELECT_FIELDS,
                "sort": sort,
            }
            if api_key:
                params["api_key"] = api_key
            if mailto:
                params["mailto"] = mailto

            for attempt in range(retries + 1):
                response = session.get(
                    OPENALEX_WORKS_URL,
                    params=params,
                    headers=build_headers(),
                    timeout=30,
                )
                if response.status_code != 429 or attempt == retries:
                    break

                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else retry_backoff_seconds * (attempt + 1)
                print(f"Rate limited by OpenAlex; retrying in {wait_seconds:.1f}s")
                time.sleep(wait_seconds)

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                if response.status_code == 429:
                    raise RuntimeError(
                        "OpenAlex returned 429 Too Many Requests. Add OPENALEX_API_KEY to .env "
                        "or retry after the rate limit resets."
                    ) from exc
                if response.status_code in {401, 403}:
                    raise RuntimeError("OpenAlex rejected the request. Check OPENALEX_API_KEY in .env.") from exc
                raise

            page_results = response.json().get("results", [])
            if not page_results:
                break

            for raw_work in page_results:
                paper = normalize_paper(raw_work, topic)
                if paper is None:
                    skipped_missing_text += 1
                    continue

                dedupe_key = paper.paper_id or f"{paper.title.lower()}::{paper.year}"
                if dedupe_key in existing_seen:
                    continue

                candidate_by_key[dedupe_key] = paper

            if len(page_results) < per_page:
                break

            page += 1
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    papers = sorted(candidate_by_key.values(), key=lambda paper: paper.citation_count, reverse=True)
    return papers[:per_topic], skipped_missing_text


def write_papers(path: Path, papers: list[Paper]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [paper.model_dump() for paper in papers]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        action="append",
        help="Topic to fetch. Repeat this flag for multiple topics. Defaults to the 5 project topics.",
    )
    parser.add_argument("--per-topic", type=int, default=50, help="Number of papers to keep per topic.")
    parser.add_argument("--output", type=Path, default=Path("data/raw_papers.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key", default=None, help="OpenAlex API key. Defaults to OPENALEX_API_KEY.")
    parser.add_argument("--mailto", default=None, help="Email for OpenAlex polite usage. Defaults to OPENALEX_EMAIL.")
    parser.add_argument("--delay-seconds", type=float, default=0.2, help="Delay between paginated requests.")
    parser.add_argument("--search-limit", type=int, default=100, help="Results requested per API page.")
    parser.add_argument("--max-pages", type=int, default=2, help="Maximum OpenAlex result pages per query alias.")
    parser.add_argument(
        "--sort",
        default="cited_by_count:desc",
        help="OpenAlex sort order. Defaults to highest-cited works first.",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retries for 429 rate-limit responses.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0, help="Base wait for 429 retries.")
    parser.add_argument("--list-topics", action="store_true", help="Print the default topic list and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_topics:
        for topic in DEFAULT_TOPICS:
            print(topic)
        return

    if args.per_topic <= 0:
        raise ValueError("--per-topic must be greater than 0")

    load_env_file(args.env_file)
    api_key = args.api_key or os.getenv("OPENALEX_API_KEY")
    mailto = args.mailto or os.getenv("OPENALEX_EMAIL")
    topics = args.topic or DEFAULT_TOPICS

    all_papers: list[Paper] = []
    with requests.Session() as session:
        global_seen_ids: set[str] = set()
        for topic in topics:
            papers, skipped = fetch_topic(
                topic=topic,
                per_topic=args.per_topic,
                session=session,
                api_key=api_key,
                mailto=mailto,
                existing_ids=global_seen_ids,
                sort=args.sort,
                delay_seconds=args.delay_seconds,
                search_limit=args.search_limit,
                max_pages=args.max_pages,
                retries=args.retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
            all_papers.extend(papers)
            global_seen_ids.update(
                paper.paper_id or f"{paper.title.lower()}::{paper.year}"
                for paper in papers
            )
            print(f"{topic}: kept {len(papers)} papers, skipped {skipped} without title/abstract")

    write_papers(args.output, all_papers)
    print(f"Wrote {len(all_papers)} papers to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None
