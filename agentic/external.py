"""HTTP adapters for external research discovery sources."""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
import threading
import xml.etree.ElementTree as ET
from typing import Any
import requests

class ExternalSearchError(RuntimeError):
    """Raised when an external provider cannot complete a search."""

@dataclass(frozen=True)
class ExternalPaper:
    source: str
    paper_id: str
    title: str
    abstract: str = ""
    url: str = ""
    published_date: str | None = None
    authors: tuple[str, ...] = ()
    citation_count: int | None = None
    relevance_score: float | None = None
    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "paper_id": self.paper_id, "title": self.title, "abstract": self.abstract, "url": self.url, "published_date": self.published_date, "authors": list(self.authors), "citation_count": self.citation_count, "relevance_score": self.relevance_score}

class ExternalSearchClient:
    """Small, synchronous client with bounded retries and provider-specific normalization."""
    ARXIV_URL = "https://export.arxiv.org/api/query"
    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    TAVILY_URL = "https://api.tavily.com/search"
    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        timeout: float = 12.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        semantic_scholar_api_key: str | None = None,
        tavily_api_key: str | None = None,
        semantic_scholar_min_interval: float = 1.1,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") if semantic_scholar_api_key is None else semantic_scholar_api_key
        self.tavily_api_key = os.getenv("TAVILY_API_KEY") if tavily_api_key is None else tavily_api_key
        self.semantic_scholar_min_interval = max(0.0, semantic_scholar_min_interval)
        self._last_semantic_scholar_request = 0.0
        self._semantic_scholar_lock = threading.Lock()

    def _throttle_semantic_scholar(self) -> None:
        with self._semantic_scholar_lock:
            elapsed = time.monotonic() - self._last_semantic_scholar_request
            delay = self.semantic_scholar_min_interval - elapsed
            if delay > 0:
                time.sleep(delay)
            self._last_semantic_scholar_request = time.monotonic()

    def _request(self, method: str, url: str, *, semantic_scholar: bool = False, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if semantic_scholar:
                self._throttle_semantic_scholar()
            try:
                response = self.session.request(method, url, **kwargs)
                status = int(getattr(response, "status_code", 200))
                if status in {408, 425, 429} or status >= 500:
                    if attempt < self.max_retries:
                        retry_after = 0.0
                        headers = getattr(response, "headers", {}) or {}
                        try:
                            retry_after = float(headers.get("Retry-After", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            retry_after = 0.0
                        time.sleep(max(self.backoff_seconds * (2**attempt), retry_after))
                        continue
                if status >= 400:
                    raise ExternalSearchError(f"provider returned HTTP {status}")
                return response
            except ExternalSearchError:
                raise
            except requests.Timeout as exc:
                # A slow/hanging provider rarely recovers within the same
                # request budget — retrying it the full max_retries times just
                # burns ~timeout-seconds per attempt for no benefit, and eats
                # into the time a fallback source could have used instead. One
                # retry (to rule out a one-off blip) is enough before bailing.
                last_error = exc
                if attempt < 1:
                    time.sleep(self.backoff_seconds * (2**attempt))
                else:
                    break
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                else:
                    break
        raise ExternalSearchError(f"provider request failed: {last_error or 'unknown error'}")

    @staticmethod
    def _validate(query: str) -> str:
        query = " ".join(query.split())
        if not query: raise ValueError("query must not be empty")
        return query
    def search_arxiv(self, query: str, max_results: int = 5) -> list[ExternalPaper]:
        query = self._validate(query)
        # arXiv's search backend has no documented default boolean for bare
        # space-separated terms within one field prefix, and in practice
        # behaves like OR — a single common token (e.g. a bare year like
        # "2026") is then enough to surface unrelated papers ahead of ones
        # that actually match every term. AND the terms explicitly so all of
        # them must match.
        terms = query.split()
        search_query = f"all:({' AND '.join(terms)})" if len(terms) > 1 else f"all:{query}"
        response = self._request("GET", self.ARXIV_URL, params={"search_query": search_query, "start": 0, "max_results": min(max_results, 20), "sortBy": "relevance"})
        try: root = ET.fromstring(response.text)
        except (ET.ParseError, AttributeError) as exc: raise ExternalSearchError("invalid Arxiv XML response") from exc
        ns = {"a": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("a:entry", ns):
            ident = (entry.findtext("a:id", "", ns) or "").strip()
            title = " ".join((entry.findtext("a:title", "", ns) or "").split())
            abstract = " ".join((entry.findtext("a:summary", "", ns) or "").split())
            authors = tuple((author.findtext("a:name", "", ns) or "").strip() for author in entry.findall("a:author", ns))
            papers.append(ExternalPaper("arxiv", ident.rsplit("/", 1)[-1], title, abstract, ident, entry.findtext("a:published", None, ns), tuple(x for x in authors if x)))
        return papers
    def search_semantic_scholar(self, query: str, max_results: int = 5) -> list[ExternalPaper]:
        query = self._validate(query)
        headers = {"x-api-key": self.semantic_scholar_api_key} if self.semantic_scholar_api_key else {}
        response = self._request("GET", self.SEMANTIC_SCHOLAR_URL, semantic_scholar=True, headers=headers, params={"query": query, "limit": min(max_results, 100), "fields": "paperId,title,abstract,authors,year,citationCount,url,publicationDate"})
        try: data = response.json()
        except ValueError as exc: raise ExternalSearchError("invalid Semantic Scholar JSON response") from exc
        papers = []
        for item in data.get("data", []):
            papers.append(ExternalPaper("semantic_scholar", item.get("paperId", ""), item.get("title", ""), item.get("abstract") or "", item.get("url") or "", item.get("publicationDate"), tuple(a.get("name", "") for a in (item.get("authors") or [])[:8]), item.get("citationCount")))
        return papers
    def search_tavily(self, query: str, max_results: int = 5) -> list[ExternalPaper]:
        query = self._validate(query)
        if not self.tavily_api_key: raise ExternalSearchError("TAVILY_API_KEY is not configured")
        response = self._request("POST", self.TAVILY_URL, json={"api_key": self.tavily_api_key, "query": query, "max_results": min(max_results, 10), "search_depth": "advanced", "include_answer": False})
        try: data = response.json()
        except ValueError as exc: raise ExternalSearchError("invalid Tavily JSON response") from exc
        return [ExternalPaper("tavily", item.get("url", ""), item.get("title", ""), (item.get("content") or "")[:2000], item.get("url", ""), item.get("published_date"), (), None, item.get("score")) for item in data.get("results", []) if item.get("url") or item.get("title")]

def _load_project_env() -> None:
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.exists(): return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip(chr(34)).strip(chr(39)))

_load_project_env()

DEFAULT_EXTERNAL_CLIENT = ExternalSearchClient()
