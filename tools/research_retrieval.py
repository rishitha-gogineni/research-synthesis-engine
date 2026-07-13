"""Tool-style wrapper for hybrid research paper retrieval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from pydantic import ValidationError

from retrieval.hybrid_search import (
    DEFAULT_BM25_PATH,
    DEFAULT_DENSE_TOP_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_SPARSE_TOP_K,
    retrieve_papers,
)
from retrieval.index_qdrant import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, get_qdrant_client, load_env_file
from retrieval.build_bm25 import load_bm25_artifact
from shared.schemas import RetrievalRequest, RetrievalResponse, RetrievedPaper


Retriever = Callable[..., list[dict[str, Any]]]


class RetrievalToolError(RuntimeError):
    """Raised when the retrieval tool cannot be executed cleanly."""


def candidate_to_retrieved_paper(candidate: dict[str, Any]) -> RetrievedPaper:
    """Convert an internal retrieval candidate into the public tool schema."""

    return RetrievedPaper(
        paper_id=candidate.get("paper_id"),
        title=candidate.get("title") or "Untitled paper",
        topic=candidate.get("topic") or "Unknown topic",
        year=candidate.get("year"),
        citation_count=candidate.get("citation_count") or 0,
        authors=candidate.get("authors") or [],
        abstract=candidate.get("abstract"),
        arxiv_id=candidate.get("arxiv_id"),
        url=candidate.get("url"),
        main_contribution=candidate.get("main_contribution"),
        methodology=candidate.get("methodology"),
        dataset_used=candidate.get("dataset_used"),
        key_result=candidate.get("key_result"),
        limitations=candidate.get("limitations"),
        dense_score=candidate.get("dense_score"),
        sparse_score=candidate.get("sparse_score"),
        hybrid_score=candidate.get("hybrid_score") or 0.0,
        matched_by=candidate.get("matched_by") or [],
    )


def build_retrieval_response(request: RetrievalRequest, candidates: list[dict[str, Any]]) -> RetrievalResponse:
    results = [candidate_to_retrieved_paper(candidate) for candidate in candidates]
    return RetrievalResponse(query=request.query, result_count=len(results), results=results)


def load_default_dependencies(
    *,
    env_file: Path = Path(".env"),
    qdrant_url: str | None = None,
    local_path: Path | None = None,
    bm25_index: Path = DEFAULT_BM25_PATH,
) -> dict[str, Any]:
    """Load live dependencies for the retrieval tool."""

    load_env_file(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RetrievalToolError("OPENAI_API_KEY is missing. Add it to .env before retrieval.")
    if not bm25_index.exists():
        raise RetrievalToolError(f"BM25 index not found at {bm25_index}. Run retrieval.build_bm25 first.")

    return {
        "openai_client": OpenAI(api_key=api_key),
        "qdrant_client": get_qdrant_client(url=qdrant_url or os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL, local_path=local_path),
        "bm25_artifact": load_bm25_artifact(bm25_index),
    }


def run_research_retrieval(
    query: str,
    *,
    top_k: int = DEFAULT_FINAL_TOP_K,
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
    collection_name: str = DEFAULT_COLLECTION,
    retriever: Retriever = retrieve_papers,
    openai_client: Any | None = None,
    qdrant_client: Any | None = None,
    bm25_artifact: dict[str, Any] | None = None,
    env_file: Path = Path(".env"),
    qdrant_url: str | None = None,
    local_path: Path | None = None,
    bm25_index: Path = DEFAULT_BM25_PATH,
) -> RetrievalResponse:
    """Run hybrid retrieval and return a schema-validated response."""

    try:
        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
        )
    except ValidationError as exc:
        raise RetrievalToolError(str(exc)) from exc

    if openai_client is None or qdrant_client is None or bm25_artifact is None:
        dependencies = load_default_dependencies(
            env_file=env_file,
            qdrant_url=qdrant_url,
            local_path=local_path,
            bm25_index=bm25_index,
        )
        openai_client = openai_client or dependencies["openai_client"]
        qdrant_client = qdrant_client or dependencies["qdrant_client"]
        bm25_artifact = bm25_artifact or dependencies["bm25_artifact"]

    try:
        candidates = retriever(
            request.query,
            openai_client=openai_client,
            qdrant_client=qdrant_client,
            bm25_artifact=bm25_artifact,
            collection_name=collection_name,
            dense_top_k=request.dense_top_k,
            sparse_top_k=request.sparse_top_k,
            final_top_k=request.top_k,
        )
    except Exception as exc:  # noqa: BLE001 - convert provider/client errors into one tool-facing error.
        raise RetrievalToolError(f"retrieval failed: {exc}") from exc

    return build_retrieval_response(request, candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Research question to retrieve papers for.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--sparse-top-k", type=int, default=DEFAULT_SPARSE_TOP_K)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--local-path", type=Path, default=None)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = run_research_retrieval(
        args.query,
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        collection_name=args.collection,
        env_file=args.env_file,
        qdrant_url=args.qdrant_url,
        local_path=args.local_path,
        bm25_index=args.bm25_index,
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except RetrievalToolError as exc:
        raise SystemExit(f"Error: {exc}") from None
