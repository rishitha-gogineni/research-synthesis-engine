"""Hybrid retrieval over dense Qdrant vectors and local BM25 scores."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from ingestion.embed import DEFAULT_EMBEDDING_MODEL, TRUNCATED_DIMENSIONS, truncate_embedding
from retrieval.build_bm25 import load_bm25_artifact, normalize_title, search_bm25
from retrieval.index_qdrant import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, get_qdrant_client, load_env_file


DEFAULT_BM25_PATH = Path("data/bm25_index.pkl")
DEFAULT_DENSE_TOP_K = 20
DEFAULT_SPARSE_TOP_K = 20
DEFAULT_FINAL_TOP_K = 10
DEFAULT_DENSE_WEIGHT = 0.65
DEFAULT_SPARSE_WEIGHT = 0.35


def embed_query(client: OpenAI, query: str, model: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    """Embed a user query with the same model and dimensions used during ingestion."""

    response = client.embeddings.create(model=model, input=query)
    return truncate_embedding(response.data[0].embedding, TRUNCATED_DIMENSIONS)


def search_dense(
    client: Any,
    collection_name: str,
    query_vector: list[float],
    top_k: int = DEFAULT_DENSE_TOP_K,
) -> list[dict[str, Any]]:
    """Search Qdrant and return payload-rich dense candidates."""

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [qdrant_point_to_candidate(point) for point in response.points]


def qdrant_point_to_candidate(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    return candidate_from_payload(payload, dense_score=float(point.score))


def candidate_from_payload(
    payload: dict[str, Any],
    dense_score: float | None = None,
    sparse_score: float | None = None,
) -> dict[str, Any]:
    return {
        "paper_id": payload.get("paper_id"),
        "title": payload.get("title"),
        "topic": payload.get("topic"),
        "year": payload.get("year"),
        "citation_count": payload.get("citation_count", 0),
        "authors": payload.get("authors", []),
        "abstract": payload.get("abstract"),
        "arxiv_id": payload.get("arxiv_id"),
        "url": payload.get("url"),
        "main_contribution": payload.get("main_contribution"),
        "methodology": payload.get("methodology"),
        "dataset_used": payload.get("dataset_used"),
        "key_result": payload.get("key_result"),
        "limitations": payload.get("limitations"),
        "dense_score": dense_score,
        "sparse_score": sparse_score,
        "hybrid_score": 0.0,
        "matched_by": [],
    }


def bm25_result_to_candidate(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    payload = {
        "paper_id": result.get("paper_id"),
        "title": result.get("title") or metadata.get("title"),
        "topic": result.get("topic") or metadata.get("topic"),
        "year": result.get("year") or metadata.get("year"),
        "citation_count": result.get("citation_count", metadata.get("citation_count", 0)),
        "authors": metadata.get("authors", []),
        "abstract": metadata.get("abstract"),
        "arxiv_id": metadata.get("arxiv_id"),
        "url": metadata.get("url"),
        "main_contribution": metadata.get("main_contribution"),
        "methodology": metadata.get("methodology"),
        "dataset_used": metadata.get("dataset_used"),
        "key_result": metadata.get("key_result"),
        "limitations": metadata.get("limitations"),
    }
    return candidate_from_payload(payload, sparse_score=float(result.get("score", 0.0)))


def candidate_key(candidate: dict[str, Any]) -> str:
    paper_id = candidate.get("paper_id")
    if paper_id:
        return f"id:{paper_id}"
    return f"title:{normalize_title(candidate.get('title'))}"


def normalize_score(value: float | None, max_score: float) -> float:
    if value is None or max_score <= 0:
        return 0.0
    return max(float(value), 0.0) / max_score


def merge_candidates(
    dense_candidates: list[dict[str, Any]],
    sparse_candidates: list[dict[str, Any]],
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
) -> list[dict[str, Any]]:
    """Merge dense and sparse candidates into one ranked list."""

    merged: dict[str, dict[str, Any]] = {}

    for candidate in dense_candidates:
        key = candidate_key(candidate)
        merged[key] = {**candidate, "matched_by": ["dense"]}

    for candidate in sparse_candidates:
        key = candidate_key(candidate)
        if key in merged:
            merged[key]["sparse_score"] = candidate.get("sparse_score")
            merged[key]["matched_by"] = sorted(set(merged[key]["matched_by"] + ["sparse"]))
            for field, value in candidate.items():
                if merged[key].get(field) is None and value is not None:
                    merged[key][field] = value
        else:
            merged[key] = {**candidate, "matched_by": ["sparse"]}

    max_dense = max((candidate.get("dense_score") or 0.0 for candidate in merged.values()), default=0.0)
    max_sparse = max((candidate.get("sparse_score") or 0.0 for candidate in merged.values()), default=0.0)

    for candidate in merged.values():
        dense_part = normalize_score(candidate.get("dense_score"), max_dense)
        sparse_part = normalize_score(candidate.get("sparse_score"), max_sparse)
        candidate["hybrid_score"] = round((dense_weight * dense_part) + (sparse_weight * sparse_part), 6)

    return sorted(
        merged.values(),
        key=lambda candidate: (
            candidate["hybrid_score"],
            candidate.get("citation_count") or 0,
            candidate.get("dense_score") or 0.0,
        ),
        reverse=True,
    )[:final_top_k]


def retrieve_papers(
    query: str,
    *,
    openai_client: OpenAI,
    qdrant_client: Any,
    bm25_artifact: dict[str, Any],
    collection_name: str = DEFAULT_COLLECTION,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
) -> list[dict[str, Any]]:
    """Retrieve candidate papers for a free-text research question."""

    if not query.strip():
        raise ValueError("query must not be empty")
    if min(dense_top_k, sparse_top_k, final_top_k) <= 0:
        raise ValueError("top-k values must be greater than 0")

    query_vector = embed_query(openai_client, query, model)
    dense_candidates = search_dense(qdrant_client, collection_name, query_vector, dense_top_k)
    sparse_candidates = [bm25_result_to_candidate(result) for result in search_bm25(bm25_artifact, query, sparse_top_k)]
    return merge_candidates(dense_candidates, sparse_candidates, final_top_k=final_top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Research question to search for.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--local-path", type=Path, default=None)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--sparse-top-k", type=int, default=DEFAULT_SPARSE_TOP_K)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before query embedding.")

    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
    openai_client = OpenAI(api_key=api_key)
    qdrant_client = get_qdrant_client(url=qdrant_url, local_path=args.local_path)
    bm25_artifact = load_bm25_artifact(args.bm25_index)

    results = retrieve_papers(
        args.query,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        bm25_artifact=bm25_artifact,
        collection_name=args.collection,
        model=args.model,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        final_top_k=args.final_top_k,
    )

    for index, result in enumerate(results, start=1):
        print(
            f"{index}. {result['hybrid_score']:.3f} | {result['title']} | "
            f"{result['topic']} | dense={result.get('dense_score')} | "
            f"bm25={result.get('sparse_score')} | matched={','.join(result['matched_by'])}"
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None
