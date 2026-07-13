"""Run a dense Qdrant sanity search for a free-text query."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from ingestion.embed import DEFAULT_EMBEDDING_MODEL, TRUNCATED_DIMENSIONS, truncate_embedding
from retrieval.index_qdrant import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, get_qdrant_client, load_env_file


def embed_query(client: OpenAI, query: str, model: str) -> list[float]:
    response = client.embeddings.create(model=model, input=query)
    return truncate_embedding(response.data[0].embedding, TRUNCATED_DIMENSIONS)


def search_qdrant(client: Any, collection_name: str, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "score": point.score,
            "title": point.payload.get("title"),
            "topic": point.payload.get("topic"),
            "year": point.payload.get("year"),
            "citation_count": point.payload.get("citation_count"),
            "paper_id": point.payload.get("paper_id"),
        }
        for point in response.points
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--local-path", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before query embedding.")

    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
    openai_client = OpenAI(api_key=api_key)
    qdrant_client = get_qdrant_client(url=qdrant_url, local_path=args.local_path)
    query_vector = embed_query(openai_client, args.query, args.model)

    for result in search_qdrant(qdrant_client, args.collection, query_vector, args.top_k):
        print(
            f"{result['score']:.3f} | {result['title']} | "
            f"{result['topic']} | citations={result['citation_count']}"
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None

