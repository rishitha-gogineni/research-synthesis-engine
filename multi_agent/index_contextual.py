"""Index contextual embeddings into Qdrant.

Run after contextual_embeddings.py has generated the embeddings file.
Requires Qdrant to be running (docker-compose up -d qdrant).

Usage:
    python3 -m multi_agent.index_contextual --collection research_paper_chunks_contextual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from ingestion.embed import TRUNCATED_DIMENSIONS, load_env_file


DEFAULT_INPUT = Path("data/embedded_full_text_chunks_contextual.json")
DEFAULT_COLLECTION = "research_paper_chunks_contextual"
DEFAULT_QDRANT_URL = "http://localhost:6333"
BATCH_SIZE = 100


def load_embeddings(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_collection_if_needed(
    client: QdrantClient,
    collection_name: str,
    dimensions: int,
) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists, will upsert points")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dimensions,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection '{collection_name}' with {dimensions} dimensions")


def index_embeddings(
    input_path: Path = DEFAULT_INPUT,
    collection_name: str = DEFAULT_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Index contextual embeddings into Qdrant."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {input_path}. "
            "Run contextual_embeddings.py first."
        )

    records = load_embeddings(input_path)
    if not records:
        print("No embeddings to index")
        return 0

    dimensions = len(records[0]["embedding"])
    client = QdrantClient(url=qdrant_url)
    create_collection_if_needed(client, collection_name, dimensions)

    indexed = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        points = []
        for i, record in enumerate(batch, start=start):
            payload = {
                "chunk_id": record.get("chunk_id"),
                "paper_id": record.get("paper_id"),
                "title": record.get("title"),
                "topic": record.get("topic"),
                "year": record.get("year"),
                "citation_count": record.get("citation_count", 0),
                "chunk_index": record.get("chunk_index"),
                "total_chunks": record.get("total_chunks"),
                "section_hint": record.get("section_hint"),
                "word_count": record.get("word_count"),
                "context": record.get("context", ""),
                "embedding_text": record.get("embedding_text", ""),
            }
            points.append(
                PointStruct(
                    id=i,
                    vector=record["embedding"],
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        indexed += len(points)
        print(f"Indexed {indexed}/{len(records)} points")

    print(f"Done. {indexed} points indexed into '{collection_name}'")
    return indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    index_embeddings(
        input_path=args.input,
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
