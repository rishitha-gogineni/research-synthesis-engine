"""Create a Qdrant collection and upsert embedded research papers."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any


DEFAULT_COLLECTION = "research_papers"
DEFAULT_QDRANT_URL = "http://localhost:6333"
VECTOR_SIZE = 1024


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_embedded_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        if len(record["embedding"]) != VECTOR_SIZE:
            raise ValueError(f"Expected {VECTOR_SIZE} dims, got {len(record['embedding'])}")
    return records


def point_id_for(record: dict[str, Any]) -> str:
    source_id = record.get("paper_id") or record["title"]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    return {
        "paper_id": record.get("paper_id"),
        "title": record["title"],
        "topic": record["topic"],
        "year": record.get("year"),
        "citation_count": record.get("citation_count", 0),
        "authors": metadata.get("authors", []),
        "abstract": metadata.get("abstract"),
        "arxiv_id": metadata.get("arxiv_id"),
        "url": metadata.get("url"),
        "main_contribution": metadata.get("main_contribution"),
        "methodology": metadata.get("methodology"),
        "dataset_used": metadata.get("dataset_used"),
        "key_result": metadata.get("key_result"),
        "limitations": metadata.get("limitations"),
        "embedding_text": record.get("embedding_text"),
        "embedding_model": record.get("embedding_model"),
        "embedding_dimensions": record.get("embedding_dimensions"),
    }


def build_points(records: list[dict[str, Any]]) -> list[Any]:
    from qdrant_client.models import PointStruct

    return [
        PointStruct(
            id=point_id_for(record),
            vector=record["embedding"],
            payload=build_payload(record),
        )
        for record in records
    ]


def get_qdrant_client(url: str | None = None, local_path: Path | None = None) -> Any:
    from qdrant_client import QdrantClient

    if local_path is not None:
        return QdrantClient(path=str(local_path))
    return QdrantClient(url=url)


def ensure_collection(client: Any, collection_name: str, vector_size: int, recreate: bool = False) -> None:
    from qdrant_client.models import Distance, VectorParams

    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_records(client: Any, collection_name: str, records: list[dict[str, Any]], batch_size: int) -> int:
    count = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        client.upsert(collection_name=collection_name, points=build_points(batch))
        count += len(batch)
        print(f"Upserted {count}/{len(records)} papers into {collection_name}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/embedded_papers.json"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--local-path", type=Path, default=None, help="Use qdrant-client local storage instead of a server URL.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
    records = load_embedded_records(args.input)
    client = get_qdrant_client(url=qdrant_url, local_path=args.local_path)
    ensure_collection(client, args.collection, VECTOR_SIZE, recreate=args.recreate)
    count = upsert_records(client, args.collection, records, args.batch_size)
    print(f"Qdrant indexing complete: {count} papers")


if __name__ == "__main__":
    main()
