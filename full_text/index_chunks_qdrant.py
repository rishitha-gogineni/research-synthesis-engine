"""Index embedded full-text chunks into a Qdrant collection."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from retrieval.index_qdrant import DEFAULT_QDRANT_URL, VECTOR_SIZE, ensure_collection, get_qdrant_client, load_env_file


DEFAULT_INPUT = Path("data/embedded_full_text_chunks.json")
DEFAULT_COLLECTION = "research_paper_chunks"


def load_embedded_chunks(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        if len(record["embedding"]) != VECTOR_SIZE:
            raise ValueError(f"Expected {VECTOR_SIZE} dims, got {len(record['embedding'])}")
    return records


def point_id_for_chunk(record: dict[str, Any]) -> str:
    source_id = record.get("chunk_id") or f"{record.get('paper_id')}::{record.get('chunk_index')}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(source_id)))


def build_chunk_payload(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    return {
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
        "text": metadata.get("text"),
        "pdf_url": metadata.get("pdf_url"),
        "source_type": metadata.get("source_type"),
        "page_count": metadata.get("page_count"),
        "embedding_model": record.get("embedding_model"),
        "embedding_dimensions": record.get("embedding_dimensions"),
    }


def build_points(records: list[dict[str, Any]]) -> list[Any]:
    from qdrant_client.models import PointStruct

    return [
        PointStruct(
            id=point_id_for_chunk(record),
            vector=record["embedding"],
            payload=build_chunk_payload(record),
        )
        for record in records
    ]


def upsert_chunks(client: Any, collection_name: str, records: list[dict[str, Any]], batch_size: int) -> int:
    count = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        client.upsert(collection_name=collection_name, points=build_points(batch))
        count += len(batch)
        print(f"Upserted {count}/{len(records)} chunks into {collection_name}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--local-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
    records = load_embedded_chunks(args.input)
    client = get_qdrant_client(url=qdrant_url, local_path=args.local_path)
    ensure_collection(client, args.collection, VECTOR_SIZE, recreate=args.recreate)
    count = upsert_chunks(client, args.collection, records, args.batch_size)
    print(f"Qdrant chunk indexing complete: {count} chunks")


if __name__ == "__main__":
    main()
