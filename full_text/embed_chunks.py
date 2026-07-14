"""Embed full-text chunks for chunk-level vector retrieval."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from ingestion.embed import DEFAULT_EMBEDDING_MODEL, TRUNCATED_DIMENSIONS, embed_texts, load_env_file, truncate_embedding


DEFAULT_INPUT = Path("data/full_text_chunks.json")
DEFAULT_OUTPUT = Path("data/embedded_full_text_chunks.json")
MAX_EMBEDDING_TEXT_CHARS = 6000


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_embeddings(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_embeddings(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def trim_for_embedding(text: str, max_chars: int = MAX_EMBEDDING_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def build_chunk_embedding_text(chunk: dict[str, Any]) -> str:
    chunk_text = trim_for_embedding(str(chunk.get("text") or ""))
    return "\n".join(
        [
            f"Title: {chunk.get('title')}",
            f"Topic: {chunk.get('topic')}",
            f"Section hint: {chunk.get('section_hint')}",
            f"Chunk: {chunk.get('chunk_index')} of {chunk.get('total_chunks')}",
            f"Text: {chunk_text}",
        ]
    )


def build_embedding_record(chunk: dict[str, Any], full_embedding: list[float], embedding_text: str, model: str, dimensions: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "paper_id": chunk.get("paper_id"),
        "title": chunk.get("title"),
        "topic": chunk.get("topic"),
        "year": chunk.get("year"),
        "citation_count": chunk.get("citation_count", 0),
        "chunk_index": chunk.get("chunk_index"),
        "total_chunks": chunk.get("total_chunks"),
        "section_hint": chunk.get("section_hint"),
        "word_count": chunk.get("word_count"),
        "embedding_model": model,
        "full_embedding_dimensions": len(full_embedding),
        "embedding_dimensions": dimensions,
        "embedding": truncate_embedding(full_embedding, dimensions),
        "embedding_text": embedding_text,
        "metadata": chunk,
    }


def run_chunk_embedding(
    input_path: Path,
    output_path: Path,
    model: str,
    dimensions: int,
    batch_size: int,
    limit: int | None,
    force: bool,
    delay_seconds: float,
) -> tuple[int, int]:
    chunks = load_chunks(input_path)
    existing = [] if force else load_existing_embeddings(output_path)
    existing_ids = {record.get("chunk_id") for record in existing}
    candidates = [chunk for chunk in chunks if force or chunk.get("chunk_id") not in existing_ids]
    if limit is not None:
        candidates = candidates[:limit]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before running chunk embeddings.")

    client = OpenAI(api_key=api_key)
    embedded_count = 0
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        texts = [build_chunk_embedding_text(chunk) for chunk in batch]
        embeddings = embed_texts(client, model, texts)
        for chunk, text, full_embedding in zip(batch, texts, embeddings):
            existing.append(build_embedding_record(chunk, full_embedding, text, model, dimensions))
            embedded_count += 1
        write_embeddings(output_path, existing)
        print(f"embedded {embedded_count}/{len(candidates)} chunks")
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return embedded_count, len(existing) - embedded_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=int, default=TRUNCATED_DIMENSIONS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.dimensions <= 0:
        raise ValueError("--dimensions must be greater than 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")

    embedded_count, skipped_count = run_chunk_embedding(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
        delay_seconds=args.delay_seconds,
    )
    print(f"Chunk embedding complete: {embedded_count} embedded, {skipped_count} already present")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None
