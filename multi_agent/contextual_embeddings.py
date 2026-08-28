"""Contextual embeddings — prepend document-level context to chunks before embedding.

Based on Anthropic's Contextual Retrieval technique:
For each chunk, generate a brief description situating it within the full document,
then prepend this context before embedding. This improves retrieval by ~35%.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from ingestion.embed import (
    DEFAULT_EMBEDDING_MODEL,
    TRUNCATED_DIMENSIONS,
    embed_texts,
    load_env_file,
)
from full_text.embed_chunks import (
    DEFAULT_INPUT,
    MAX_EMBEDDING_TEXT_CHARS,
    load_chunks,
    load_existing_embeddings,
    write_embeddings,
    trim_for_embedding,
)


DEFAULT_CONTEXTUAL_OUTPUT = Path("data/embedded_full_text_chunks_contextual.json")
CONTEXT_MODEL = "gpt-4o-mini"


def _load_papers_by_id(papers_path: Path) -> dict[str, dict[str, Any]]:
    """Load papers keyed by paper_id for document-level context."""
    if not papers_path.exists():
        return {}
    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    return {p.get("paper_id", p.get("id", "")): p for p in papers}


def generate_chunk_context(
    chunk: dict[str, Any],
    paper: dict[str, Any] | None,
    client: OpenAI,
    model: str = CONTEXT_MODEL,
) -> str:
    """Generate a 1-2 sentence context for a chunk within its document."""
    title = chunk.get("title", "Unknown")
    section = chunk.get("section_hint", "unknown section")
    chunk_text = str(chunk.get("text", ""))[:500]

    paper_abstract = ""
    if paper:
        paper_abstract = paper.get("abstract", "")[:300]

    prompt = f"""Given this paper and chunk, write a 1-2 sentence context that situates
this chunk within the document. Be specific about what this chunk discusses.

Paper title: {title}
Paper abstract: {paper_abstract}
Section: {section}
Chunk text (first 500 chars): {chunk_text}

Context (1-2 sentences only):"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return f"This chunk is from the {section} section of '{title}'."


def build_contextual_embedding_text(
    chunk: dict[str, Any],
    context: str,
) -> str:
    """Build embedding text with contextual description prepended."""
    chunk_text = trim_for_embedding(str(chunk.get("text") or ""))
    return "\n".join(
        [
            f"Context: {context}",
            f"Title: {chunk.get('title')}",
            f"Topic: {chunk.get('topic')}",
            f"Section: {chunk.get('section_hint')}",
            f"Text: {chunk_text}",
        ]
    )


def run_contextual_embedding(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_CONTEXTUAL_OUTPUT,
    papers_path: Path = Path("data/enriched_papers_final.json"),
    model: str = DEFAULT_EMBEDDING_MODEL,
    context_model: str = CONTEXT_MODEL,
    dimensions: int = TRUNCATED_DIMENSIONS,
    batch_size: int = 32,
    limit: int | None = None,
    force: bool = False,
    delay_seconds: float = 0.2,
) -> tuple[int, int]:
    """Run contextual embedding pipeline.

    For each chunk:
    1. Generate contextual description using LLM
    2. Prepend context to chunk text
    3. Embed the contextual text
    """
    chunks = load_chunks(input_path)
    existing = [] if force else load_existing_embeddings(output_path)
    existing_ids = {r.get("chunk_id") for r in existing}
    candidates = [c for c in chunks if force or c.get("chunk_id") not in existing_ids]
    if limit is not None:
        candidates = candidates[:limit]

    papers_by_id = _load_papers_by_id(papers_path)
    client = OpenAI()

    # Group chunks by paper_id for prompt caching efficiency
    paper_groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in candidates:
        pid = chunk.get("paper_id", "unknown")
        paper_groups.setdefault(pid, []).append(chunk)

    embedded_count = 0
    for paper_id, paper_chunks in paper_groups.items():
        paper = papers_by_id.get(paper_id)

        for batch_start in range(0, len(paper_chunks), batch_size):
            batch = paper_chunks[batch_start : batch_start + batch_size]

            # Generate contexts for batch
            contexts = []
            for chunk in batch:
                context = generate_chunk_context(chunk, paper, client, context_model)
                contexts.append(context)

            # Build contextual embedding texts
            texts = [
                build_contextual_embedding_text(chunk, ctx)
                for chunk, ctx in zip(batch, contexts)
            ]

            # Embed
            embeddings = embed_texts(client, model, texts, dimensions=dimensions)

            for chunk, text, ctx, embedding in zip(batch, texts, contexts, embeddings):
                record = {
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
                    "context": ctx,
                    "embedding_model": model,
                    "embedding_dimensions": dimensions,
                    "embedding": embedding,
                    "embedding_text": text,
                    "metadata": chunk,
                }
                existing.append(record)
                embedded_count += 1

            write_embeddings(output_path, existing)
            print(f"Contextual embedding: {embedded_count}/{len(candidates)} chunks")

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    return embedded_count, len(existing) - embedded_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONTEXTUAL_OUTPUT)
    parser.add_argument("--papers", type=Path, default=Path("data/enriched_papers_final.json"))
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--context-model", default=CONTEXT_MODEL)
    parser.add_argument("--dimensions", type=int, default=TRUNCATED_DIMENSIONS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)

    embedded_count, skipped_count = run_contextual_embedding(
        input_path=args.input,
        output_path=args.output,
        papers_path=args.papers,
        model=args.model,
        context_model=args.context_model,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
        delay_seconds=args.delay_seconds,
    )
    print(f"Contextual embedding complete: {embedded_count} embedded, {skipped_count} skipped")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from None
