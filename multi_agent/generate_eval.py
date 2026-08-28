"""Generate a fresh evaluation set for chunk-level retrieval.

Picks random chunks from the corpus and uses GPT-4o-mini to generate
natural queries that should retrieve each chunk. Creates a clean eval
that isn't biased toward any particular embedding style.

Usage:
    python3 -m multi_agent.generate_eval --count 50
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from openai import OpenAI

from ingestion.embed import load_env_file


DEFAULT_CHUNKS = Path("data/full_text_chunks.json")
DEFAULT_OUTPUT = Path("tests/fixtures/eval_queries_fresh.json")
GEN_MODEL = "gpt-4o-mini"


QUERY_GEN_PROMPT = """You are creating a retrieval evaluation query.

Given this chunk from a research paper, write a natural question that a researcher
would ask if they wanted to find this specific chunk.

Rules:
- Question should be 8-15 words
- Focus on the CONCEPT/FACT in the chunk, not the exact wording
- Do NOT copy phrases directly from the chunk
- The question should be answerable by the chunk content
- Vary question style: some "what", some "how", some "why", some "compare"

Paper title: {title}
Section: {section}
Chunk content:
{content}

Return JSON:
{{
  "query": "The natural question here",
  "reasoning": "Brief explanation of what the chunk answers",
  "difficulty": "easy|medium|hard"
}}"""


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_query_for_chunk(
    chunk: dict[str, Any],
    client: OpenAI,
    model: str = GEN_MODEL,
) -> dict[str, Any] | None:
    """Generate a natural query that should retrieve this chunk."""
    text = str(chunk.get("text", ""))[:1500]
    if len(text) < 200:
        return None

    prompt = QUERY_GEN_PROMPT.format(
        title=chunk.get("title", "Unknown"),
        section=chunk.get("section_hint", "unknown"),
        content=text,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        data = json.loads(response.choices[0].message.content or "{}")
        if not data.get("query"):
            return None
        return {
            "query": data["query"],
            "gold_chunk_ids": [chunk["chunk_id"]],
            "paper_id": chunk.get("paper_id"),
            "title": chunk.get("title"),
            "section": chunk.get("section_hint"),
            "difficulty": data.get("difficulty", "medium"),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception:
        return None


def run_generate_eval(
    chunks_path: Path = DEFAULT_CHUNKS,
    output_path: Path = DEFAULT_OUTPUT,
    count: int = 50,
    seed: int = 42,
    model: str = GEN_MODEL,
) -> list[dict[str, Any]]:
    """Generate a fresh eval set by sampling chunks and creating queries."""
    load_env_file(Path(".env"))
    chunks = load_chunks(chunks_path)

    # Filter chunks with substantial content
    good_chunks = [
        c for c in chunks
        if len(str(c.get("text", ""))) >= 300
        and c.get("section_hint") not in {"references", "bibliography"}
    ]

    random.seed(seed)
    random.shuffle(good_chunks)

    # Diverse sampling — pick chunks from different papers when possible
    seen_papers: set = set()
    sampled: list[dict[str, Any]] = []
    for c in good_chunks:
        pid = c.get("paper_id")
        if pid in seen_papers and len(sampled) < count // 2:
            continue
        seen_papers.add(pid)
        sampled.append(c)
        if len(sampled) >= count * 2:  # oversample, some may fail generation
            break

    client = OpenAI()
    eval_queries: list[dict[str, Any]] = []

    print(f"Generating {count} eval queries from {len(sampled)} candidate chunks...")

    for i, chunk in enumerate(sampled, 1):
        result = generate_query_for_chunk(chunk, client, model)
        if result:
            eval_queries.append(result)
            print(f"  [{len(eval_queries)}/{count}] {result['query'][:70]}")
        if len(eval_queries) >= count:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(eval_queries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nGenerated {len(eval_queries)} queries → {output_path}")
    _print_summary(eval_queries)
    return eval_queries


def _print_summary(queries: list[dict[str, Any]]) -> None:
    difficulties: dict[str, int] = {}
    sections: dict[str, int] = {}
    for q in queries:
        d = q.get("difficulty", "unknown")
        difficulties[d] = difficulties.get(d, 0) + 1
        s = q.get("section", "unknown")
        sections[s] = sections.get(s, 0) + 1
    print("\nDifficulty distribution:", difficulties)
    print("Section distribution:", sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_generate_eval(
        chunks_path=args.chunks,
        output_path=args.output,
        count=args.count,
        seed=args.seed,
    )
