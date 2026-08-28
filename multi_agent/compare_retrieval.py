"""Compare retrieval quality: original embeddings vs contextual embeddings.

Runs a set of queries against both Qdrant collections and compares Hit@10, MRR.
Requires both collections indexed in Qdrant.

Usage:
    python3 -m multi_agent.compare_retrieval --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from ingestion.embed import DEFAULT_EMBEDDING_MODEL, TRUNCATED_DIMENSIONS, load_env_file


DEFAULT_EVAL = Path("tests/fixtures/eval_queries_100_chunk_grounded.json")
ORIGINAL_COLLECTION = "research_paper_chunks"
CONTEXTUAL_COLLECTION = "research_paper_chunks_contextual"
DEFAULT_QDRANT_URL = "http://localhost:6333"


def embed_query(client: OpenAI, query: str, model: str = DEFAULT_EMBEDDING_MODEL, dimensions: int = TRUNCATED_DIMENSIONS) -> list[float]:
    response = client.embeddings.create(input=[query], model=model, dimensions=dimensions)
    return response.data[0].embedding


def search_collection(
    qdrant: QdrantClient,
    collection: str,
    vector: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    results = qdrant.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
    ).points
    return [
        {
            "title": r.payload.get("title", ""),
            "paper_id": r.payload.get("paper_id", ""),
            "chunk_id": r.payload.get("chunk_id", ""),
            "score": r.score,
            "section": r.payload.get("section_hint", ""),
        }
        for r in results
    ]


def compute_hit_at_k(results: list[dict], relevant_ids: list[str], k: int = 10) -> float:
    """1.0 if any relevant ID appears in top-k results."""
    result_ids = {r.get("paper_id", "") for r in results[:k]}
    result_ids |= {r.get("chunk_id", "") for r in results[:k]}
    return 1.0 if any(rid in result_ids for rid in relevant_ids) else 0.0


def compute_mrr(results: list[dict], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank."""
    for i, r in enumerate(results, 1):
        if r.get("paper_id", "") in relevant_ids or r.get("chunk_id", "") in relevant_ids:
            return 1.0 / i
    return 0.0


def run_comparison(
    eval_path: Path = DEFAULT_EVAL,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    limit: int | None = None,
) -> dict[str, Any]:
    """Compare original vs contextual retrieval."""
    load_env_file(Path(".env"))

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    queries = json.loads(eval_path.read_text(encoding="utf-8"))
    if limit:
        queries = queries[:limit]

    openai_client = OpenAI()
    qdrant = QdrantClient(url=qdrant_url)

    # Check collections exist
    collections = [c.name for c in qdrant.get_collections().collections]
    if ORIGINAL_COLLECTION not in collections:
        raise RuntimeError(f"Collection '{ORIGINAL_COLLECTION}' not found in Qdrant")
    if CONTEXTUAL_COLLECTION not in collections:
        raise RuntimeError(f"Collection '{CONTEXTUAL_COLLECTION}' not found in Qdrant")

    original_hits = []
    contextual_hits = []
    original_mrrs = []
    contextual_mrrs = []
    skipped = 0

    print(f"Comparing retrieval on {len(queries)} queries...")

    for i, q in enumerate(queries, 1):
        query_text = q.get("query", q.get("question", ""))
        # Prefer chunk_ids for chunk-level eval, fall back to paper_ids
        relevant_ids = q.get("gold_chunk_ids", []) or q.get("expected_relevant_ids", [])

        # Skip metadata-filter and out-of-corpus queries (not meaningful for chunk retrieval)
        route = q.get("expected_route", "")
        category = q.get("benchmark_category", "")
        if route == "metadata_filter" or category == "out_of_corpus":
            skipped += 1
            continue

        if not query_text or not relevant_ids:
            skipped += 1
            continue

        vector = embed_query(openai_client, query_text)

        # Search original
        orig_results = search_collection(qdrant, ORIGINAL_COLLECTION, vector)
        orig_hit = compute_hit_at_k(orig_results, relevant_ids)
        orig_mrr = compute_mrr(orig_results, relevant_ids)
        original_hits.append(orig_hit)
        original_mrrs.append(orig_mrr)

        # Search contextual
        ctx_results = search_collection(qdrant, CONTEXTUAL_COLLECTION, vector)
        ctx_hit = compute_hit_at_k(ctx_results, relevant_ids)
        ctx_mrr = compute_mrr(ctx_results, relevant_ids)
        contextual_hits.append(ctx_hit)
        contextual_mrrs.append(ctx_mrr)

        if i % 10 == 0:
            print(f"  Processed {i}/{len(queries)} (skipped {skipped})")

    # Compute averages
    n = len(original_hits)
    results = {
        "queries_evaluated": n,
        "original": {
            "hit_at_10": sum(original_hits) / n if n else 0,
            "mrr": sum(original_mrrs) / n if n else 0,
        },
        "contextual": {
            "hit_at_10": sum(contextual_hits) / n if n else 0,
            "mrr": sum(contextual_mrrs) / n if n else 0,
        },
        "improvement": {
            "hit_at_10_delta": (sum(contextual_hits) - sum(original_hits)) / n if n else 0,
            "mrr_delta": (sum(contextual_mrrs) - sum(original_mrrs)) / n if n else 0,
        },
    }

    # Print results
    print("\n" + "=" * 50)
    print("RETRIEVAL COMPARISON: Original vs Contextual")
    print("=" * 50)
    print(f"Queries evaluated: {n}")
    print(f"\n{'Metric':<15} {'Original':<12} {'Contextual':<12} {'Delta':<12}")
    print("-" * 50)
    print(f"{'Hit@10':<15} {results['original']['hit_at_10']:<12.3f} {results['contextual']['hit_at_10']:<12.3f} {results['improvement']['hit_at_10_delta']:+<12.3f}")
    print(f"{'MRR':<15} {results['original']['mrr']:<12.3f} {results['contextual']['mrr']:<12.3f} {results['improvement']['mrr_delta']:+<12.3f}")
    print("=" * 50)

    # Save results
    output = Path("data/retrieval_comparison.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_comparison(eval_path=args.eval, qdrant_url=args.qdrant_url, limit=args.limit)
