"""Compare retrieval quality across configurations.

Modes:
  --mode embedding   Compare original vs contextual embeddings (raw Qdrant)
  --mode pipeline    Evaluate chunk retrieval with BM25 fusion + cross-encoder reranking

Usage:
    python3 -m multi_agent.compare_retrieval --mode embedding --eval tests/fixtures/eval_queries_fresh.json
    python3 -m multi_agent.compare_retrieval --mode pipeline --eval tests/fixtures/eval_queries_fresh.json --qdrant-url http://localhost:6333
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient

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


def _parse_eval_query(q: dict) -> tuple[str, list[str], bool]:
    """Extract query text, relevant IDs, and whether to skip."""
    query_text = q.get("query", q.get("question", ""))
    relevant_ids = q.get("gold_chunk_ids", []) or q.get("expected_relevant_ids", [])
    route = q.get("expected_route", "")
    category = q.get("benchmark_category", "")
    skip = (
        route == "metadata_filter"
        or category == "out_of_corpus"
        or not query_text
        or not relevant_ids
    )
    return query_text, relevant_ids, skip


def run_comparison(
    eval_path: Path = DEFAULT_EVAL,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    limit: int | None = None,
) -> dict[str, Any]:
    """Compare original vs contextual retrieval (raw Qdrant vectors only)."""
    load_env_file(Path(".env"))

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    queries = json.loads(eval_path.read_text(encoding="utf-8"))
    if limit:
        queries = queries[:limit]

    openai_client = OpenAI()
    qdrant = QdrantClient(url=qdrant_url)

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
        query_text, relevant_ids, skip = _parse_eval_query(q)
        if skip:
            skipped += 1
            continue

        vector = embed_query(openai_client, query_text)

        orig_results = search_collection(qdrant, ORIGINAL_COLLECTION, vector)
        original_hits.append(compute_hit_at_k(orig_results, relevant_ids))
        original_mrrs.append(compute_mrr(orig_results, relevant_ids))

        ctx_results = search_collection(qdrant, CONTEXTUAL_COLLECTION, vector)
        contextual_hits.append(compute_hit_at_k(ctx_results, relevant_ids))
        contextual_mrrs.append(compute_mrr(ctx_results, relevant_ids))

        if i % 10 == 0:
            print(f"  Processed {i}/{len(queries)} (skipped {skipped})")

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

    print("\n" + "=" * 60)
    print("EMBEDDING COMPARISON: Original vs Contextual (raw vectors)")
    print("=" * 60)
    print(f"Queries evaluated: {n}")
    print(f"\n{'Metric':<15} {'Original':<12} {'Contextual':<12} {'Delta':<12}")
    print("-" * 60)
    print(f"{'Hit@10':<15} {results['original']['hit_at_10']:<12.3f} {results['contextual']['hit_at_10']:<12.3f} {results['improvement']['hit_at_10_delta']:+<12.3f}")
    print(f"{'MRR':<15} {results['original']['mrr']:<12.3f} {results['contextual']['mrr']:<12.3f} {results['improvement']['mrr_delta']:+<12.3f}")
    print("=" * 60)

    output = Path("data/retrieval_comparison.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output}")

    return results


def run_pipeline_eval(
    eval_path: Path = DEFAULT_EVAL,
    limit: int | None = None,
    top_k: int = 10,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    chunk_collection: str | None = None,
) -> dict[str, Any]:
    """Evaluate chunk retrieval with BM25 fusion and cross-encoder reranking.

    Uses retrieve_chunks (dense + BM25) and rerank_and_blend directly,
    bypassing paper-level retrieval which isn't needed for chunk-level eval.
    """
    from retrieval.unified_search import retrieve_chunks
    from retrieval.chunk_bm25 import load_chunk_bm25_artifact
    from retrieval.rerank import rerank_and_blend
    from retrieval.index_qdrant import get_qdrant_client

    load_env_file(Path(".env"))

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    queries = json.loads(eval_path.read_text(encoding="utf-8"))
    if limit:
        queries = queries[:limit]

    collection = chunk_collection or CONTEXTUAL_COLLECTION
    qdrant = get_qdrant_client(qdrant_url)
    openai_client = OpenAI()

    bm25_artifact = None
    try:
        bm25_artifact = load_chunk_bm25_artifact()
    except Exception:
        print("  BM25 index not found, using dense-only retrieval")

    cross_encoder = None
    has_reranker = False
    try:
        from retrieval.rerank import load_cross_encoder
        cross_encoder = load_cross_encoder()
        has_reranker = True
    except Exception:
        pass

    hits_raw = []
    mrrs_raw = []
    hits_reranked = []
    mrrs_reranked = []
    skipped = 0

    print(f"Evaluating chunk pipeline on {len(queries)} queries (top_k={top_k})...")
    print(f"  collection={collection}")
    print(f"  qdrant_url={qdrant_url}")
    print(f"  BM25={'yes' if bm25_artifact else 'no'}")
    print(f"  cross-encoder={'yes' if has_reranker else 'no'}")

    for i, q in enumerate(queries, 1):
        query_text, relevant_ids, skip = _parse_eval_query(q)
        if skip:
            skipped += 1
            continue

        try:
            candidates = retrieve_chunks(
                query_text,
                openai_client=openai_client,
                qdrant_client=qdrant,
                collection_name=collection,
                top_k=top_k * 3,
                bm25_artifact=bm25_artifact,
            )

            raw_results = [
                {"chunk_id": c.get("chunk_id", ""), "paper_id": c.get("paper_id", "")}
                for c in candidates[:top_k]
            ]
            hits_raw.append(compute_hit_at_k(raw_results, relevant_ids, k=top_k))
            mrrs_raw.append(compute_mrr(raw_results, relevant_ids))

            if has_reranker:
                reranked = rerank_and_blend(
                    query_text, candidates, top_k=top_k, model=cross_encoder
                )
            else:
                reranked = candidates[:top_k]

            reranked_results = [
                {"chunk_id": c.get("chunk_id", ""), "paper_id": c.get("paper_id", "")}
                for c in reranked[:top_k]
            ]
            hits_reranked.append(compute_hit_at_k(reranked_results, relevant_ids, k=top_k))
            mrrs_reranked.append(compute_mrr(reranked_results, relevant_ids))

        except Exception as exc:
            print(f"  [{i}] ERROR: {exc}")
            skipped += 1
            continue

        if i % 10 == 0:
            print(f"  Processed {i}/{len(queries)} (skipped {skipped})")

    n = len(hits_raw)
    results = {
        "mode": "chunk_pipeline",
        "collection": collection,
        "queries_evaluated": n,
        "skipped": skipped,
        "bm25_fusion": {
            "hit_at_10": sum(hits_raw) / n if n else 0,
            "mrr": sum(mrrs_raw) / n if n else 0,
        },
        "bm25_fusion_plus_rerank": {
            "hit_at_10": sum(hits_reranked) / n if n else 0,
            "mrr": sum(mrrs_reranked) / n if n else 0,
        },
    }

    print("\n" + "=" * 60)
    print(f"CHUNK PIPELINE EVAL ({collection})")
    print("=" * 60)
    print(f"Queries evaluated: {n}  (skipped {skipped})")
    print(f"\n{'Metric':<15} {'Dense+BM25':<14} {'+ Rerank':<14}")
    print("-" * 45)
    print(f"{'Hit@10':<15} {results['bm25_fusion']['hit_at_10']:<14.3f} {results['bm25_fusion_plus_rerank']['hit_at_10']:<14.3f}")
    print(f"{'MRR':<15} {results['bm25_fusion']['mrr']:<14.3f} {results['bm25_fusion_plus_rerank']['mrr']:<14.3f}")
    print("=" * 60)

    output = Path("data/pipeline_eval.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["embedding", "pipeline"], default="embedding",
                        help="embedding: raw vector comparison; pipeline: chunk retrieval with BM25 + reranking")
    parser.add_argument("--chunk-collection", default=None,
                        help="Override chunk collection for pipeline mode")
    args = parser.parse_args()

    if args.mode == "pipeline":
        run_pipeline_eval(
            eval_path=args.eval,
            limit=args.limit,
            qdrant_url=args.qdrant_url,
            chunk_collection=args.chunk_collection,
        )
    else:
        run_comparison(eval_path=args.eval, qdrant_url=args.qdrant_url, limit=args.limit)
