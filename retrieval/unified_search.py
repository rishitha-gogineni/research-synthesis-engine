"""Route-aware unified retrieval over paper records and full-text chunks."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from pydantic import ValidationError

from full_text.index_chunks_qdrant import DEFAULT_COLLECTION as DEFAULT_CHUNK_COLLECTION
from ingestion.embed import DEFAULT_EMBEDDING_MODEL
from retrieval.build_bm25 import load_bm25_artifact
from retrieval.hybrid_search import (
    DEFAULT_BM25_PATH,
    DEFAULT_DENSE_TOP_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_SPARSE_TOP_K,
    embed_query,
    retrieve_papers,
)
from retrieval.index_qdrant import DEFAULT_COLLECTION as DEFAULT_PAPER_COLLECTION
from retrieval.index_qdrant import DEFAULT_QDRANT_URL, get_qdrant_client, load_env_file
from retrieval.rerank import rerank_and_blend
from retrieval.router import route_query
from shared.schemas import (
    QueryRoute,
    RetrievedChunk,
    RetrievedPaper,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)


PaperRetriever = Callable[..., list[dict[str, Any]]]
ChunkRetriever = Callable[..., list[dict[str, Any]]]
Reranker = Callable[..., list[dict[str, Any]]]
Router = Callable[[str], QueryRoute]


class UnifiedSearchError(RuntimeError):
    """Raised when unified retrieval cannot complete cleanly."""


def qdrant_chunk_point_to_candidate(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "chunk_id": payload.get("chunk_id"),
        "paper_id": payload.get("paper_id"),
        "title": payload.get("title") or "Untitled paper",
        "topic": payload.get("topic") or "Unknown topic",
        "year": payload.get("year"),
        "citation_count": payload.get("citation_count", 0),
        "chunk_index": payload.get("chunk_index"),
        "total_chunks": payload.get("total_chunks"),
        "section_hint": payload.get("section_hint"),
        "word_count": payload.get("word_count"),
        "text": payload.get("text") or "",
        "pdf_url": payload.get("pdf_url"),
        "source_type": payload.get("source_type"),
        "page_count": payload.get("page_count"),
        "dense_score": float(point.score),
        "matched_by": ["chunk_dense"],
    }


def search_chunks(
    qdrant_client: Any,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [qdrant_chunk_point_to_candidate(point) for point in response.points]


def retrieve_chunks(
    query: str,
    *,
    openai_client: OpenAI,
    qdrant_client: Any,
    collection_name: str = DEFAULT_CHUNK_COLLECTION,
    model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = DEFAULT_FINAL_TOP_K,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_vector = embed_query(openai_client, query, model)
    return search_chunks(qdrant_client, collection_name, query_vector, top_k)


def paper_to_schema(candidate: dict[str, Any]) -> RetrievedPaper:
    return RetrievedPaper(
        paper_id=candidate.get("paper_id"),
        title=candidate.get("title") or "Untitled paper",
        topic=candidate.get("topic") or "Unknown topic",
        year=candidate.get("year"),
        citation_count=candidate.get("citation_count") or 0,
        authors=candidate.get("authors") or [],
        abstract=candidate.get("abstract"),
        arxiv_id=candidate.get("arxiv_id"),
        url=candidate.get("url"),
        main_contribution=candidate.get("main_contribution"),
        methodology=candidate.get("methodology"),
        dataset_used=candidate.get("dataset_used"),
        key_result=candidate.get("key_result"),
        limitations=candidate.get("limitations"),
        dense_score=candidate.get("dense_score"),
        sparse_score=candidate.get("sparse_score"),
        hybrid_score=candidate.get("hybrid_score") or 0.0,
        matched_by=candidate.get("matched_by") or [],
        rerank_raw_score=candidate.get("rerank_raw_score"),
        rerank_score=candidate.get("rerank_score"),
        citation_score=candidate.get("citation_score"),
        blended_score=candidate.get("blended_score"),
        score_breakdown=candidate.get("score_breakdown"),
    )


def chunk_to_schema(candidate: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=candidate.get("chunk_id"),
        paper_id=candidate.get("paper_id"),
        title=candidate.get("title") or "Untitled paper",
        topic=candidate.get("topic") or "Unknown topic",
        year=candidate.get("year"),
        citation_count=candidate.get("citation_count") or 0,
        chunk_index=candidate.get("chunk_index"),
        total_chunks=candidate.get("total_chunks"),
        section_hint=candidate.get("section_hint"),
        word_count=candidate.get("word_count"),
        text=candidate.get("text") or "No chunk text available.",
        pdf_url=candidate.get("pdf_url"),
        source_type=candidate.get("source_type"),
        page_count=candidate.get("page_count"),
        dense_score=candidate.get("dense_score"),
        matched_by=candidate.get("matched_by") or [],
        rerank_raw_score=candidate.get("rerank_raw_score"),
        rerank_score=candidate.get("rerank_score"),
        citation_score=candidate.get("citation_score"),
        blended_score=candidate.get("blended_score"),
        score_breakdown=candidate.get("score_breakdown"),
    )


def maybe_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    enabled: bool,
    reranker: Reranker = rerank_and_blend,
    top_k: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if not enabled:
        return candidates[:top_k]
    return reranker(query, candidates, top_k=top_k)


def parse_year_filters(query: str) -> tuple[int | None, int | None]:
    lowered = query.lower()
    after = None
    before = None
    after_match = re.search(r"\bafter\s+(20\d{2}|19\d{2})", lowered)
    before_match = re.search(r"\bbefore\s+(20\d{2}|19\d{2})", lowered)
    if after_match:
        after = int(after_match.group(1))
    if before_match:
        before = int(before_match.group(1))
    return after, before


def infer_topic_filter(query: str, papers: list[dict[str, Any]]) -> str | None:
    lowered = query.lower()
    topics = sorted({
        paper.get("topic") or paper.get("metadata", {}).get("topic")
        for paper in papers
        if paper.get("topic") or paper.get("metadata", {}).get("topic")
    })
    for topic in topics:
        topic_lower = topic.lower()
        if topic_lower in lowered:
            return topic
        compact_tokens = [token for token in re.findall(r"[a-z0-9]+", topic_lower) if len(token) > 2]
        if compact_tokens and any(token in lowered for token in compact_tokens):
            return topic
    aliases = {
        "rag": "Retrieval-Augmented Generation (RAG)",
        "retrieval augmented": "Retrieval-Augmented Generation (RAG)",
        "hallucination": "LLM Evaluation & Hallucination Detection",
        "lora": "Fine-tuning (LoRA / PEFT)",
        "peft": "Fine-tuning (LoRA / PEFT)",
        "agent": "AI Agents & Tool Use",
        "agents": "AI Agents & Tool Use",
        "transformer": "Transformers / Attention Mechanisms",
        "attention": "Transformers / Attention Mechanisms",
    }
    for token, topic in aliases.items():
        if token in lowered:
            return topic
    return None


def metadata_filter_papers(query: str, bm25_artifact: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    papers = bm25_artifact.get("papers", [])
    after_year, before_year = parse_year_filters(query)
    topic_filter = infer_topic_filter(query, papers)

    candidates: list[dict[str, Any]] = []
    for paper in papers:
        metadata = paper.get("metadata", {})
        year = paper.get("year") or metadata.get("year")
        topic = paper.get("topic") or metadata.get("topic")
        if after_year is not None and (year is None or int(year) <= after_year):
            continue
        if before_year is not None and (year is None or int(year) >= before_year):
            continue
        if topic_filter and topic != topic_filter:
            continue
        candidates.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title") or metadata.get("title"),
                "topic": topic,
                "year": year,
                "citation_count": paper.get("citation_count", metadata.get("citation_count", 0)),
                "authors": metadata.get("authors", []),
                "abstract": metadata.get("abstract"),
                "arxiv_id": metadata.get("arxiv_id"),
                "url": metadata.get("url"),
                "main_contribution": metadata.get("main_contribution"),
                "methodology": metadata.get("methodology"),
                "dataset_used": metadata.get("dataset_used"),
                "key_result": metadata.get("key_result"),
                "limitations": metadata.get("limitations"),
                "hybrid_score": 0.0,
                "matched_by": ["metadata_filter"],
            }
        )

    return sorted(
        candidates,
        key=lambda candidate: (candidate.get("citation_count") or 0, candidate.get("year") or 0),
        reverse=True,
    )[:top_k]


def build_unified_response(
    request: UnifiedSearchRequest,
    route: QueryRoute,
    paper_candidates: list[dict[str, Any]],
    chunk_candidates: list[dict[str, Any]],
) -> UnifiedSearchResponse:
    papers = [paper_to_schema(candidate) for candidate in paper_candidates]
    chunks = [chunk_to_schema(candidate) for candidate in chunk_candidates]
    return UnifiedSearchResponse(
        query=request.query,
        route=route,
        paper_result_count=len(papers),
        chunk_result_count=len(chunks),
        paper_results=papers,
        chunk_results=chunks,
    )


def run_unified_search(
    query: str,
    *,
    top_k: int = DEFAULT_FINAL_TOP_K,
    paper_top_k: int | None = None,
    chunk_top_k: int | None = None,
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
    apply_reranking: bool = True,
    router: Router = route_query,
    paper_retriever: PaperRetriever = retrieve_papers,
    chunk_retriever: ChunkRetriever = retrieve_chunks,
    reranker: Reranker = rerank_and_blend,
    openai_client: Any | None = None,
    qdrant_client: Any | None = None,
    bm25_artifact: dict[str, Any] | None = None,
    paper_collection: str = DEFAULT_PAPER_COLLECTION,
    chunk_collection: str = DEFAULT_CHUNK_COLLECTION,
    model: str = DEFAULT_EMBEDDING_MODEL,
    env_file: Path = Path(".env"),
    qdrant_url: str | None = None,
    local_path: Path | None = None,
    bm25_index: Path = DEFAULT_BM25_PATH,
) -> UnifiedSearchResponse:
    try:
        request = UnifiedSearchRequest(
            query=query,
            top_k=top_k,
            paper_top_k=paper_top_k or top_k,
            chunk_top_k=chunk_top_k or top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            apply_reranking=apply_reranking,
        )
    except ValidationError as exc:
        raise UnifiedSearchError(str(exc)) from exc

    route = router(request.query)
    needs_papers = route.route in {"paper_level", "hybrid_both", "metadata_filter"}
    needs_chunks = route.route in {"chunk_level", "hybrid_both"}

    needs_vector_clients = route.route != "metadata_filter" and (needs_papers or needs_chunks)
    needs_bm25 = needs_papers

    if (needs_vector_clients and (openai_client is None or qdrant_client is None)) or (needs_bm25 and bm25_artifact is None):
        load_env_file(env_file)
        if needs_vector_clients and openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise UnifiedSearchError("OPENAI_API_KEY is missing. Add it to .env before unified retrieval.")
            openai_client = OpenAI(api_key=api_key)
        if needs_vector_clients and qdrant_client is None:
            qdrant_client = get_qdrant_client(
                url=qdrant_url or os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL,
                local_path=local_path,
            )
        if needs_bm25 and bm25_artifact is None:
            if not bm25_index.exists():
                raise UnifiedSearchError(f"BM25 index not found at {bm25_index}. Run retrieval.build_bm25 first.")
            bm25_artifact = load_bm25_artifact(bm25_index)

    paper_candidates: list[dict[str, Any]] = []
    chunk_candidates: list[dict[str, Any]] = []

    try:
        if route.route == "metadata_filter":
            paper_candidates = metadata_filter_papers(request.query, bm25_artifact or {}, request.paper_top_k)
            paper_candidates = maybe_rerank(
                request.query,
                paper_candidates,
                enabled=request.apply_reranking,
                reranker=reranker,
                top_k=request.paper_top_k,
            )
        else:
            if needs_papers:
                paper_candidates = paper_retriever(
                    request.query,
                    openai_client=openai_client,
                    qdrant_client=qdrant_client,
                    bm25_artifact=bm25_artifact,
                    collection_name=paper_collection,
                    model=model,
                    dense_top_k=request.dense_top_k,
                    sparse_top_k=request.sparse_top_k,
                    final_top_k=request.paper_top_k,
                )
                paper_candidates = maybe_rerank(
                    request.query,
                    paper_candidates,
                    enabled=request.apply_reranking,
                    reranker=reranker,
                    top_k=request.paper_top_k,
                )
            if needs_chunks:
                chunk_candidates = chunk_retriever(
                    request.query,
                    openai_client=openai_client,
                    qdrant_client=qdrant_client,
                    collection_name=chunk_collection,
                    model=model,
                    top_k=request.chunk_top_k,
                )
                chunk_candidates = maybe_rerank(
                    request.query,
                    chunk_candidates,
                    enabled=request.apply_reranking,
                    reranker=reranker,
                    top_k=request.chunk_top_k,
                )
    except Exception as exc:  # noqa: BLE001 - normalize provider/retriever errors for callers.
        raise UnifiedSearchError(f"unified retrieval failed: {exc}") from exc

    return build_unified_response(request, route, paper_candidates, chunk_candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Research question to route and retrieve for.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--paper-top-k", type=int, default=None)
    parser.add_argument("--chunk-top-k", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--sparse-top-k", type=int, default=DEFAULT_SPARSE_TOP_K)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--paper-collection", default=DEFAULT_PAPER_COLLECTION)
    parser.add_argument("--chunk-collection", default=DEFAULT_CHUNK_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--local-path", type=Path, default=None)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = run_unified_search(
        args.query,
        top_k=args.top_k,
        paper_top_k=args.paper_top_k,
        chunk_top_k=args.chunk_top_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        apply_reranking=not args.no_rerank,
        paper_collection=args.paper_collection,
        chunk_collection=args.chunk_collection,
        model=args.model,
        env_file=args.env_file,
        qdrant_url=args.qdrant_url,
        local_path=args.local_path,
        bm25_index=args.bm25_index,
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except UnifiedSearchError as exc:
        raise SystemExit(f"Error: {exc}") from None
