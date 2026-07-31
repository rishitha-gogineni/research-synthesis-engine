from __future__ import annotations

from retrieval.corpus_index import CorpusIndex
from retrieval.unified_search import metadata_filter_papers, run_unified_search


def make_corpus_index():
    papers = [
        {
            "paper_id": "p-bitfit",
            "title": "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models",
            "topic": "Fine-tuning (LoRA / PEFT)",
            "year": 2022,
            "citation_count": 677,
            "abstract": "BitFit updates only bias terms.",
            "main_contribution": "Introduction of BitFit.",
        },
        {
            "paper_id": "p-agent-1",
            "title": "A survey on large language model based autonomous agents",
            "topic": "AI Agents & Tool Use",
            "year": 2024,
            "citation_count": 1205,
            "abstract": "Survey of autonomous agents.",
            "main_contribution": "A comprehensive survey of LLM agents.",
        },
        {
            "paper_id": "p-agent-2",
            "title": "A review of large language models and autonomous agents in chemistry",
            "topic": "AI Agents & Tool Use",
            "year": 2024,
            "citation_count": 204,
            "abstract": "Review of LLM agents in chemistry.",
            "main_contribution": "Review of agents in chemistry.",
        },
        {
            "paper_id": "p-agent-old",
            "title": "Older survey on agents",
            "topic": "AI Agents & Tool Use",
            "year": 2023,
            "citation_count": 9999,
            "abstract": "Older survey.",
            "main_contribution": "Older survey.",
        },
    ]
    chunks = [
        {
            "chunk_id": "c-bitfit-1",
            "paper_id": "p-bitfit",
            "title": "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models",
            "topic": "Fine-tuning (LoRA / PEFT)",
            "chunk_index": 0,
            "text": "BitFit updates only bias terms during fine-tuning.",
        }
    ]
    return CorpusIndex.from_records(papers, chunks)


def test_metadata_filter_can_use_corpus_index_without_bm25_artifact():
    results = metadata_filter_papers(
        "Show highly cited AI agent survey papers published after 2023",
        None,
        top_k=5,
        corpus_index=make_corpus_index(),
    )

    assert [paper["paper_id"] for paper in results] == ["p-agent-1", "p-agent-2"]
    assert all("corpus_index" in paper["matched_by"] for paper in results)


def test_run_unified_search_paper_lookup_uses_title_index_without_vector_clients():
    response = run_unified_search(
        "Explain the BitFit paper.",
        top_k=3,
        corpus_index=make_corpus_index(),
        apply_reranking=False,
    )

    assert response.route.route == "hybrid_both"
    assert response.route.matched_signals[0].startswith("paper_lookup")
    assert response.paper_result_count == 1
    assert response.paper_results[0].paper_id == "p-bitfit"
    assert response.chunk_result_count == 1
    assert response.chunk_results[0].chunk_id == "c-bitfit-1"


def test_run_unified_search_ranked_list_uses_metadata_index_without_vector_clients():
    response = run_unified_search(
        "Show highly cited AI agent survey papers published after 2023",
        top_k=5,
        corpus_index=make_corpus_index(),
        apply_reranking=False,
    )

    assert response.route.route == "metadata_filter"
    assert [paper.paper_id for paper in response.paper_results] == ["p-agent-1", "p-agent-2"]
    assert response.chunk_result_count == 0
