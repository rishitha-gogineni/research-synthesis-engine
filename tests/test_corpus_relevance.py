"""Unit tests for corpus_relevance pre-check (mocked, no network)."""

from __future__ import annotations

from types import SimpleNamespace

from multi_agent.corpus_relevance import (
    check_corpus_relevance,
    format_relevance_context,
    best_score,
    CorpusRelevance,
)


def _chunk(title, score, **kw):
    return SimpleNamespace(
        title=title, topic=kw.get("topic", "RAG"), year=kw.get("year", 2023),
        blended_score=score, rerank_score=None, hybrid_score=None, dense_score=None,
    )


def _paper(title, score, **kw):
    return SimpleNamespace(
        title=title, topic=kw.get("topic", "PEFT"), year=kw.get("year", 2021),
        blended_score=score, rerank_score=None, hybrid_score=None, dense_score=None,
    )


def _response(papers=None, chunks=None):
    return SimpleNamespace(paper_results=papers or [], chunk_results=chunks or [])


def test_best_score_prefers_blended():
    item = SimpleNamespace(blended_score=0.9, rerank_score=0.2, hybrid_score=0.1, dense_score=0.05)
    assert best_score(item) == 0.9


def test_best_score_falls_through():
    item = SimpleNamespace(blended_score=None, rerank_score=None, hybrid_score=0.42, dense_score=0.1)
    assert best_score(item) == 0.42


def test_best_score_missing_all_is_zero():
    assert best_score(SimpleNamespace()) == 0.0


def test_full_text_match_when_strong_chunk():
    r = check_corpus_relevance(
        "lora memory",
        searcher=lambda q, k: _response(chunks=[_chunk("LoRA paper", 0.71)]),
    )
    assert r.state == "full_text_match"
    assert r.matching_papers[0]["level"] == "chunk"


def test_abstract_only_when_paper_but_no_chunk():
    r = check_corpus_relevance(
        "some method",
        searcher=lambda q, k: _response(papers=[_paper("Abstract paper", 0.5)],
                                        chunks=[_chunk("weak chunk", 0.1)]),
    )
    assert r.state == "abstract_only"
    assert r.matching_papers[0]["level"] == "paper"


def test_no_match_when_all_below_threshold():
    r = check_corpus_relevance(
        "quantum recipes",
        searcher=lambda q, k: _response(papers=[_paper("p", 0.2)], chunks=[_chunk("c", 0.1)]),
    )
    assert r.state == "no_match"


def test_no_match_on_empty_corpus():
    r = check_corpus_relevance("anything", searcher=lambda q, k: _response())
    assert r.state == "no_match"


def test_searcher_exception_degrades_to_no_match():
    def boom(q, k):
        raise RuntimeError("qdrant down")
    r = check_corpus_relevance("x", searcher=boom)
    assert r.state == "no_match"
    assert "qdrant down" in r.reason


def test_threshold_is_configurable():
    r = check_corpus_relevance(
        "x",
        searcher=lambda q, k: _response(chunks=[_chunk("c", 0.4)]),
        threshold=0.85,
    )
    assert r.state == "no_match"


def test_dedup_same_title_chunks():
    r = check_corpus_relevance(
        "x",
        searcher=lambda q, k: _response(chunks=[_chunk("Same", 0.7), _chunk("Same", 0.6)]),
    )
    assert len(r.matching_papers) == 1


def test_format_context_full_text_says_only_local():
    ctx = format_relevance_context(
        CorpusRelevance(state="full_text_match", matching_papers=[{"title": "T", "year": 2022, "score": 0.7, "level": "chunk"}])
    )
    assert "ONLY local_corpus" in ctx


def test_format_context_abstract_says_parallel():
    ctx = format_relevance_context(
        CorpusRelevance(state="abstract_only", matching_papers=[{"title": "T", "year": 2022, "score": 0.5, "level": "paper"}])
    )
    assert "arxiv" in ctx and "parallel" in ctx


def test_format_context_no_match_says_skip():
    ctx = format_relevance_context(CorpusRelevance(state="no_match", reason="nothing"))
    assert "SKIP local_corpus" in ctx


def test_external_intent_detector():
    from multi_agent.lead import _needs_external_despite_corpus as needs

    # Queries that need external sources even when the corpus matches
    assert needs("What is the citation count for the original LoRA paper?")
    assert needs("How many citations does Attention Is All You Need have?")
    assert needs("Which PEFT papers have the highest citation counts?")
    assert needs("What are the latest arXiv papers on Mamba in 2026?")
    assert needs("Compare our indexed RAG papers with the latest arXiv work.")

    # Pure corpus queries should NOT bypass the cap
    assert not needs("How does LoRA reduce GPU memory during fine-tuning?")
    assert not needs("How does multi-head self-attention work?")
    assert not needs("What is the ReAct approach in LLM agents?")
