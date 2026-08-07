"""Tests for deterministic, route-aware candidate promotion."""

from __future__ import annotations

import math

import pytest

from retrieval.promotion import (
    DIVERSITY_CHUNKS_PER_PAPER,
    MAX_PROMOTION_BONUS,
    apply_parent_paper_cap,
    chunks_per_paper_cap,
    detect_query_intents,
    is_diversity_query,
    promote_candidates,
    promotion_bonus,
    rank_prior,
)


def make_chunk(chunk_id: str, paper_id: str, *, section_hint: str = "unknown", text: str = "") -> dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "title": "A paper",
        "section_hint": section_hint,
        "text": text,
    }


def test_detect_query_intents_finds_multiple_intents():
    intents = detect_query_intents("Which datasets and evaluation metrics are used?")

    assert "dataset" in intents
    assert "metric" in intents


def test_detect_query_intents_returns_empty_for_plain_query():
    assert detect_query_intents("Tell me about transformers") == ()


def test_is_diversity_query_matches_comparison_and_overview():
    assert is_diversity_query("What is the difference between RAG and fine-tuning?")
    assert is_diversity_query("Give an overview of parameter-efficient fine-tuning methods.")
    assert not is_diversity_query("How much does LoRA reduce GPU memory?")


def test_rank_prior_is_strictly_decreasing_and_pool_independent():
    priors = [rank_prior(rank) for rank in range(1, 21)]

    assert all(earlier > later for earlier, later in zip(priors, priors[1:]))
    assert math.isclose(rank_prior(10), 0.5)
    # The bonus budget is sized to move a candidate about five positions.
    assert math.isclose(rank_prior(10) - rank_prior(15), 0.1, abs_tol=1e-9)


def test_rank_prior_rejects_zero_rank():
    with pytest.raises(ValueError):
        rank_prior(0)


def test_promotion_bonus_is_clamped():
    candidate = make_chunk(
        "c1",
        "p1",
        section_hint="limitations",
        text="limitations of this approach include failure cases",
    )
    bonus, signals = promotion_bonus(candidate, ("limitation",), level="chunk")

    assert bonus <= MAX_PROMOTION_BONUS
    assert "section_hint:limitations" in signals
    assert "intent_vocab" in signals


def test_promotion_bonus_ignores_unknown_section_hint():
    candidate = make_chunk("c1", "p1", section_hint="unknown")
    bonus, signals = promotion_bonus(candidate, ("limitation",), level="chunk")

    assert bonus == 0.0
    assert signals == []


def test_paper_level_bonus_uses_enriched_metadata_fields():
    paper = {"paper_id": "p1", "title": "A paper", "dataset_used": "SQuAD, Natural Questions"}
    bonus, signals = promotion_bonus(paper, ("dataset",), level="paper")

    assert bonus > 0.0
    assert "paper_field_match" in signals


def test_promote_candidates_lifts_section_matched_chunk_across_the_cutoff():
    """A section-matched chunk just below rank 10 should become visible at k=10."""

    candidates = [make_chunk(f"c{index}", f"p{index}", section_hint="introduction") for index in range(1, 13)]
    candidates[11]["section_hint"] = "limitations"

    promoted = promote_candidates(
        "What are the limitations of this method?", candidates, top_k=10, level="chunk"
    )

    assert "c12" in [candidate["chunk_id"] for candidate in promoted]


def test_promote_candidates_does_not_churn_the_head_of_the_list():
    """Rank-prior gaps are wide at the top, so a single bonus cannot reorder ranks 1-3."""

    candidates = [
        make_chunk("c1", "p1", section_hint="introduction"),
        make_chunk("c2", "p2", section_hint="introduction"),
        make_chunk("c3", "p3", section_hint="limitations"),
    ]
    promoted = promote_candidates(
        "What are the limitations of this method?", candidates, top_k=3, level="chunk"
    )

    assert [candidate["chunk_id"] for candidate in promoted] == ["c1", "c2", "c3"]


def test_promote_candidates_is_pool_invariant():
    """Appending weaker candidates must not reorder the ones already present."""

    base = [make_chunk(f"c{index}", f"p{index}") for index in range(1, 6)]
    widened = base + [make_chunk(f"c{index}", f"p{index}") for index in range(6, 21)]

    narrow_order = [candidate["chunk_id"] for candidate in promote_candidates("plain query", base, top_k=5)]
    wide_order = [
        candidate["chunk_id"]
        for candidate in promote_candidates("plain query", widened, top_k=20)
    ][:5]

    assert narrow_order == wide_order


def test_promote_candidates_preserves_order_without_intent():
    candidates = [make_chunk(f"c{index}", f"p{index}") for index in range(1, 6)]
    promoted = promote_candidates("tell me about transformers", candidates, top_k=5)

    assert [candidate["chunk_id"] for candidate in promoted] == ["c1", "c2", "c3", "c4", "c5"]


def test_promote_candidates_annotates_without_requiring_rerank_fields():
    """The free-tier path has no blended_score/rerank_score; promotion must not need them."""

    candidates = [make_chunk("c1", "p1", section_hint="results")]
    promoted = promote_candidates("What results are reported?", candidates, top_k=1)

    assert "promotion_score" in promoted[0]
    assert "promotion_signals" in promoted[0]
    assert "blended_score" not in promoted[0]


def test_promote_candidates_rejects_bad_arguments():
    with pytest.raises(ValueError):
        promote_candidates("q", [make_chunk("c1", "p1")], top_k=0)
    with pytest.raises(ValueError):
        promote_candidates("q", [make_chunk("c1", "p1")], top_k=1, level="sentence")


def test_chunks_per_paper_cap_is_tight_for_comparisons_and_loose_for_evidence():
    assert chunks_per_paper_cap("difference between RAG and fine-tuning", ()) == DIVERSITY_CHUNKS_PER_PAPER
    assert chunks_per_paper_cap("which datasets are used", ("dataset",)) == 4
    assert chunks_per_paper_cap("tell me about transformers", ()) is None


def test_apply_parent_paper_cap_demotes_rather_than_drops():
    candidates = [
        make_chunk("c1", "p1"),
        make_chunk("c2", "p1"),
        make_chunk("c3", "p1"),
        make_chunk("c4", "p2"),
    ]
    capped = apply_parent_paper_cap(candidates, 2)

    assert [candidate["chunk_id"] for candidate in capped] == ["c1", "c2", "c4", "c3"]
    assert len(capped) == len(candidates)


def test_apply_parent_paper_cap_noop_when_cap_is_none():
    candidates = [make_chunk("c1", "p1"), make_chunk("c2", "p1")]

    assert apply_parent_paper_cap(candidates, None) == candidates


def test_promote_candidates_keeps_same_paper_chunks_for_evidence_queries():
    """The MMR ablation showed same-paper chunks are often the correct answer."""

    candidates = [make_chunk(f"c{index}", "p1", section_hint="results") for index in range(1, 5)]
    promoted = promote_candidates("How much does LoRA reduce GPU memory?", candidates, top_k=4)

    assert [candidate["chunk_id"] for candidate in promoted] == ["c1", "c2", "c3", "c4"]


def test_is_reading_path_query_matches_expected_patterns():
    from retrieval.promotion import is_reading_path_query

    assert is_reading_path_query("Which LoRA papers should I read first?")
    assert is_reading_path_query("What are the foundational papers to understand RAG?")
    assert is_reading_path_query("Where should a beginner start with LLM agents?")
    assert is_reading_path_query("Recommend key surveys covering autonomous agents.")
    assert is_reading_path_query("What are the most important RAG evaluation papers to study?")
    assert not is_reading_path_query("How much does LoRA reduce GPU memory?")
    assert not is_reading_path_query("What datasets are used?")


def test_reading_path_bonus_rewards_high_citation_surveys():
    from retrieval.promotion import reading_path_bonus

    survey = {"paper_id": "p1", "title": "A Survey on LLM Agents", "topic": "AI Agents", "citation_count": 1200}
    high_cite = {"paper_id": "p2", "title": "LoRA Paper", "topic": "Fine-tuning", "citation_count": 300}
    low_cite = {"paper_id": "p3", "title": "Minor Paper", "topic": "RAG", "citation_count": 50}

    assert reading_path_bonus(survey) == 0.05
    assert reading_path_bonus(high_cite) == 0.04
    assert reading_path_bonus(low_cite) == 0.0


def test_reading_path_boost_integrates_with_promote_candidates():
    papers = [
        {"paper_id": f"p{i}", "title": "Regular Paper", "topic": "RAG", "citation_count": 50}
        for i in range(1, 13)
    ]
    papers[11]["title"] = "A Survey on Retrieval-Augmented Generation"
    papers[11]["citation_count"] = 600

    promoted = promote_candidates(
        "What are the foundational papers to understand RAG?",
        papers, top_k=10, level="paper", enable_reading_path=True,
    )

    assert "p12" in [p["paper_id"] for p in promoted]


def test_affinity_bonus_fires_when_parent_retrieved():
    from retrieval.promotion import affinity_bonus

    assert affinity_bonus({"paper_id": "p1", "parent_retrieved": True}) == 0.03
    assert affinity_bonus({"paper_id": "p1"}) == 0.0


def test_promote_candidates_with_affinity_lifts_affiliated_chunk():
    candidates = [make_chunk(f"c{i}", f"p{i}") for i in range(1, 12)]
    candidates[10]["parent_retrieved"] = True

    promoted = promote_candidates(
        "plain query", candidates, top_k=10, level="chunk", enable_affinity=True,
    )

    assert "c11" in [c["chunk_id"] for c in promoted]
