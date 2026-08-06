import math

import pytest

from retrieval.rerank import (
    agent_task_intent_boost,
    apply_citation_blended_scores,
    apply_query_intent_boosts,
    attach_rerank_scores,
    candidate_to_text,
    compress_text_for_query,
    normalize_scores,
    normalize_scores_batch_relative,
    normalized_citation_scores,
    fallback_retrieval_scores,
    rerank_and_blend,
    score_with_cross_encoder,
    split_into_sentences,
)


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


def make_candidates():
    return [
        {
            "paper_id": "paper-1",
            "title": "Hallucination Detection",
            "topic": "LLM Evaluation & Hallucination Detection",
            "abstract": "Detects unsupported generated claims.",
            "citation_count": 10,
        },
        {
            "paper_id": "paper-2",
            "title": "High Citation Survey",
            "topic": "LLM Evaluation & Hallucination Detection",
            "abstract": "Survey of hallucination benchmarks.",
            "citation_count": 1000,
        },
        {
            "chunk_id": "chunk-1",
            "paper_id": "paper-3",
            "title": "Benchmark Details",
            "section_hint": "experiments",
            "text": "TruthfulQA and HaluEval are used as hallucination evaluation benchmarks.",
            "citation_count": 50,
        },
    ]


def test_candidate_to_text_supports_paper_and_chunk_candidates():
    text = candidate_to_text(make_candidates()[2])

    assert "Benchmark Details" in text
    assert "experiments" in text
    assert "TruthfulQA" in text


def test_candidate_to_text_truncates_long_text():
    text = candidate_to_text({"title": "A", "text": "x" * 100}, max_chars=10)

    assert len(text) == 10


def test_split_into_sentences_splits_on_terminal_punctuation():
    sentences = split_into_sentences("First sentence. Second sentence! Third one?")

    assert sentences == ["First sentence.", "Second sentence!", "Third one?"]


def test_compress_text_for_query_returns_text_unchanged_when_it_fits():
    text = "Short chunk about datasets."
    assert compress_text_for_query(text, "datasets", budget_chars=1000) == text


def test_compress_text_for_query_prefers_query_relevant_sentences_over_prefix():
    text = (
        "This paper introduces a new transformer architecture for vision tasks. "
        "We build on prior work in convolutional networks. "
        "The dataset used for evaluation is TruthfulQA, a hallucination benchmark. "
        "Training used eight GPUs for three days. "
        "Related work includes several attention-based baselines."
    )
    # A naive prefix truncation at this budget would cut off before the
    # dataset sentence entirely; query-aware compression should keep it.
    compressed = compress_text_for_query(text, "what dataset was used for evaluation", budget_chars=90)

    assert "TruthfulQA" in compressed
    assert len(compressed) <= 90


def test_compress_text_for_query_falls_back_to_prefix_when_no_term_overlap():
    text = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima."
    compressed = compress_text_for_query(text, "zzz unrelated query terms", budget_chars=20)

    assert compressed == text[:20]


def test_compress_text_for_query_falls_back_to_prefix_for_single_sentence_text():
    text = "One single very long sentence without any terminal punctuation at all"
    compressed = compress_text_for_query(text, "sentence", budget_chars=15)

    assert compressed == text[:15]


def test_candidate_to_text_uses_query_aware_compression_for_long_chunks():
    long_text = (
        "Introductory filler sentence about background context that is not relevant. "
        "Another filler sentence padding out the passage even more than before. "
        "The key result reported is a 12 percent improvement on the benchmark. "
        "Yet more filler discussing unrelated prior work in the field."
    )
    candidate = {"title": "A Paper", "text": long_text}

    without_query = candidate_to_text(candidate, max_chars=60)
    with_query = candidate_to_text(candidate, max_chars=60, query="what was the key result reported")

    # Without a query, behavior is unchanged: a blind prefix cut that misses the result sentence.
    assert "12 percent" not in without_query
    # With a query, compression should surface the sentence that actually answers it.
    assert "12 percent" in with_query


def test_candidate_to_text_query_path_matches_no_query_path_when_text_already_fits():
    candidate = {"title": "A", "topic": "RAG", "text": "Short fitting text."}

    assert candidate_to_text(candidate, query="anything") == candidate_to_text(candidate)


def test_normalize_scores_uses_fixed_scale_range():
    assert normalize_scores([-10.0, 0.0, 10.0]) == [0.0, 0.5, 1.0]
    assert normalize_scores([-20.0, 20.0]) == [0.0, 1.0]
    assert normalize_scores([]) == []


def test_normalize_scores_batch_relative_keeps_original_behavior():
    assert normalize_scores_batch_relative([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert normalize_scores_batch_relative([3.0, 3.0]) == [1.0, 1.0]
    assert normalize_scores_batch_relative([]) == []


def test_normalized_citation_scores_uses_log_scale():
    candidates = [{"citation_count": 0}, {"citation_count": 9}, {"citation_count": 99}]
    scores = normalized_citation_scores(candidates)

    assert scores[0] == 0.0
    assert 0.0 < scores[1] < scores[2]
    # Scaling is against a fixed reference, so the batch maximum is not 1.0.
    assert scores[2] < 1.0


def test_normalized_citation_scores_clip_at_the_reference():
    scores = normalized_citation_scores(
        [{"citation_count": 10_000}, {"citation_count": 500_000}],
        reference_citations=10_000,
    )

    assert math.isclose(scores[0], 1.0)
    assert scores[1] == 1.0


def test_normalized_citation_scores_are_pool_invariant():
    """Adding a very highly cited paper must not rescale the others."""

    narrow = normalized_citation_scores([{"citation_count": 10}, {"citation_count": 100}])
    widened = normalized_citation_scores(
        [{"citation_count": 10}, {"citation_count": 100}, {"citation_count": 50_000}]
    )

    assert narrow == widened[:2]


def test_score_with_cross_encoder_uses_query_candidate_pairs():
    model = FakeCrossEncoder([0.2, 0.8])
    candidates = make_candidates()[:2]

    scores = score_with_cross_encoder("hallucination benchmarks", candidates, model=model)

    assert scores == [0.2, 0.8]
    assert model.pairs[0][0] == "hallucination benchmarks"
    assert "Hallucination Detection" in model.pairs[0][1]


def test_score_with_cross_encoder_rejects_blank_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        score_with_cross_encoder(" ", make_candidates(), model=FakeCrossEncoder([]))


def test_attach_rerank_scores_sorts_by_normalized_score_without_mutating_input():
    candidates = make_candidates()[:2]
    reranked = attach_rerank_scores(candidates, [-10.0, 10.0])

    assert reranked[0]["paper_id"] == "paper-2"
    assert reranked[0]["rerank_score"] == 1.0
    assert "rerank_score" not in candidates[0]


def test_attach_rerank_scores_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        attach_rerank_scores(make_candidates(), [0.1])


def test_apply_citation_blended_scores_adds_breakdown_and_sorts():
    candidates = [
        {"paper_id": "relevant", "rerank_score": 1.0, "citation_count": 10},
        {"paper_id": "cited", "rerank_score": 0.6, "citation_count": 10000},
    ]

    results = apply_citation_blended_scores(candidates)

    assert results[0]["paper_id"] == "relevant"
    assert results[0]["blended_score"] > results[1]["blended_score"]
    assert results[0]["score_breakdown"]["rerank_weight"] == 0.75
    assert results[0]["score_breakdown"]["citation_weight"] == 0.25


def test_apply_citation_blended_scores_rejects_bad_weights():
    with pytest.raises(ValueError, match="non-negative"):
        apply_citation_blended_scores([], rerank_weight=-1)
    with pytest.raises(ValueError, match="positive"):
        apply_citation_blended_scores([], rerank_weight=0, citation_weight=0)


def test_agent_task_intent_boost_favors_tool_execution_evidence():
    query = "How does an agent perform tasks compared with a normal chatbot?"
    candidate = {
        "title": "A Survey on Large Language Model based Autonomous Agents",
        "topic": "AI Agents & Tool Use",
        "text": "Autonomous agents use planning, tool APIs, execution actions, environment feedback, and workflows.",
    }

    assert agent_task_intent_boost(query, candidate) > 0.08
    assert agent_task_intent_boost("What are hallucination benchmarks?", candidate) == 0.0


def test_query_intent_boost_reranks_agent_tool_sources_above_debate_only_examples():
    query = "How does an agent perform tasks while a plain chatbot mainly answers?"
    candidates = [
        {
            "paper_id": "debate",
            "title": "Encouraging Divergent Thinking through Multi-Agent Debate",
            "topic": "AI Agents & Tool Use",
            "text": "Agents express arguments in a debate setting.",
            "citation_count": 204,
            "rerank_score": 0.9,
            "blended_score": 0.72,
            "score_breakdown": {"rerank_score": 0.9},
        },
        {
            "chunk_id": "tool-survey",
            "title": "A Survey on Large Language Model based Autonomous Agents",
            "topic": "AI Agents & Tool Use",
            "text": "Agents use planning, external tools, REST APIs, action execution, environment observation, feedback, and workflows to complete tasks.",
            "citation_count": 1205,
            "rerank_score": 0.82,
            "blended_score": 0.67,
            "score_breakdown": {"rerank_score": 0.82},
        },
    ]

    boosted = apply_query_intent_boosts(query, candidates)

    assert boosted[0]["chunk_id"] == "tool-survey"
    assert boosted[0]["score_breakdown"]["intent_boost"] > 0
    assert "intent_boost" not in boosted[1]["score_breakdown"]


def test_rerank_and_blend_runs_end_to_end_with_mocked_model():
    candidates = make_candidates()
    model = FakeCrossEncoder([-5.0, -4.0, 10.0])

    results = rerank_and_blend("Which benchmarks evaluate hallucination?", candidates, model=model, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-1"
    assert "blended_score" in results[0]
    assert "citation_score" in results[0]


def test_rerank_and_blend_rejects_bad_top_k():
    with pytest.raises(ValueError, match="top_k"):
        rerank_and_blend("query", make_candidates(), model=FakeCrossEncoder([]), top_k=0)

def test_fallback_retrieval_scores_use_existing_scores_before_position():
    candidates = [
        {"paper_id": "a", "dense_score": 0.4},
        {"paper_id": "b", "hybrid_score": 0.9},
        {"paper_id": "c"},
    ]

    assert fallback_retrieval_scores(candidates) == [0.4, 0.9, 1.0]


def test_rerank_and_blend_falls_back_without_reordering_when_default_cross_encoder_unavailable(monkeypatch):
    def fail_to_load(*args, **kwargs):
        raise ImportError("torch binary mismatch")

    monkeypatch.setattr("retrieval.rerank.load_cross_encoder", fail_to_load)
    candidates = [
        {"paper_id": "first", "title": "First", "dense_score": 0.2, "citation_count": 1},
        {"paper_id": "second", "title": "Second", "dense_score": 0.9, "citation_count": 1},
    ]

    results = rerank_and_blend("attention mechanisms", candidates, top_k=2)

    assert [result["paper_id"] for result in results] == ["first", "second"]
    assert results[0]["rerank_fallback"] == "cross_encoder_unavailable"
    assert "blended_score" not in results[0]


def test_rerank_and_blend_does_not_hide_explicit_model_errors():
    class BrokenModel:
        def predict(self, pairs):
            raise RuntimeError("explicit model failed")

    with pytest.raises(RuntimeError, match="explicit model failed"):
        rerank_and_blend("attention mechanisms", make_candidates(), model=BrokenModel())

