import math

import pytest

from retrieval.rerank import (
    apply_citation_blended_scores,
    attach_rerank_scores,
    candidate_to_text,
    normalize_scores,
    normalized_citation_scores,
    fallback_retrieval_scores,
    rerank_and_blend,
    score_with_cross_encoder,
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


def test_normalize_scores_handles_range_and_equal_values():
    assert normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert normalize_scores([3.0, 3.0]) == [1.0, 1.0]
    assert normalize_scores([]) == []


def test_normalized_citation_scores_uses_log_scale():
    candidates = [{"citation_count": 0}, {"citation_count": 9}, {"citation_count": 99}]
    scores = normalized_citation_scores(candidates)

    assert scores[0] == 0.0
    assert 0.0 < scores[1] < scores[2]
    assert math.isclose(scores[2], 1.0)


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
    reranked = attach_rerank_scores(candidates, [0.1, 0.9])

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


def test_rerank_and_blend_runs_end_to_end_with_mocked_model():
    candidates = make_candidates()
    model = FakeCrossEncoder([0.2, 0.3, 0.95])

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


def test_rerank_and_blend_falls_back_when_default_cross_encoder_unavailable(monkeypatch):
    def fail_to_load(*args, **kwargs):
        raise ImportError("torch binary mismatch")

    monkeypatch.setattr("retrieval.rerank.load_cross_encoder", fail_to_load)
    candidates = [
        {"paper_id": "lower", "title": "Lower", "dense_score": 0.2, "citation_count": 1},
        {"paper_id": "higher", "title": "Higher", "dense_score": 0.9, "citation_count": 1},
    ]

    results = rerank_and_blend("attention mechanisms", candidates, top_k=2)

    assert results[0]["paper_id"] == "higher"
    assert results[0]["rerank_fallback"] == "cross_encoder_unavailable"
    assert "blended_score" in results[0]


def test_rerank_and_blend_does_not_hide_explicit_model_errors():
    class BrokenModel:
        def predict(self, pairs):
            raise RuntimeError("explicit model failed")

    with pytest.raises(RuntimeError, match="explicit model failed"):
        rerank_and_blend("attention mechanisms", make_candidates(), model=BrokenModel())

