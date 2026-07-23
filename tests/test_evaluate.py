import json
import subprocess
import sys

import pytest

from agent.query_rewriter import QueryRewriteResult
from retrieval.evaluate import (
    EvaluationError,
    evaluate_response,
    format_rate,
    keyword_hit,
    load_eval_queries,
    parse_top_ks,
    reciprocal_rank,
    run_evaluation,
    summary_to_text,
    topic_hit,
)
from shared.schemas import ConfidenceAssessment, EvaluationQuery, QueryRoute, RetrievedChunk, RetrievedPaper, UnifiedSearchResponse


def make_response(query, route, paper_ids=None, chunk_ids=None, topic="LLM Evaluation & Hallucination Detection"):
    paper_ids = paper_ids or []
    chunk_ids = chunk_ids or []
    papers = [
        RetrievedPaper(
            paper_id=paper_id,
            title=f"Paper {paper_id}",
            topic=topic,
            abstract="hallucination benchmark retrieval grounding",
            citation_count=10,
        )
        for paper_id in paper_ids
    ]
    chunks = [
        RetrievedChunk(
            chunk_id=chunk_id,
            paper_id=f"paper-for-{chunk_id}",
            title=f"Chunk {chunk_id}",
            topic=topic,
            text="TruthfulQA HaluEval benchmark metric hallucination",
            citation_count=5,
        )
        for chunk_id in chunk_ids
    ]
    return UnifiedSearchResponse(
        query=query,
        route=QueryRoute(query=query, route=route, reason="test", confidence=0.9),
        paper_result_count=len(papers),
        chunk_result_count=len(chunks),
        paper_results=papers,
        chunk_results=chunks,
    )


def make_confidence(response, decision="sufficient_evidence"):
    return ConfidenceAssessment(
        query=response.query,
        route=response.route.route,
        confidence_score=0.9 if decision == "sufficient_evidence" else 0.3,
        decision=decision,
        reason="test confidence",
        recommended_action="test action",
        signals=["test"],
        result_count=response.paper_result_count + response.chunk_result_count,
        top_score=0.9,
        route_confidence=0.9,
    )


def test_evaluation_query_defaults_expected_relevant_ids_to_empty():
    query = EvaluationQuery(query="What are RAG themes?", expected_route="paper_level")

    assert query.expected_relevant_ids == []


def test_load_eval_queries_reads_fixture():
    queries = load_eval_queries(__import__("pathlib").Path("tests/fixtures/eval_queries.json"))

    assert len(queries) >= 20
    assert any(query.expected_relevant_ids for query in queries)


def test_load_eval_queries_reports_bad_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")

    with pytest.raises(EvaluationError, match="failed to load"):
        load_eval_queries(bad)


def test_topic_and_keyword_hits_use_top_k_results():
    results = make_response("q", "paper_level", paper_ids=["p1"]).paper_results

    assert topic_hit(results, ["LLM Evaluation & Hallucination Detection"], 1) is True
    assert topic_hit(results, ["Retrieval-Augmented Generation (RAG)"], 1) is False
    assert keyword_hit(results, ["benchmark"], 1) is True
    assert keyword_hit(results, ["nonexistent"], 1) is False


def test_reciprocal_rank_finds_first_relevant_id():
    results = make_response("q", "paper_level", paper_ids=["p1", "p2", "p3"]).paper_results

    assert reciprocal_rank(results, ["p3"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(results, ["missing"]) == 0.0


def test_evaluate_response_uses_expected_route_result_set_for_ids():
    query = EvaluationQuery(
        query="Which datasets evaluate hallucinations?",
        expected_route="chunk_level",
        expected_topics=["LLM Evaluation & Hallucination Detection"],
        expected_keywords=["TruthfulQA"],
        expected_relevant_ids=["c2"],
    )
    response = make_response("q", "chunk_level", paper_ids=["c2"], chunk_ids=["c1", "c2"])

    evaluation = evaluate_response(query, response, (1, 2))

    assert evaluation["route_correct"] is True
    assert evaluation["id_hit_sets"][1] == set()
    assert evaluation["id_hit_sets"][2] == {"c2"}
    assert evaluation["reciprocal_rank"] == pytest.approx(0.5)


def test_run_evaluation_computes_recall_only_on_labeled_subset():
    queries = [
        EvaluationQuery(
            query="labeled hit",
            expected_route="paper_level",
            expected_topics=["LLM Evaluation & Hallucination Detection"],
            expected_keywords=["benchmark"],
            expected_relevant_ids=["p2"],
        ),
        EvaluationQuery(
            query="unlabeled sanity only",
            expected_route="paper_level",
            expected_topics=["LLM Evaluation & Hallucination Detection"],
            expected_keywords=["benchmark"],
            expected_relevant_ids=[],
        ),
        EvaluationQuery(
            query="labeled miss",
            expected_route="paper_level",
            expected_topics=["Wrong Topic"],
            expected_keywords=["missing"],
            expected_relevant_ids=["missing"],
        ),
    ]

    def fake_runner(query, **kwargs):
        if query == "labeled hit":
            return make_response(query, "paper_level", paper_ids=["p1", "p2"])
        if query == "unlabeled sanity only":
            return make_response(query, "paper_level", paper_ids=["p3"])
        return make_response(query, "chunk_level", paper_ids=["p4"])

    summary, evaluations = run_evaluation(queries, search_runner=fake_runner, top_ks=(1, 2), apply_reranking=False)

    assert summary["queries"] == 3
    assert summary["queries_with_relevant_ids"] == 2
    assert summary["queries_topic_keyword_only"] == 1
    assert summary["route_accuracy"] == pytest.approx(2 / 3)
    assert summary["recall"][1]["value"] == 0.0
    assert summary["recall"][1]["n"] == 2
    assert summary["recall"][2]["value"] == 0.5
    assert summary["recall"][2]["n"] == 2
    assert summary["mrr"]["value"] == pytest.approx(0.25)
    assert len(evaluations) == 3


def test_run_evaluation_rewrites_contextual_queries_and_reports_rewrite_metric():
    queries = [
        EvaluationQuery(
            query="What are its limitations?",
            category="multi_turn",
            expected_route="chunk_level",
            chat_history=[{"role": "user", "content": "Explain LoRA fine-tuning."}],
            expected_standalone_keywords=["LoRA", "limitations"],
        )
    ]
    seen_queries = []

    def fake_rewriter(query, chat_history):
        assert chat_history[0].content == "Explain LoRA fine-tuning."
        return QueryRewriteResult(
            original_query=query,
            standalone_query="What are the limitations of LoRA fine-tuning?",
            rewrite_used=True,
            method="heuristic",
            reason="test",
        )

    def fake_runner(query, **kwargs):
        seen_queries.append(query)
        return make_response(query, "chunk_level", chunk_ids=["c1"], topic="Fine-tuning (LoRA / PEFT)")

    summary, evaluations = run_evaluation(
        queries,
        search_runner=fake_runner,
        rewriter=fake_rewriter,
        top_ks=(1,),
        apply_reranking=False,
    )

    assert seen_queries == ["What are the limitations of LoRA fine-tuning?"]
    assert evaluations[0]["rewrite_used"] is True
    assert evaluations[0]["rewrite_keyword_hit"] is True
    assert summary["multi_turn_queries"] == 1
    assert summary["rewrite_keyword_hit_rate"] == {"value": 1.0, "n": 1}


def test_run_evaluation_reports_confidence_and_crag_fallback_metrics():
    queries = [
        EvaluationQuery(
            query="What does this corpus say about quantum cryptography hardware?",
            category="out_of_corpus",
            expected_route="hybrid_both",
            expected_confidence_decision="insufficient_evidence",
        ),
        EvaluationQuery(
            query="What are RAG themes?",
            expected_route="paper_level",
            expected_confidence_decision="sufficient_evidence",
        ),
    ]

    def fake_runner(query, **kwargs):
        route = "hybrid_both" if "quantum" in query else "paper_level"
        return make_response(query, route, paper_ids=["p1"])

    def fake_confidence(response):
        decision = "insufficient_evidence" if "quantum" in response.query else "sufficient_evidence"
        return make_confidence(response, decision)

    summary, evaluations = run_evaluation(
        queries,
        search_runner=fake_runner,
        confidence_checker=fake_confidence,
        top_ks=(1,),
    )

    assert summary["out_of_corpus_queries"] == 1
    assert summary["confidence_decision_accuracy"] == {"value": 1.0, "n": 2}
    assert summary["crag_fallback_success_rate"] == {"value": 1.0, "n": 1}
    assert evaluations[0]["actual_confidence_decision"] == "insufficient_evidence"


def test_summary_to_text_labels_rigorous_and_sanity_metrics():
    summary = {
        "queries": 20,
        "queries_with_relevant_ids": 12,
        "queries_topic_keyword_only": 8,
        "multi_turn_queries": 2,
        "out_of_corpus_queries": 1,
        "route_accuracy": 0.9,
        "rewrite_keyword_hit_rate": {"value": 0.5, "n": 2},
        "confidence_decision_accuracy": {"value": 0.75, "n": 4},
        "crag_fallback_success_rate": {"value": 0.67, "n": 3},
        "topic_hit_rate": {5: {"value": 0.85, "n": 20}},
        "keyword_hit_rate": {5: {"value": 0.75, "n": 20}},
        "recall": {5: {"value": 0.72, "n": 12}},
        "mrr": {"value": 0.68, "n": 12},
    }

    text = summary_to_text(summary, (5,))

    assert "queries_with_relevant_ids: 12" in text
    assert "multi_turn_queries: 2" in text
    assert "out_of_corpus_queries: 1" in text
    assert "rewrite_keyword_hit_rate: 0.50 (contextual subset, n=2)" in text
    assert "confidence_decision_accuracy: 0.75 (labeled confidence subset, n=4)" in text
    assert "crag_fallback_success_rate: 0.67 (expected fallback subset, n=3)" in text
    assert "topic_hit_rate@5: 0.85 (sanity check, n=20)" in text
    assert "recall@5 (labeled subset, n=12): 0.72" in text
    assert "mrr (labeled subset, n=12): 0.68" in text


def test_format_rate_and_parse_top_ks():
    assert format_rate(None) == "n/a"
    assert format_rate(0.1234) == "0.12"
    assert parse_top_ks("10,5,5") == (5, 10)


def test_parse_top_ks_rejects_bad_values():
    with pytest.raises(Exception):
        parse_top_ks("0")


def test_evaluate_cli_json_loads_fixture_without_live_search_for_bad_query_file(tmp_path):
    fixture = tmp_path / "eval.json"
    fixture.write_text(json.dumps([]))
    completed = subprocess.run(
        [sys.executable, "-m", "retrieval.evaluate", "--queries", str(fixture), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["summary"]["queries"] == 0
    assert payload["summary"]["queries_with_relevant_ids"] == 0
