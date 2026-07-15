import json
import subprocess
import sys

import pytest

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
from shared.schemas import EvaluationQuery, QueryRoute, RetrievedChunk, RetrievedPaper, UnifiedSearchResponse


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


def test_evaluation_query_defaults_expected_relevant_ids_to_empty():
    query = EvaluationQuery(query="What are RAG themes?", expected_route="paper_level")

    assert query.expected_relevant_ids == []


def test_load_eval_queries_reads_fixture():
    queries = load_eval_queries(__import__("pathlib").Path("tests/fixtures/eval_queries.json"))

    assert len(queries) == 20
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


def test_summary_to_text_labels_rigorous_and_sanity_metrics():
    summary = {
        "queries": 20,
        "queries_with_relevant_ids": 12,
        "queries_topic_keyword_only": 8,
        "route_accuracy": 0.9,
        "topic_hit_rate": {5: {"value": 0.85, "n": 20}},
        "keyword_hit_rate": {5: {"value": 0.75, "n": 20}},
        "recall": {5: {"value": 0.72, "n": 12}},
        "mrr": {"value": 0.68, "n": 12},
    }

    text = summary_to_text(summary, (5,))

    assert "queries_with_relevant_ids: 12" in text
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
