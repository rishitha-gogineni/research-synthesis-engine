import json
import subprocess
import sys

import pytest

from agent.query_rewriter import QueryRewriteResult
from retrieval.evaluate import (
    EvaluationError,
    canonical_identifier,
    effective_routes,
    evaluate_response,
    format_rate,
    keyword_hit,
    load_eval_queries,
    merge_ranked_lists,
    parse_top_ks,
    reciprocal_rank,
    result_id,
    run_evaluation,
    select_results,
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
    assert query.acceptable_routes == []
    assert effective_routes(query) == ["paper_level"]


def test_load_eval_queries_reads_fixture():
    queries = load_eval_queries(__import__("pathlib").Path("tests/fixtures/eval_queries_100_chunk_grounded.json"))

    assert len(queries) >= 50
    assert sum(bool(query.expected_relevant_ids) for query in queries) >= 35
    assert {query.evaluation_focus for query in queries} >= {
        "full_text_evidence",
        "metadata_filter",
        "confidence_gate",
    }


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


def test_duplicate_paper_aliases_resolve_to_one_stable_identifier():
    aliases = {"paper-alias": "paper-canonical"}

    assert canonical_identifier("paper-alias", aliases) == "paper-canonical"
    assert canonical_identifier("paper-canonical", aliases) == "paper-canonical"


def test_evaluate_response_matches_duplicate_paper_alias_to_canonical_id():
    query = EvaluationQuery(
        query="What does the duplicate paper report?",
        expected_route="paper_level",
        expected_relevant_ids=["https://openalex.org/W4391136507"],
    )
    response = make_response(
        "q",
        "paper_level",
        paper_ids=["https://openalex.org/W4383605161"],
    )

    evaluation = evaluate_response(query, response, (1,))

    assert evaluation["id_hit_sets"][1] == ["https://openalex.org/W4383605161"]
    assert evaluation["reciprocal_rank"] == 1.0


def test_duplicate_aliases_count_as_one_relevance_judgment():
    query = EvaluationQuery(
        query="What does the duplicated record report?",
        expected_route="paper_level",
        expected_relevant_ids=[
            "https://openalex.org/W4391136507",
            "https://openalex.org/W4383605161",
        ],
    )
    response = make_response(
        "q",
        "paper_level",
        paper_ids=["https://openalex.org/W4391136507"],
    )

    evaluation = evaluate_response(query, response, (1,))

    assert evaluation["id_hit_fractions"][1] == 1.0


def test_evaluate_response_accepts_valid_route_alternative():
    query = EvaluationQuery(
        query="Which benchmarks are used for tool-use agents?",
        expected_route="chunk_level",
        acceptable_routes=["hybrid_both"],
        expected_topics=["AI Agents & Tool Use"],
        expected_keywords=["benchmark"],
        expected_relevant_ids=["c1"],
    )
    response = make_response("q", "hybrid_both", paper_ids=["p1"], chunk_ids=["c1"], topic="AI Agents & Tool Use")

    evaluation = evaluate_response(query, response, (1,))

    assert evaluation["route_correct"] is True
    assert evaluation["acceptable_routes"] == ["chunk_level", "hybrid_both"]
    assert evaluation["id_hit_sets"][1] == ["c1"]


def test_evaluate_response_true_recall_differs_from_any_hit():
    # Two relevant chunks expected, but only one is actually retrieved in the
    # route's result set. A hit-rate style metric ("did we get >=1 relevant
    # id?") should say yes; true recall ("what fraction of relevant ids did
    # we retrieve?") should say 0.5, not 1.0. These must not collapse to the
    # same number, or the "recall" label is misleading.
    query = EvaluationQuery(
        query="Which papers benchmark hallucination detection?",
        expected_route="chunk_level",
        expected_relevant_ids=["c1", "c2"],
    )
    response = make_response("q", "chunk_level", chunk_ids=["c1", "c-unrelated"])

    evaluation = evaluate_response(query, response, (5,))

    assert evaluation["id_hit_sets"][5] == ["c1"]
    assert evaluation["id_hit_fractions"][5] == pytest.approx(0.5)


def test_evaluate_response_is_json_native_and_preserves_rationale():
    query = EvaluationQuery(
        query="What accuracy does the paper report?",
        expected_route="chunk_level",
        expected_relevant_ids=["c1"],
        rationale="[factual] regression fixture",
    )
    response = make_response("q", "chunk_level", chunk_ids=["c1"])

    evaluation = evaluate_response(query, response, (5,))

    assert evaluation["rationale"] == "[factual] regression fixture"
    assert json.loads(json.dumps(evaluation))["id_hit_sets"]["5"] == ["c1"]
    assert evaluation["expected_relevant_ids"] == ["c1"]
    assert evaluation["route_matched_signals"] == []


def test_summarize_evaluations_reports_hit_rate_and_recall_as_distinct_numbers():
    queries = [
        EvaluationQuery(query="q1", expected_route="chunk_level", expected_relevant_ids=["c1", "c2"]),
    ]
    responses = {"q1": make_response("q1", "chunk_level", chunk_ids=["c1", "c-unrelated"])}

    def fake_runner(query_text, **_kwargs):
        return responses[query_text]

    summary, _evaluations = run_evaluation(queries, search_runner=fake_runner, top_ks=(5,), apply_query_rewriting=False)

    # Any-hit metric: at least one of the two relevant ids was retrieved -> 1.0.
    assert summary["id_relevant_hit_rate"][5]["value"] == pytest.approx(1.0)
    # True recall: only 1 of 2 relevant ids was retrieved -> 0.5. These must differ.
    assert summary["recall"][5]["value"] == pytest.approx(0.5)
    assert summary["id_relevant_hit_rate"][5]["value"] != summary["recall"][5]["value"]


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
    assert evaluation["id_hit_sets"][1] == []
    assert evaluation["id_hit_sets"][2] == ["c2"]
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
    assert summary["evaluation_focus_counts"] == {"route_selection": 3}
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
        "evaluation_focus_counts": {"full_text_evidence": 8, "confidence_gate": 3},
        "route_accuracy": 0.9,
        "rewrite_keyword_hit_rate": {"value": 0.5, "n": 2},
        "confidence_decision_accuracy": {"value": 0.75, "n": 4},
        "crag_fallback_success_rate": {"value": 0.67, "n": 3},
        "topic_hit_rate": {5: {"value": 0.85, "n": 20}},
        "keyword_hit_rate": {5: {"value": 0.75, "n": 20}},
        "id_relevant_hit_rate": {5: {"value": 0.9, "n": 12}},
        "recall": {5: {"value": 0.72, "n": 12}},
        "mrr": {"value": 0.68, "n": 12},
    }

    text = summary_to_text(summary, (5,))

    assert "queries_with_relevant_ids: 12" in text
    assert "multi_turn_queries: 2" in text
    assert "out_of_corpus_queries: 1" in text
    assert "evaluation_focus_counts: confidence_gate=3, full_text_evidence=8" in text
    assert "rewrite_keyword_hit_rate: 0.50 (contextual subset, n=2)" in text
    assert "confidence_decision_accuracy: 0.75 (labeled confidence subset, n=4)" in text
    assert "crag_fallback_success_rate: 0.67 (expected fallback subset, n=3)" in text
    assert "topic_hit_rate@5: 0.85 (sanity check, n=20)" in text
    assert "hit_rate@5 (>=1 relevant id in top-5, labeled subset, n=12): 0.90" in text
    assert "recall@5 (fraction of all relevant ids retrieved, labeled subset, n=12): 0.72" in text
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


def test_merge_ranked_lists_interleaves_by_reciprocal_rank():
    response = make_response("q", "hybrid_both", paper_ids=["p1", "p2"], chunk_ids=["c1", "c2"])
    merged = merge_ranked_lists(list(response.paper_results), list(response.chunk_results))

    assert [result_id(result) for result in merged] == ["p1", "c1", "p2", "c2"]


def test_merge_ranked_lists_handles_one_empty_list():
    response = make_response("q", "hybrid_both", paper_ids=["p1", "p2"])
    merged = merge_ranked_lists(list(response.paper_results), [])

    assert [result_id(result) for result in merged] == ["p1", "p2"]


def test_select_results_concatenation_hides_chunks_at_the_cutoff():
    """Documents the structural defect that --merge-hybrid exists to address."""

    response = make_response(
        "q",
        "hybrid_both",
        paper_ids=[f"p{index}" for index in range(1, 11)],
        chunk_ids=[f"c{index}" for index in range(1, 11)],
    )
    concatenated = select_results(response, "hybrid_both")

    assert all(result_id(result).startswith("p") for result in concatenated[:10])


def test_select_results_merge_hybrid_makes_chunks_reachable_at_the_cutoff():
    response = make_response(
        "q",
        "hybrid_both",
        paper_ids=[f"p{index}" for index in range(1, 11)],
        chunk_ids=[f"c{index}" for index in range(1, 11)],
    )
    merged = select_results(response, "hybrid_both", merge_hybrid=True)
    visible = [result_id(result) for result in merged[:10]]

    assert any(identifier.startswith("c") for identifier in visible)
    assert len(merged) == 20


def test_select_results_merge_hybrid_does_not_affect_other_routes():
    response = make_response("q", "paper_level", paper_ids=["p1"], chunk_ids=["c1"])

    assert select_results(response, "paper_level", merge_hybrid=True) == select_results(
        response, "paper_level"
    )


def test_select_results_conditional_merge_applies_to_diversity_queries():
    response = make_response(
        "Compare RAG and fine-tuning approaches",
        "hybrid_both",
        paper_ids=[f"p{i}" for i in range(1, 11)],
        chunk_ids=[f"c{i}" for i in range(1, 11)],
    )
    merged = select_results(response, "hybrid_both", conditional_merge=True)
    visible = [result_id(r) for r in merged[:10]]

    assert any(i.startswith("c") for i in visible)


def test_select_results_conditional_merge_leaves_non_diversity_hybrid_alone():
    response = make_response(
        "How does RAG reduce hallucination?",
        "hybrid_both",
        paper_ids=[f"p{i}" for i in range(1, 11)],
        chunk_ids=[f"c{i}" for i in range(1, 11)],
    )
    conditional = select_results(response, "hybrid_both", conditional_merge=True)
    default = select_results(response, "hybrid_both")

    assert conditional == default
