import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from retrieval.router import route_query, score_signals
from shared.schemas import QueryRoute


def test_query_route_schema_strips_and_validates_query():
    route = QueryRoute(
        query="  What are the main approaches?  ",
        route="paper_level",
        reason="Broad overview",
        confidence=0.8,
    )

    assert route.query == "What are the main approaches?"


def test_query_route_schema_rejects_blank_query():
    with pytest.raises(ValidationError):
        QueryRoute(query="   ", route="paper_level", reason="bad", confidence=0.5)


def test_router_selects_paper_level_for_broad_questions():
    decision = route_query("What are the main approaches for reducing hallucinations in LLMs?")

    assert decision.route == "paper_level"
    assert decision.confidence >= 0.8
    assert any("main approaches" in signal for signal in decision.matched_signals)


def test_router_selects_chunk_level_for_dataset_and_metric_questions():
    decision = route_query("Which datasets and metrics are used to evaluate hallucination detection?")

    assert decision.route == "chunk_level"
    assert decision.confidence >= 0.8
    assert any("datasets" in signal for signal in decision.matched_signals)
    assert any("metrics" in signal for signal in decision.matched_signals)


def test_router_selects_hybrid_both_for_comparison_questions():
    decision = route_query("Compare RAG and self-verification methods for reducing hallucinations.")

    assert decision.route == "hybrid_both"
    assert decision.confidence >= 0.78
    assert any("compare" in signal for signal in decision.matched_signals)


def test_router_selects_metadata_filter_for_ranking_and_year_questions():
    decision = route_query("Show top-cited LoRA papers after 2020.")

    assert decision.route == "metadata_filter"
    assert decision.confidence >= 0.8
    assert any("top-cited" in signal or "after" in signal for signal in decision.matched_signals)


def test_router_defaults_ambiguous_queries_to_hybrid_both():
    decision = route_query("Tell me about hallucination detection.")

    assert decision.route == "hybrid_both"
    assert decision.confidence == 0.55
    assert "fallback: no confident route-specific signal matched" in decision.matched_signals


def test_router_defaults_tied_queries_to_hybrid_both():
    decision = route_query("Give an overview of datasets for hallucination detection.")

    assert decision.route == "hybrid_both"
    assert decision.confidence == 0.55
    assert "fallback: ambiguous or low-confidence route match" in decision.matched_signals


def test_score_signals_returns_explainable_signal_text():
    scores, signals = score_signals("List recent papers by citation count.")

    assert scores["metadata_filter"] >= 2
    assert any("recent" in signal for signal in signals["metadata_filter"])


def test_router_cli_outputs_json():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "retrieval.router",
            "Which benchmarks and metrics evaluate hallucination detection?",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["route"] == "chunk_level"
    assert payload["query"] == "Which benchmarks and metrics evaluate hallucination detection?"
