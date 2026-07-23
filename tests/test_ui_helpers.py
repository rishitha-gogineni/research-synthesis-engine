import os

import pytest

from ui import api_client
from ui import streamlit_app


def sample_guidance_payload():
    return {
        "retrieval": {
            "route": {"route": "hybrid_both", "reason": "test", "confidence": 0.9, "matched_signals": ["compare"]},
            "paper_result_count": 1,
            "chunk_result_count": 1,
            "paper_results": [
                {
                    "paper_id": "p1",
                    "title": "Paper One",
                    "topic": "Retrieval-Augmented Generation (RAG)",
                    "year": 2024,
                    "citation_count": 12,
                    "blended_score": 0.91,
                }
            ],
            "chunk_results": [
                {
                    "chunk_id": "c1",
                    "paper_id": "p1",
                    "title": "Paper One",
                    "topic": "Retrieval-Augmented Generation (RAG)",
                    "section_hint": "Evaluation",
                    "dense_score": 0.82,
                }
            ],
        },
        "confidence": {"decision": "sufficient_evidence"},
        "evidence_matrix": {
            "rows": [
                {
                    "claim": "Retrieval grounds answers.",
                    "source_ids": ["paper:p1"],
                    "methodology": "Hybrid retrieval.",
                    "dataset": "HaluEval",
                    "key_result": "Fewer unsupported claims.",
                    "limitation": "Limited evaluation.",
                    "evidence_strength": "high",
                }
            ]
        },
        "reading_path": {
            "stages": [
                {
                    "stage": "foundational",
                    "papers": [
                        {
                            "order": 1,
                            "title": "Paper One",
                            "publication_year": 2024,
                            "citation_count": 12,
                            "reason_to_read": "Start here.",
                            "source_ids": ["paper:p1"],
                        }
                    ],
                }
            ]
        },
        "open_problems": {
            "problems": [
                {
                    "title": "Evaluation coverage",
                    "category": "evaluation",
                    "evidence_strength": "weak",
                    "why_it_matters": "Benchmarks shape conclusions.",
                    "supporting_source_ids": ["paper:p1"],
                }
            ]
        },
        "metrics": {"total_ms": 12.3, "retrieval_ms": 3.2},
    }


def test_build_guidance_payload_uses_question_and_optional_filters():
    payload = api_client.build_guidance_payload(
        question="  Compare RAG and verification.  ",
        top_k=5,
        research_areas=["Retrieval-Augmented Generation (RAG)"],
        publication_year_min=2020,
        publication_year_max=2026,
        full_text_only=True,
        include_debug=True,
    )

    assert payload["question"] == "Compare RAG and verification."
    assert "query" not in payload
    assert payload["research_areas"] == ["Retrieval-Augmented Generation (RAG)"]
    assert payload["publication_year_min"] == 2020
    assert payload["publication_year_max"] == 2026
    assert payload["full_text_only"] is True
    assert payload["include_debug"] is True


def test_error_message_formats_structured_api_errors():
    message = api_client.error_message(
        {"error": {"code": "RETRIEVAL_FAILED", "message": "Unable to retrieve.", "request_id": "abc-123"}}
    )

    assert message == "RETRIEVAL_FAILED: Unable to retrieve. Request ID: abc-123"


def test_table_helpers_flatten_guidance_response():
    payload = sample_guidance_payload()

    assert api_client.evidence_rows(payload)[0]["Claim"] == "Retrieval grounds answers."
    assert api_client.reading_path_rows(payload)[0]["Stage"] == "foundational"
    assert api_client.open_problem_rows(payload)[0]["Problem"] == "Evaluation coverage"
    papers, chunks = api_client.source_rows(payload)
    assert papers[0]["Paper ID"] == "p1"
    assert chunks[0]["Chunk"] == "c1"


def test_summary_and_metric_rows_use_api_response_shape():
    payload = sample_guidance_payload()

    summary = api_client.summary_items(payload, "request-1")
    metrics = api_client.metric_rows(payload)

    assert summary == {
        "Route": "hybrid_both",
        "Confidence": "sufficient_evidence",
        "Papers": 1,
        "Chunks": 1,
        "Request ID": "request-1",
    }
    assert {row["Metric"] for row in metrics} == {"total_ms", "retrieval_ms"}


def test_api_base_url_uses_env_override(monkeypatch):
    monkeypatch.setenv("RSE_API_URL", "http://localhost:9999/")

    assert api_client.api_base_url() == "http://localhost:9999"


def test_new_request_id_has_ui_prefix():
    assert api_client.new_request_id().startswith("ui-")


def test_post_api_returns_structured_error_for_http_error(monkeypatch):
    class FakeResponse:
        status_code = 503
        headers = {"X-Request-ID": "request-1"}
        text = ""

        def json(self):
            return {"error": {"code": "QDRANT_UNAVAILABLE", "message": "Vector store unavailable.", "request_id": "request-1"}}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    body, request_id = api_client.post_api("/guidance", {"question": "test"}, request_id="request-1")

    assert request_id == "request-1"
    assert body["_error_status"] == 503
    assert api_client.error_message(body).startswith("QDRANT_UNAVAILABLE")

def test_theme_rows_flatten_brief_themes():
    payload = sample_guidance_payload()
    payload["brief"] = {
        "themes": [
            {
                "theme": "Evidence grounding",
                "summary": "Retrieved evidence grounds the answer.",
                "supporting_source_ids": ["paper:p1", "chunk:c1"],
            }
        ]
    }

    rows = api_client.theme_rows(payload)

    assert rows == [
        {
            "Theme": "Evidence grounding",
            "Description": "Retrieved evidence grounds the answer.",
            "Sources": "paper:p1, chunk:c1",
        }
    ]


def test_top_supporting_evidence_merges_papers_and_chunks_by_score():
    payload = sample_guidance_payload()
    payload["retrieval"]["paper_results"][0]["main_contribution"] = "Explains retrieval grounding."
    payload["retrieval"]["chunk_results"][0]["text"] = "Detailed evaluation snippet."
    payload["retrieval"]["chunk_results"][0]["blended_score"] = 0.99

    rows = api_client.top_supporting_evidence(payload, limit=2)

    assert [row["Source"] for row in rows] == ["full text", "paper"]
    assert rows[0]["Source ID"] == "chunk:c1"
    assert rows[1]["Why It Matters"] == "Explains retrieval grounding."


def test_ordered_sections_adapts_to_question_intent():
    assert api_client.ordered_sections("Which LoRA papers should I read first?")[0] == "Reading Path"
    assert api_client.ordered_sections("What are the open problems in hallucination detection?")[1] == "Open Problems"
    assert api_client.ordered_sections("Which datasets evaluate hallucination?")[1] == "Evidence"
    assert api_client.ordered_sections("What are attention mechanisms in transformers?")[:2] == ["Brief", "Top Evidence"]

def test_confidence_style_and_route_label_are_display_ready():
    assert api_client.confidence_style("sufficient_evidence") == ("success", "High confidence")
    assert api_client.confidence_style("insufficient_evidence") == ("danger", "Insufficient evidence")
    assert api_client.confidence_style("unexpected_decision") == ("route", "unexpected decision")
    assert api_client.route_label("hybrid_both") == "hybrid both"


def test_section_counts_summarize_result_sections():
    payload = sample_guidance_payload()

    counts = api_client.section_counts(payload)

    assert counts["Evidence"] == "1 claims"
    assert counts["Top Evidence"] == "2 sources"
    assert counts["Reading Path"] == "1 stages"
    assert counts["Open Problems"] == "1 found"
    assert counts["Sources"] == "1 papers / 1 chunks"

def test_trust_summary_and_answerable_gate_use_confidence_decision():
    payload = sample_guidance_payload()

    summary = api_client.trust_summary(payload)

    assert api_client.is_answerable(payload) is True
    assert summary["decision"] == "sufficient_evidence"
    assert summary["kind"] == "success"
    assert summary["label"] == "High confidence"
    assert summary["route"] == "hybrid_both"
    assert summary["paper_count"] == 1
    assert summary["chunk_count"] == 1


def test_weak_evidence_guidance_does_not_treat_empty_labels_as_failure():
    payload = sample_guidance_payload()
    payload["confidence"] = {"decision": "insufficient_evidence"}

    guidance = api_client.weak_evidence_guidance(payload)

    assert api_client.is_answerable(payload) is False
    assert any("did not mark them as sufficient" in item for item in guidance)
    assert any("Sources tab" in item for item in guidance)


def test_weak_evidence_guidance_handles_no_retrieved_sources():
    payload = sample_guidance_payload()
    payload["confidence"] = {"decision": "insufficient_evidence"}
    payload["retrieval"]["paper_result_count"] = 0
    payload["retrieval"]["chunk_result_count"] = 0

    guidance = api_client.weak_evidence_guidance(payload)

    assert guidance[0] == "No matching papers or full-text chunks were retrieved for this question."



def test_build_guidance_payload_includes_chat_history_when_present():
    history = [{"role": "user", "content": "Explain LoRA."}]

    payload = api_client.build_guidance_payload(
        question="What are its limitations?",
        top_k=5,
        chat_history=history,
    )

    assert payload["chat_history"] == history


def test_agent_trace_rows_flattens_agent_response():
    rows = api_client.agent_trace_rows(
        {
            "trace": [
                {"step": "Context rewrite", "status": "completed", "detail": "Standalone query: LoRA limitations"},
                {"step": "Retrieval attempt 1"},
            ]
        }
    )

    assert rows == [
        {"Step": "Context rewrite", "Status": "completed", "Detail": "Standalone query: LoRA limitations"},
        {"Step": "Retrieval attempt 1", "Status": "completed", "Detail": "-"},
    ]


def test_run_agent_research_posts_to_agent_endpoint(monkeypatch):
    calls = []

    def fake_post(endpoint, payload, *, request_id=None, timeout=120):
        calls.append((endpoint, payload, request_id, timeout))
        return {"trace": []}, request_id

    monkeypatch.setattr(api_client, "post_api", fake_post)

    body, request_id = api_client.run_agent_research({"question": "test"}, request_id="agent-ui-1")

    assert body == {"trace": []}
    assert request_id == "agent-ui-1"
    assert calls == [("/agent/research", {"question": "test"}, "agent-ui-1", 180)]


def test_rewrite_summary_formats_guidance_rewrite_fields():
    summary = api_client.rewrite_summary(
        {
            "question": "What are its limitations?",
            "standalone_query": "What are the limitations of LoRA?",
            "rewrite_used": True,
            "rewrite_method": "llm",
            "rewrite_reason": "Resolved pronoun.",
        }
    )

    assert summary["Original Question"] == "What are its limitations?"
    assert summary["Standalone Query"] == "What are the limitations of LoRA?"
    assert summary["Rewrite Used"] == "yes"
    assert summary["Method"] == "llm"


def test_chunk_display_label_uses_title_section_and_score():
    label = streamlit_app.chunk_display_label(
        {
            "title": "A Survey on Large Language Model based Autonomous Agents",
            "section_hint": "results",
            "blended_score": 0.876,
        },
        2,
    )

    assert label.startswith("2. A Survey on Large Language Model based Autonomous Agents")
    assert "results" in label
    assert "score 0.88" in label
    assert label != "results"
