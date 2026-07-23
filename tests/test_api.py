import time

from fastapi.testclient import TestClient

import api.main as api_main
from shared.schemas import (
    BriefTheme,
    ConfidenceAssessment,
    EvidenceMatrix,
    EvidenceMatrixRow,
    EvidenceSource,
    OpenProblem,
    OpenProblemsReport,
    QueryRoute,
    ReadingPath,
    ReadingPathItem,
    ReadingPathStage,
    ResearchBrief,
    RetrievedPaper,
    UnifiedSearchResponse,
)


client = TestClient(api_main.app)


def make_retrieval(query="What reduces hallucinations?"):
    paper = RetrievedPaper(
        paper_id="p1",
        title="Grounded RAG Paper",
        authors=["A. Researcher"],
        topic="Retrieval-Augmented Generation (RAG)",
        year=2024,
        citation_count=42,
        abstract="RAG grounds generation in retrieved evidence.",
        main_contribution="Shows how retrieval can ground answers.",
        methodology="Retrieval method study.",
        dataset_used="HaluEval",
        key_result="Grounding reduces unsupported claims.",
        limitations="Evaluation coverage is limited.",
        blended_score=0.95,
        rerank_score=0.95,
        hybrid_score=0.95,
        score_breakdown={"rerank_score": 0.9, "citation_score": 0.2},
    )
    return UnifiedSearchResponse(
        query=query,
        route=QueryRoute(query=query, route="paper_level", reason="test route", confidence=0.95, matched_signals=["main: main ideas question"]),
        paper_result_count=1,
        chunk_result_count=0,
        paper_results=[paper],
        chunk_results=[],
    )


def make_confidence(query="What reduces hallucinations?", decision="sufficient_evidence"):
    return ConfidenceAssessment(
        query=query,
        route="paper_level",
        confidence_score=0.91,
        decision=decision,
        reason="retrieval is strong",
        recommended_action="proceed",
        signals=["top_score=0.95"],
        result_count=1,
        top_score=0.95,
        route_confidence=0.95,
    )


def make_brief(query="What reduces hallucinations?", status="generated"):
    return ResearchBrief(
        query=query,
        status=status,
        confidence_decision="sufficient_evidence" if status == "generated" else "broaden_search",
        direct_answer="Retrieved evidence supports grounded generation.",
        themes=[BriefTheme(theme="Grounding", summary="Use retrieval evidence.", supporting_source_ids=["paper:p1"])],
        evidence_bullets=["RAG grounds answers [paper:p1]."],
        limitations=["Limited corpus."],
        open_problems=["Better evaluation coverage."],
        sources=[EvidenceSource(source_id="paper:p1", title="Grounded RAG Paper", topic="RAG", paper_id="p1", citation_count=42, evidence_text="RAG grounds generation.", score=0.95)],
    )


def make_matrix(query="What reduces hallucinations?"):
    return EvidenceMatrix(
        query=query,
        rows=[
            EvidenceMatrixRow(
                claim="Retrieval can ground answers.",
                supporting_papers=["Grounded RAG Paper"],
                source_ids=["paper:p1"],
                methodology="Retrieval method study.",
                dataset="HaluEval",
                key_result="Grounding reduces unsupported claims.",
                limitation="Evaluation coverage is limited.",
                evidence_strength="high",
                evidence_snippet="RAG grounds generation.",
            )
        ],
        markdown="| Claim | Sources |\n| --- | --- |",
    )


def make_reading_path(query="What reduces hallucinations?"):
    return ReadingPath(
        question=query,
        stages=[
            ReadingPathStage(
                stage="foundational",
                description="Start with grounding.",
                papers=[
                    ReadingPathItem(
                        order=1,
                        stage="foundational",
                        paper_id="p1",
                        title="Grounded RAG Paper",
                        authors=["A. Researcher"],
                        publication_year=2024,
                        citation_count=42,
                        topic="RAG",
                        reason_to_read="It explains retrieval grounding.",
                        focus_points=["grounding"],
                        prerequisites=[],
                        connection_to_next=None,
                        source_ids=["paper:p1"],
                        evidence_snippet="RAG grounds generation.",
                        relevance_score=0.95,
                    )
                ],
            )
        ],
        total_papers=1,
        confidence_decision="sufficient_evidence",
        limitations=[],
    )


def make_open_problems(query="What reduces hallucinations?"):
    return OpenProblemsReport(
        question=query,
        problems=[
            OpenProblem(
                title="Benchmark coverage remains limited",
                description="Retrieved evidence mentions limited evaluation coverage.",
                category="evaluation",
                why_it_matters="Weak benchmarks make comparisons less reliable.",
                evidence_summary="Evaluation coverage is limited.",
                supporting_paper_ids=["p1"],
                supporting_source_ids=["paper:p1"],
                evidence_snippets=["Evaluation coverage is limited."],
                evidence_strength="weak",
                suggested_research_directions=["Evaluate across more settings."],
                confidence=0.7,
            )
        ],
        recurring_limitations=[],
        conflicting_findings=[],
        evidence_gaps=[],
        corpus_limitations=[],
        confidence_decision="sufficient_evidence",
    )


def patch_core_services(monkeypatch, retrieval_calls=None, confidence_decision="sufficient_evidence"):
    retrieval_calls = retrieval_calls if retrieval_calls is not None else []

    def fake_retrieval(query, **kwargs):
        retrieval_calls.append((query, kwargs))
        return make_retrieval(query)

    def fake_brief(response, confidence=None):
        status = "generated" if confidence is None or confidence.decision == "sufficient_evidence" else "skipped_low_confidence"
        return make_brief(response.query, status=status)

    monkeypatch.setattr(api_main, "run_unified_search", fake_retrieval)
    monkeypatch.setattr(api_main, "assess_confidence", lambda response: make_confidence(response.query, confidence_decision))
    monkeypatch.setattr(api_main, "build_research_brief", fake_brief)
    monkeypatch.setattr(api_main, "build_evidence_matrix", lambda response, brief=None, max_rows=10: make_matrix(response.query))
    monkeypatch.setattr(api_main, "build_reading_path", lambda response, confidence=None, max_papers=8: make_reading_path(response.query))
    monkeypatch.setattr(
        api_main,
        "build_open_problems_report",
        lambda response, confidence=None, max_problems=6: make_open_problems(response.query),
    )
    return retrieval_calls


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "degraded"}
    assert payload["service"] == "research-synthesis-engine"
    assert "dependencies" in payload
    assert "OPENAI_API_KEY" not in str(payload)


def test_corpus_stats_endpoint_has_expected_keys():
    response = client.get("/corpus/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["topics"] == 5
    assert "research_papers" in payload["qdrant_collections"]


def test_retrieve_endpoint_uses_unified_search_once(monkeypatch):
    calls = patch_core_services(monkeypatch)

    response = client.post("/retrieve", json={"question": "What reduces hallucinations?", "top_k": 5})

    assert response.status_code == 200
    assert response.json()["paper_result_count"] == 1
    assert response.json()["question"] == "What reduces hallucinations?"
    assert len(calls) == 1
    assert calls[0][1]["top_k"] == 5


def test_confidence_endpoint_returns_assessment(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/confidence", json={"question": "What reduces hallucinations?"})

    assert response.status_code == 200
    assert response.json()["decision"] == "sufficient_evidence"


def test_brief_endpoint_returns_research_brief(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/brief", json={"query": "What reduces hallucinations?"})

    assert response.status_code == 200
    assert response.json()["status"] == "generated"
    assert response.json()["themes"][0]["supporting_source_ids"] == ["paper:p1"]


def test_evidence_matrix_endpoint_returns_rows(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/evidence-matrix", json={"query": "What reduces hallucinations?"})

    assert response.status_code == 200
    assert response.json()["rows"][0]["source_ids"] == ["paper:p1"]


def test_reading_path_endpoint_returns_stages(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/reading-path", json={"query": "What reduces hallucinations?", "max_papers": 3})

    assert response.status_code == 200
    assert response.json()["total_papers"] == 1
    assert response.json()["stages"][0]["stage"] == "foundational"


def test_open_problems_endpoint_returns_problems(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/open-problems", json={"query": "What remains unresolved?", "max_problems": 3})

    assert response.status_code == 200
    assert response.json()["problems"][0]["category"] == "evaluation"


def test_guidance_endpoint_reuses_one_retrieval_response(monkeypatch):
    calls = patch_core_services(monkeypatch)

    response = client.post("/guidance", json={"query": "Compare RAG and verification.", "top_k": 4})

    assert response.status_code == 200
    payload = response.json()
    assert payload["brief"]["status"] == "generated"
    assert payload["evidence_matrix"]["rows"][0]["source_ids"] == ["paper:p1"]
    assert payload["reading_path"]["total_papers"] == 1
    assert payload["open_problems"]["problems"][0]["supporting_source_ids"] == ["paper:p1"]
    assert len(calls) == 1


def test_guidance_endpoint_can_skip_heavy_optional_sections(monkeypatch):
    patch_core_services(monkeypatch)

    def fail_reading_path(*args, **kwargs):
        raise AssertionError("reading path should not be generated")

    def fail_open_problems(*args, **kwargs):
        raise AssertionError("open problems should not be generated")

    monkeypatch.setattr(api_main, "build_reading_path", fail_reading_path)
    monkeypatch.setattr(api_main, "build_open_problems_report", fail_open_problems)

    response = client.post(
        "/guidance",
        json={
            "query": "Compare RAG and verification.",
            "include_reading_path": False,
            "include_open_problems": False,
            "include_debug": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["brief"]["status"] == "generated"
    assert payload["evidence_matrix"]["rows"]
    assert payload["reading_path"] is None
    assert payload["open_problems"] is None
    assert payload["metrics"].get("reading_path_ms") is None
    assert payload["metrics"].get("open_problems_ms") is None


def test_guidance_endpoint_builds_optional_sections_concurrently(monkeypatch):
    patch_core_services(monkeypatch)

    def slow_matrix(response, brief=None, max_rows=10):
        time.sleep(0.2)
        return make_matrix(response.query)

    def slow_reading_path(response, confidence=None, max_papers=8):
        time.sleep(0.2)
        return make_reading_path(response.query)

    def slow_open_problems(response, confidence=None, max_problems=6):
        time.sleep(0.2)
        return make_open_problems(response.query)

    monkeypatch.setattr(api_main, "build_evidence_matrix", slow_matrix)
    monkeypatch.setattr(api_main, "build_reading_path", slow_reading_path)
    monkeypatch.setattr(api_main, "build_open_problems_report", slow_open_problems)

    started = time.perf_counter()
    response = client.post("/guidance", json={"query": "Compare RAG and verification.", "include_debug": True})
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    payload = response.json()
    assert elapsed < 0.45
    assert payload["metrics"]["evidence_matrix_ms"] >= 190
    assert payload["metrics"]["reading_path_ms"] >= 190
    assert payload["metrics"]["open_problems_ms"] >= 190


def test_guidance_endpoint_keeps_brief_when_optional_section_fails(monkeypatch):
    patch_core_services(monkeypatch)

    def fail_matrix(response, brief=None, max_rows=10):
        raise api_main.EvidenceMatrixError("matrix JSON was malformed")

    monkeypatch.setattr(api_main, "build_evidence_matrix", fail_matrix)

    response = client.post("/guidance", json={"query": "Compare agents and RAG.", "top_k": 4})

    assert response.status_code == 200
    payload = response.json()
    assert payload["brief"]["status"] == "generated"
    assert payload["evidence_matrix"] is None
    assert payload["reading_path"]["total_papers"] == 1
    assert payload["open_problems"]["problems"]
    assert any("Evidence matrix could not be generated" in warning for warning in payload["warnings"])


def test_guidance_endpoint_skips_day19_when_confidence_is_low(monkeypatch):
    patch_core_services(monkeypatch, confidence_decision="broaden_search")

    response = client.post("/guidance", json={"query": "Too broad?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["brief"]["status"] == "skipped_low_confidence"
    assert payload["evidence_matrix"] is None
    assert payload["reading_path"] is None
    assert payload["open_problems"] is None
    assert payload["warnings"]


def test_agent_research_endpoint_returns_trace_and_retry_metadata(monkeypatch):
    def fail_retrieval(*args, **kwargs):
        raise AssertionError("agent endpoint test should use the mocked graph runner")

    def fake_agent(query, **kwargs):
        assert query == "What are its limitations?"
        assert kwargs["chat_history"][0].content == "Explain AI agents."
        retrieval = make_retrieval("What are the limitations of AI agents?")
        confidence = make_confidence(retrieval.query)
        return {
            "original_query": query,
            "chat_history": kwargs["chat_history"],
            "standalone_query": retrieval.query,
            "retrieved_papers": retrieval.paper_results,
            "retrieved_chunks": retrieval.chunk_results,
            "confidence_decision": "sufficient_evidence",
            "retry_count": 1,
            "retrieval_response": retrieval,
            "confidence": confidence,
            "brief": make_brief(retrieval.query),
            "attempted_queries": [query, retrieval.query],
            "warnings": ["Retry 1 used expanded query after low confidence."],
        }

    monkeypatch.setattr(api_main, "run_unified_search", fail_retrieval)
    monkeypatch.setattr(api_main, "run_research_agent", fake_agent)

    response = client.post(
        "/agent/research",
        json={
            "question": "What are its limitations?",
            "chat_history": [{"role": "user", "content": "Explain AI agents."}],
            "include_debug": True,
        },
        headers={"X-Request-ID": "agent-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "agent-request-123"
    payload = response.json()
    assert payload["original_query"] == "What are its limitations?"
    assert payload["standalone_query"] == "What are the limitations of AI agents?"
    assert payload["retry_count"] == 1
    assert payload["retrieved_paper_count"] == 1
    assert payload["confidence_decision"] == "sufficient_evidence"
    assert payload["retrieval"]["question"] == "What are the limitations of AI agents?"
    assert payload["brief"]["status"] == "generated"
    assert payload["trace"][0]["step"] == "Context rewrite"
    assert any(item["step"] == "CRAG retry 1" for item in payload["trace"])
    assert payload["debug"]["attempted_queries"] == ["What are its limitations?", "What are the limitations of AI agents?"]


def test_agent_research_endpoint_returns_structured_error(monkeypatch):
    def fail_agent(query, **kwargs):
        raise api_main.SynthesisError("provider returned malformed JSON")

    monkeypatch.setattr(api_main, "run_research_agent", fail_agent)

    response = client.post(
        "/agent/research",
        json={"question": "Compare RAG and agents."},
        headers={"X-Request-ID": "agent-error-123"},
    )

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "agent-error-123"
    payload = response.json()
    assert payload["error"]["code"] == "LLM_GENERATION_FAILED"
    assert payload["error"]["request_id"] == "agent-error-123"


def test_invalid_query_returns_422():
    response = client.post("/retrieve", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_service_error_maps_to_503(monkeypatch):
    def fail_retrieval(query, **kwargs):
        raise api_main.UnifiedSearchError("Qdrant unavailable")

    monkeypatch.setattr(api_main, "run_unified_search", fail_retrieval)

    response = client.post("/retrieve", json={"query": "What reduces hallucinations?"})

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "QDRANT_UNAVAILABLE"
    assert "Qdrant unavailable" in payload["detail"]


def test_query_alias_still_works_for_backward_compatibility(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/retrieve", json={"query": "What reduces hallucinations?"})

    assert response.status_code == 200
    assert response.json()["question"] == "What reduces hallucinations?"



def test_guidance_uses_rewritten_query_for_contextual_followup(monkeypatch):
    calls = patch_core_services(monkeypatch)

    def fake_rewrite(question, chat_history):
        assert question == "What are its limitations?"
        assert chat_history[0].content == "Explain LoRA fine-tuning."
        return api_main.QueryRewriteResult(
            original_query=question,
            standalone_query="What are the limitations of LoRA fine-tuning?",
            rewrite_used=True,
            method="llm",
            reason="Resolved its to LoRA fine-tuning.",
        )

    monkeypatch.setattr(api_main, "rewrite_query", fake_rewrite)

    response = client.post(
        "/guidance",
        json={
            "question": "What are its limitations?",
            "chat_history": [{"role": "user", "content": "Explain LoRA fine-tuning."}],
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert calls[0][0] == "What are the limitations of LoRA fine-tuning?"
    assert payload["question"] == "What are its limitations?"
    assert payload["standalone_query"] == "What are the limitations of LoRA fine-tuning?"
    assert payload["rewrite_used"] is True
    assert payload["rewrite_method"] == "llm"
    assert payload["retrieval"]["question"] == "What are the limitations of LoRA fine-tuning?"

def test_route_preview_does_not_run_retrieval(monkeypatch):
    def fail_retrieval(*args, **kwargs):
        raise AssertionError("route preview must not run retrieval")

    monkeypatch.setattr(api_main, "run_unified_search", fail_retrieval)

    response = client.post("/route", json={"question": "Compare RAG and self-verification methods."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_route"] == "hybrid_both"
    assert payload["route_confidence"] > 0
    assert payload["matched_signals"]


def test_valid_filters_are_applied_with_warnings(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post(
        "/retrieve",
        json={
            "question": "What reduces hallucinations?",
            "research_areas": ["Retrieval-Augmented Generation (RAG)"],
            "publication_year_min": 2020,
            "publication_year_max": 2025,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_result_count"] == 1
    assert any("Filters are applied after retrieval" in warning for warning in payload["warnings"])


def test_invalid_research_area_returns_structured_validation_error():
    response = client.post("/retrieve", json={"question": "What?", "research_areas": ["Quantum Baking"]})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_invalid_year_range_returns_structured_validation_error():
    response = client.post(
        "/retrieve",
        json={"question": "What reduces hallucinations?", "publication_year_min": 2025, "publication_year_max": 2020},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_full_text_only_filter_warns_and_omits_paper_results(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/retrieve", json={"question": "What reduces hallucinations?", "full_text_only": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_result_count"] == 0
    assert any("full_text_only" in warning for warning in payload["warnings"])


def test_debug_false_omits_score_breakdowns_and_metrics(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/retrieve", json={"question": "What reduces hallucinations?", "include_debug": False})

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"] is None
    assert payload["debug"] is None
    assert payload["paper_results"][0]["score_breakdown"] is None
    assert payload["route"]["matched_signals"] == []


def test_debug_true_includes_signals_score_breakdowns_and_metrics(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/guidance", json={"question": "What reduces hallucinations?", "include_debug": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["total_ms"] >= 0
    assert payload["debug"]["matched_signals"] == ["main: main ideas question"]
    assert payload["debug"]["confidence_signals"] == ["top_score=0.95"]
    assert payload["debug"]["score_breakdowns"]["p1"]["rerank_score"] == 0.9
    assert payload["retrieval"]["metrics"]["retrieval_ms"] >= 0


def test_request_id_is_returned_for_success_response(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post(
        "/retrieve",
        json={"question": "What reduces hallucinations?"},
        headers={"X-Request-ID": "test-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-123"


def test_request_id_is_returned_for_error_response(monkeypatch):
    def fail_retrieval(query, **kwargs):
        raise api_main.UnifiedSearchError("Qdrant unavailable")

    monkeypatch.setattr(api_main, "run_unified_search", fail_retrieval)

    response = client.post(
        "/retrieve",
        json={"question": "What reduces hallucinations?"},
        headers={"X-Request-ID": "error-request-123"},
    )

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "error-request-123"
    assert response.json()["error"]["request_id"] == "error-request-123"


def test_request_id_is_generated_when_missing(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/retrieve", json={"question": "What reduces hallucinations?"})

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")


def test_cors_allows_local_streamlit_origin():
    response = client.options(
        "/guidance",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"


def test_openapi_includes_route_and_guidance_summaries():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/route" in paths
    assert paths["/guidance"]["post"]["summary"] == "Generate the complete research analyst response"

