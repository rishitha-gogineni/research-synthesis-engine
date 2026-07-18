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
    )
    return UnifiedSearchResponse(
        query=query,
        route=QueryRoute(query=query, route="paper_level", reason="test route", confidence=0.95),
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

    monkeypatch.setattr(api_main, "run_unified_search", fake_retrieval)
    monkeypatch.setattr(api_main, "assess_confidence", lambda response: make_confidence(response.query, confidence_decision))
    monkeypatch.setattr(api_main, "build_research_brief", lambda response, confidence=None: make_brief(response.query))
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
    assert response.json()["status"] == "ok"


def test_corpus_stats_endpoint_has_expected_keys():
    response = client.get("/corpus/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["topics"] == 5
    assert "research_papers" in payload["qdrant_collections"]


def test_retrieve_endpoint_uses_unified_search_once(monkeypatch):
    calls = patch_core_services(monkeypatch)

    response = client.post("/retrieve", json={"query": "What reduces hallucinations?", "top_k": 5})

    assert response.status_code == 200
    assert response.json()["paper_result_count"] == 1
    assert len(calls) == 1
    assert calls[0][1]["top_k"] == 5


def test_confidence_endpoint_returns_assessment(monkeypatch):
    patch_core_services(monkeypatch)

    response = client.post("/confidence", json={"query": "What reduces hallucinations?"})

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


def test_guidance_endpoint_skips_day19_when_confidence_is_low(monkeypatch):
    patch_core_services(monkeypatch, confidence_decision="broaden_search")

    response = client.post("/guidance", json={"query": "Too broad?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["reading_path"] is None
    assert payload["open_problems"] is None
    assert payload["warnings"]


def test_invalid_query_returns_422():
    response = client.post("/retrieve", json={"query": "   "})

    assert response.status_code == 422


def test_service_error_maps_to_503(monkeypatch):
    def fail_retrieval(query, **kwargs):
        raise api_main.UnifiedSearchError("Qdrant unavailable")

    monkeypatch.setattr(api_main, "run_unified_search", fail_retrieval)

    response = client.post("/retrieve", json={"query": "What reduces hallucinations?"})

    assert response.status_code == 503
    assert "Qdrant unavailable" in response.json()["detail"]
