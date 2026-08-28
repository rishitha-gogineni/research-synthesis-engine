"""End-to-end test for the multi-agent research pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from multi_agent.config import classify_effort, EFFORT_SIMPLE, EFFORT_MODERATE, EFFORT_COMPLEX
from multi_agent.findings_store import FindingsStore, Finding, SubagentResult
from multi_agent.trace import Tracer


class TestEffortClassification:
    def test_simple_query(self):
        assert classify_effort("What is RAG?").name == "simple"
        assert classify_effort("Define transformers").name == "simple"

    def test_moderate_query(self):
        result = classify_effort("How do transformer models handle long sequences in practice?")
        assert result.name == "moderate"

    def test_complex_query(self):
        result = classify_effort("Compare and contrast the trade-offs between dense and sparse retrieval methods")
        assert result.name == "complex"


class TestFindingsStore:
    def test_store_and_retrieve(self):
        store = FindingsStore()
        agent_id = store.create_agent_id("arxiv")
        result = SubagentResult(
            agent_id=agent_id,
            agent_type="arxiv",
            subtask="Find RAG papers",
            status="complete",
            findings=[
                Finding(source="arxiv", title="Paper 1", content="Abstract 1"),
                Finding(source="arxiv", title="Paper 2", content="Abstract 2"),
            ],
            summary="Found 2 papers",
            queries_used=["RAG survey"],
            tool_calls_count=2,
        )
        store.store(result)

        assert len(store.get_all()) == 1
        assert len(store.get_completed()) == 1
        assert len(store.get_all_findings()) == 2

    def test_summary(self):
        store = FindingsStore()
        store.store(SubagentResult(
            agent_id="test_1", agent_type="arxiv", subtask="t1",
            status="complete", findings=[Finding("arxiv", "P1", "C1")],
        ))
        store.store(SubagentResult(
            agent_id="test_2", agent_type="web", subtask="t2",
            status="failed", error="timeout",
        ))

        summary = store.summary()
        assert summary["total_agents"] == 2
        assert summary["completed"] == 1
        assert summary["failed"] == 1


class TestTracer:
    def test_log_and_retrieve(self):
        tracer = Tracer()
        tracer.log("agent_1", "search", query="test")
        tracer.log("agent_1", "evaluate", result="good")
        tracer.log("agent_2", "search", query="other")

        assert len(tracer.get_events()) == 3
        assert len(tracer.get_events("agent_1")) == 2

    def test_summary(self):
        tracer = Tracer()
        tracer.log("lead", "plan_start")
        tracer.log("sub_1", "search")
        tracer.log("sub_1", "complete")

        summary = tracer.summary()
        assert summary["total_events"] == 3
        assert "lead" in summary["agents_involved"]


class TestSubagent:
    @patch("multi_agent.subagent.OpenAI")
    @patch("multi_agent.subagent._search_source")
    def test_run_subagent_basic(self, mock_search, mock_openai):
        from multi_agent.subagent import run_subagent

        mock_search.return_value = [
            Finding(source="arxiv", title="Test Paper", content="Test abstract")
        ]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "sufficient": True,
            "reasoning": "Found what we need",
        })
        mock_client.chat.completions.create.return_value = mock_response

        store = FindingsStore()
        tracer = Tracer()
        subtask = {
            "source": "arxiv",
            "objective": "Find papers on RAG",
            "queries": ["RAG retrieval"],
        }

        result = run_subagent(
            subtask, store, tracer,
            openai_client=mock_client,
            external_client=MagicMock(),
            max_tool_calls=5,
        )

        assert result.status == "complete"
        assert len(result.findings) == 1
        assert result.findings[0].title == "Test Paper"


class TestLeadAgent:
    @patch("multi_agent.lead.OpenAI")
    def test_create_plan(self, mock_openai):
        from multi_agent.lead import create_plan

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "reasoning": "Need to search multiple sources",
            "subtasks": [
                {
                    "id": "subtask_1",
                    "objective": "Search local corpus for RAG papers",
                    "source": "local_corpus",
                    "queries": ["RAG retrieval"],
                    "boundaries": "Only academic papers",
                    "output_format": "List of papers",
                }
            ],
        })
        mock_client.chat.completions.create.return_value = mock_response

        tracer = Tracer()
        plan = create_plan("What is RAG?", tracer, client=mock_client)

        assert "subtasks" in plan
        assert len(plan["subtasks"]) == 1
        assert plan["subtasks"][0]["source"] == "local_corpus"


class TestEndToEnd:
    @patch("multi_agent.orchestrator.OpenAI")
    @patch("multi_agent.subagent._search_source")
    def test_full_pipeline_mocked(self, mock_search, mock_openai_cls):
        from multi_agent.orchestrator import run_research

        mock_search.return_value = [
            Finding(source="arxiv", title="RAG Paper", content="About RAG systems")
        ]

        # Mock all LLM calls
        mock_client = MagicMock()

        # Plan response
        plan_response = MagicMock()
        plan_response.choices = [MagicMock()]
        plan_response.choices[0].message.content = json.dumps({
            "reasoning": "Simple query",
            "subtasks": [{
                "id": "s1", "objective": "Find RAG info",
                "source": "arxiv", "queries": ["RAG"],
                "boundaries": "", "output_format": "papers",
            }],
        })

        # Evaluate response (sufficient)
        eval_response = MagicMock()
        eval_response.choices = [MagicMock()]
        eval_response.choices[0].message.content = json.dumps({
            "sufficient": True, "reasoning": "Found enough",
        })

        # Synthesis response
        synth_response = MagicMock()
        synth_response.choices = [MagicMock()]
        synth_response.choices[0].message.content = json.dumps({
            "synthesis": "RAG combines retrieval with generation.",
            "key_themes": ["retrieval", "generation"],
            "sources_used": [{"title": "RAG Paper", "source": "arxiv", "url": ""}],
            "gaps": [],
            "confidence": "high",
            "needs_more_research": False,
            "follow_up_subtasks": [],
        })

        # Citation response
        cite_response = MagicMock()
        cite_response.choices = [MagicMock()]
        cite_response.choices[0].message.content = json.dumps({
            "cited_report": "RAG combines retrieval with generation [1].",
            "references": [{"id": 1, "title": "RAG Paper", "source": "arxiv", "url": ""}],
            "uncited_claims": [],
        })

        # Judge response
        judge_response = MagicMock()
        judge_response.choices = [MagicMock()]
        judge_response.choices[0].message.content = json.dumps({
            "factual_accuracy": 0.9,
            "citation_accuracy": 0.8,
            "completeness": 0.7,
            "source_quality": 0.85,
            "tool_efficiency": 0.9,
            "overall": 0.83,
            "pass": True,
            "reasoning": "Good quality research output",
        })

        mock_client.chat.completions.create.side_effect = [
            plan_response,
            eval_response,
            synth_response,
            cite_response,
            judge_response,
        ]
        mock_openai_cls.return_value = mock_client

        result = run_research("What is RAG?", openai_client=mock_client)

        assert result["query"] == "What is RAG?"
        assert result["synthesis"]["confidence"] == "high"
        assert result["cited_report"]["references"][0]["title"] == "RAG Paper"
        assert result["judge_scores"]["overall"] == 0.83
        assert result["judge_scores"]["pass"] is True
