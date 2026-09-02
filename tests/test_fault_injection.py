"""Fault-injection tests for subagent search failures.

Layer 1/3 tests already cover the happy path (a search succeeds). These
deliberately break a source mid-run -- a provider timing out, an API
erroring out -- and assert the recovery path (SOURCE_FALLBACKS in
multi_agent/subagent.py) actually gets exercised, and that a source with no
configured fallback fails cleanly rather than propagating an exception.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agentic.external import ExternalPaper
from multi_agent.findings_store import Finding, FindingsStore
from multi_agent.subagent import run_subagent
from multi_agent.trace import Tracer
from tests.harness import FailingExternalClient, make_llm_client

_EVAL_SUFFICIENT = {"sufficient": True, "reasoning": "Found enough"}


class TestSubagentFallback:
    def test_primary_source_failure_falls_back_and_still_completes(self):
        # arxiv fails outright; semantic_scholar (its configured fallback) succeeds.
        external = FailingExternalClient(
            fail_sources=["arxiv"],
            papers_by_source={
                "semantic_scholar": [
                    ExternalPaper(
                        source="semantic_scholar", paper_id="p1",
                        title="Fallback Paper", abstract="Found via fallback",
                    )
                ]
            },
        )
        openai_client = make_llm_client(_EVAL_SUFFICIENT)
        subtask = {"source": "arxiv", "objective": "Find X", "queries": ["query one"]}

        result = run_subagent(
            subtask, FindingsStore(), Tracer(),
            openai_client=openai_client, external_client=external, max_tool_calls=5,
        )

        assert result.status == "complete"
        assert len(result.findings) == 1
        assert result.findings[0].source == "semantic_scholar"
        assert result.findings[0].title == "Fallback Paper"

    def test_all_sources_including_fallbacks_fail_reports_failed_status(self):
        # A total wipeout across the primary source and every fallback must
        # not self-report as "complete" -- get_completed() elsewhere relies
        # on status meaning "has usable findings".
        external = FailingExternalClient(fail_sources=["arxiv", "semantic_scholar", "web"])
        openai_client = make_llm_client()  # must not be reached -- no results to evaluate
        subtask = {"source": "arxiv", "objective": "Find X", "queries": ["query one"]}

        result = run_subagent(
            subtask, FindingsStore(), Tracer(),
            openai_client=openai_client, external_client=external, max_tool_calls=5,
        )

        assert result.status == "failed"
        assert result.findings == []
        openai_client.chat.completions.create.assert_not_called()

    def test_source_with_no_fallback_fails_cleanly(self):
        # "web" has no configured fallback (SOURCE_FALLBACKS["web"] == []) --
        # a Tavily failure must not raise past run_subagent.
        external = FailingExternalClient(fail_sources=["web"])
        openai_client = make_llm_client()
        subtask = {"source": "web", "objective": "Find X", "queries": ["query one"]}

        result = run_subagent(
            subtask, FindingsStore(), Tracer(),
            openai_client=openai_client, external_client=external, max_tool_calls=5,
        )

        assert result.status == "failed"
        assert result.findings == []


class TestOrchestratorPartialFailure:
    @patch("multi_agent.orchestrator.OpenAI")
    @patch("multi_agent.subagent._search_source")
    def test_one_source_fails_pipeline_still_synthesizes_from_the_other(
        self, mock_search, mock_openai_cls
    ):
        from multi_agent.orchestrator import run_research

        def search_side_effect(source, query, client, max_results=5):
            if source == "web":
                raise RuntimeError("simulated web outage")
            return [Finding(source="arxiv", title="Surviving Paper", content="About X")]

        mock_search.side_effect = search_side_effect

        mock_client = MagicMock()
        plan_response = MagicMock()
        plan_response.choices = [MagicMock()]
        plan_response.choices[0].message.content = json.dumps({
            "reasoning": "Need two sources",
            "subtasks": [
                {"id": "s1", "objective": "Find X", "source": "arxiv",
                 "queries": ["X"], "boundaries": "", "output_format": "papers"},
                {"id": "s2", "objective": "Find X live", "source": "web",
                 "queries": ["X"], "boundaries": "", "output_format": "papers"},
            ],
        })

        eval_response = MagicMock()
        eval_response.choices = [MagicMock()]
        eval_response.choices[0].message.content = json.dumps(_EVAL_SUFFICIENT)

        synth_response = MagicMock()
        synth_response.choices = [MagicMock()]
        synth_response.choices[0].message.content = json.dumps({
            "synthesis": "Found information about X from one source.",
            "key_themes": ["X"], "sources_used": [], "gaps": [],
            "confidence": "medium", "needs_more_research": False,
            "follow_up_subtasks": [],
        })

        cite_response = MagicMock()
        cite_response.choices = [MagicMock()]
        cite_response.choices[0].message.content = json.dumps({
            "cited_report": "Found information about X from one source [1].",
            "references": [{"id": 1, "title": "Surviving Paper", "source": "arxiv", "url": ""}],
            "uncited_claims": [],
        })

        judge_response = MagicMock()
        judge_response.choices = [MagicMock()]
        judge_response.choices[0].message.content = json.dumps({
            "factual_accuracy": 0.8, "citation_accuracy": 0.8, "completeness": 0.6,
            "source_quality": 0.7, "tool_efficiency": 0.8, "overall": 0.7,
            "pass": True, "reasoning": "Partial but honest coverage",
        })

        # "web" has no fallback: run_subagent's except-branch `continue`s
        # straight past the evaluate step when there's no fallback to try,
        # so only the surviving "arxiv" subagent ever calls _evaluate_results.
        mock_client.chat.completions.create.side_effect = [
            plan_response, eval_response,
            synth_response, cite_response, judge_response,
        ]
        mock_openai_cls.return_value = mock_client

        # A "simple"-effort query caps to 1 subagent regardless of plan
        # content (multi_agent/config.py), which would silently drop the
        # second subtask before it ever gets a chance to fail. Use a query
        # that classifies as "moderate" (max_subagents=3) so both subtasks
        # in the mocked plan actually run.
        result = run_research(
            "How do transformer models handle long sequences in practice?",
            openai_client=mock_client,
        )

        assert result["synthesis"]["synthesis"]
        agents = result["store_summary"]["agents"]
        assert any(a["status"] == "complete" and a["findings"] for a in agents)
        assert any(a["agent_type"] == "web" and not a["findings"] for a in agents)
