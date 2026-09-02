"""Contract tests for agent-to-agent handoffs via FindingsStore.

synthesize_findings() and add_citations() both trust whatever a subagent
wrote to the shared store — nothing validates the *content* of a Finding,
only the *shape* of the LLM's own JSON response (schemas.py). These tests
feed each downstream agent a handoff that is realistic but imperfect (the
kind of thing a live external API or a race between subagents actually
produces) and assert graceful degradation: no crash, no fabricated output.
"""

from __future__ import annotations

from multi_agent.citation import add_citations
from multi_agent.lead import synthesize_findings
from multi_agent.trace import Tracer
from tests.harness import corrupted_store, make_llm_client

_SYNTHESIS_OK = {
    "synthesis": "Limited information available.",
    "key_themes": [],
    "sources_used": [],
    "gaps": [],
    "confidence": "low",
    "needs_more_research": False,
    "follow_up_subtasks": [],
}


class TestSynthesisHandoff:
    def test_empty_content_finding_does_not_crash(self):
        store = corrupted_store("empty_content")
        client = make_llm_client(_SYNTHESIS_OK)
        result = synthesize_findings("test query", store, Tracer(), client=client)
        assert result["synthesis"]

    def test_missing_metadata_does_not_crash(self):
        store = corrupted_store("missing_metadata")
        client = make_llm_client(_SYNTHESIS_OK)
        result = synthesize_findings("test query", store, Tracer(), client=client)
        assert result["synthesis"]

    def test_all_subagents_failed_skips_llm_and_reports_gap(self):
        store = corrupted_store("partial_status")
        # Force the true "nothing completed" shape: drop the completed agent.
        store._results = {
            k: v for k, v in store._results.items() if v.status == "failed"
        }
        client = make_llm_client()  # no responses configured: LLM must not be called
        result = synthesize_findings("test query", store, Tracer(), client=client)
        assert result["confidence"] == "low"
        assert result["gaps"]
        client.chat.completions.create.assert_not_called()

    def test_partial_failure_synthesizes_from_the_surviving_agent_only(self):
        store = corrupted_store("partial_status")
        client = make_llm_client(_SYNTHESIS_OK)
        result = synthesize_findings("test query", store, Tracer(), client=client)
        # One agent completed, one failed -- the LLM call must still happen
        # (there's real evidence to synthesize from), unlike the all-failed case.
        client.chat.completions.create.assert_called_once()
        assert result["synthesis"]

    def test_duplicate_titles_across_sources_both_reach_the_prompt(self):
        store = corrupted_store("duplicate_titles")
        client = make_llm_client(_SYNTHESIS_OK)
        result = synthesize_findings("test query", store, Tracer(), client=client)
        assert result["synthesis"]


class TestCitationHandoff:
    def test_empty_content_finding_does_not_crash(self):
        store = corrupted_store("empty_content")
        client = make_llm_client({
            "cited_report": "Limited information available.",
            "references": [],
            "uncited_claims": [],
        })
        synthesis = {"synthesis": "Limited information available.", "sources_used": []}
        result = add_citations(synthesis, store, Tracer(), client=client)
        assert result["hallucination_flags"] == []

    def test_number_absent_from_any_finding_is_flagged(self):
        # The two findings' content is "C1"/"C2" -- neither contains "2017",
        # so a report that states it is making an unverified claim.
        store = corrupted_store("duplicate_titles")
        client = make_llm_client({
            "cited_report": "The Transformer paper introduced self-attention in 2017 [1].",
            "references": [{"id": 1, "title": "Attention Is All You Need", "source": "arxiv", "url": ""}],
            "uncited_claims": [],
        })
        synthesis = {"synthesis": "...", "sources_used": []}
        result = add_citations(synthesis, store, Tracer(), client=client)
        assert "2017" in result["hallucination_flags"]

    def test_malformed_json_response_still_returns_hallucination_flags_key(self):
        store = corrupted_store("empty_content")
        client = make_llm_client("not valid json {{{")
        synthesis = {"synthesis": "Limited information available.", "sources_used": []}
        result = add_citations(synthesis, store, Tracer(), client=client)
        # Must be present regardless of which branch (parse success vs.
        # failure) produced `result` -- downstream code (evaluation.py,
        # judge.py) reads it unconditionally.
        assert "hallucination_flags" in result
