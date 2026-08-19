import pytest
from agentic.planner import plan_query
def test_default_routes_to_corpus():
    assert plan_query("How does LoRA reduce trainable parameters?").route == "corpus"
def test_latest_routes_to_live():
    assert plan_query("What are the latest retrieval papers?").route == "live"
def test_current_comparison_routes_to_hybrid():
    plan = plan_query("Compare our corpus with the latest retrieval papers.")
    assert plan.route == "hybrid"
    assert "search_local_corpus" in plan.tools
def test_blank_query_rejected():
    with pytest.raises(ValueError): plan_query(" ")
