import pytest
from agentic.tools import AgentToolError, execute_tool, tool_definitions
def test_tool_schema_is_openai_compatible():
    tool = tool_definitions()[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "search_local_corpus"
    assert "query" in tool["function"]["parameters"]["required"]
def test_tool_arguments_are_validated():
    with pytest.raises(AgentToolError): execute_tool("search_local_corpus", {"query": ""})
    with pytest.raises(AgentToolError): execute_tool("unknown", {"query": "x"})
def test_tool_clamps_top_k():
    seen = {}
    def fake(query, top_k):
        seen.update(query=query, top_k=top_k)
        return {"ok": True}
    assert execute_tool("search_local_corpus", {"query": "x", "top_k": 100}, searcher=fake) == {"ok": True}
    assert seen == {"query": "x", "top_k": 50}
