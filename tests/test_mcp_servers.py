from mcp_servers.research_tools import MCP_AVAILABLE, tool_names
def test_mcp_server_exposes_all_research_tools():
    assert MCP_AVAILABLE
    assert tool_names() == ["search_local_corpus", "search_arxiv", "search_semantic_scholar", "search_tavily"]
