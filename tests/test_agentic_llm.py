from types import SimpleNamespace
import pytest
from agentic.llm import LLMResult, ToolCallingError, run_grounded_answer
class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)
class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))
def response(content="", tool_calls=None, prompt_tokens=3, completion_tokens=4):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
def test_grounded_answer_extracts_citations_and_usage():
    client = FakeClient([response("The answer is supported [source_1].")])
    result = run_grounded_answer("What is RAG?", [{"kind": "chunk", "text": "evidence"}], client=client)
    assert result.answer.endswith("[source_1].")
    assert result.citations == ["source_1"]
    assert result.usage["total_tokens"] == 7
    assert client.chat.completions.calls[0]["tools"]
def test_tool_call_is_executed_and_traced():
    client = FakeClient([response(tool_calls=[tool_call("search_arxiv", '{"query":"rag"}')]), response("Current evidence [source_1].")])
    seen = []
    def executor(name, arguments):
        seen.append((name, arguments))
        return {"results": [{"title": "RAG"}]}
    result = run_grounded_answer("latest RAG papers", [{"kind": "external", "title": "seed"}], client=client, executor=executor, max_tool_calls=2)
    assert seen == [("search_arxiv", {"query": "rag"})]
    assert result.tool_calls[0]["status"] == "completed"
    assert len(client.chat.completions.calls) == 2
def test_bad_tool_arguments_are_reported_without_crashing():
    client = FakeClient([response(tool_calls=[tool_call("search_arxiv", "not-json")]), response("Insufficient evidence.")])
    result = run_grounded_answer("latest papers", [], client=client)
    assert result.tool_calls[0]["status"] == "failed"
    assert result.answer == "Insufficient evidence."
def test_missing_key_is_safe(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ToolCallingError, match="OPENAI_API_KEY"):
        run_grounded_answer("question", [])
