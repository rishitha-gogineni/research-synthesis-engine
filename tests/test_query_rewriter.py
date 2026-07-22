from agent.query_rewriter import ChatTurn, heuristic_rewrite, rewrite_query


def test_no_chat_history_keeps_original_query():
    result = rewrite_query("What are the limitations of LoRA?")

    assert result.standalone_query == "What are the limitations of LoRA?"
    assert result.rewrite_used is False
    assert result.method == "none"


def test_llm_rewriter_runs_before_heuristic_when_history_exists():
    history = [ChatTurn(role="user", content="Explain LoRA fine-tuning.")]

    def fake_generator(prompt: str) -> str:
        assert "Explain LoRA fine-tuning" in prompt
        return '{"standalone_query":"What are the limitations of LoRA fine-tuning in AI research papers?","rewrite_used":true,"reason":"Resolved its to LoRA fine-tuning."}'

    result = rewrite_query("What are its limitations?", history, generator=fake_generator)

    assert result.standalone_query == "What are the limitations of LoRA fine-tuning in AI research papers?"
    assert result.rewrite_used is True
    assert result.method == "llm"


def test_heuristic_fallback_rewrites_contextual_followup_when_llm_fails():
    history = [ChatTurn(role="user", content="Explain LoRA fine-tuning.")]

    def failing_generator(prompt: str) -> str:
        raise RuntimeError("provider unavailable")

    result = rewrite_query("What are its limitations?", history, generator=failing_generator)

    assert "LoRA fine-tuning" in result.standalone_query
    assert "limitations" in result.standalone_query.lower()
    assert result.rewrite_used is True
    assert result.method == "heuristic"


def test_heuristic_does_not_rewrite_standalone_question():
    history = [ChatTurn(role="user", content="Explain LoRA fine-tuning.")]

    result = heuristic_rewrite("What are the main approaches for reducing hallucinations in LLMs?", history)

    assert result.standalone_query == "What are the main approaches for reducing hallucinations in LLMs?"
    assert result.rewrite_used is False

def test_standalone_question_with_history_skips_llm_generator():
    history = [ChatTurn(role="user", content="Explain LoRA fine-tuning.")]

    def forbidden_generator(prompt: str) -> str:
        raise AssertionError("standalone question should not call LLM rewriter")

    result = rewrite_query(
        "What are the main approaches for reducing hallucinations in LLMs?",
        history,
        generator=forbidden_generator,
    )

    assert result.standalone_query == "What are the main approaches for reducing hallucinations in LLMs?"
    assert result.rewrite_used is False
    assert result.method == "none"
    assert "standalone" in result.reason

