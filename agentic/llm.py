"""Bounded OpenAI tool-calling and citation-aware synthesis."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Callable
from openai import OpenAI
from agentic.prompts import build_tool_system_prompt, build_tool_user_prompt
from agentic.tools import execute_tool, tool_definitions

class ToolCallingError(RuntimeError):
    """Raised when the model cannot complete a safe tool-calling request."""

@dataclass
class LLMResult:
    answer: str
    citations: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

def _value(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default

def _message_dict(message: Any) -> dict[str, Any]:
    payload = {"role": "assistant", "content": _value(message, "content") or ""}
    calls = []
    for call in _value(message, "tool_calls", []) or []:
        function = _value(call, "function")
        calls.append({"id": _value(call, "id", ""), "type": "function", "function": {"name": _value(function, "name", ""), "arguments": _value(function, "arguments", "{}")}})
    if calls: payload["tool_calls"] = calls
    return payload

def _citations(answer: str, evidence: list[dict[str, Any]]) -> list[str]:
    valid = {f"source_{index}" for index, _ in enumerate(evidence, start=1)}
    return [ref for ref in re.findall(r"\[([^\]]+)\]", answer) if ref in valid]

def _client_from_env() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise ToolCallingError("OPENAI_API_KEY is missing")
    return OpenAI(api_key=api_key)

def run_grounded_answer(
    query: str,
    evidence: list[dict[str, Any]],
    *,
    client: Any | None = None,
    model: str = "gpt-4o-mini",
    max_tool_calls: int = 3,
    executor: Callable[..., dict[str, Any]] = execute_tool,
    allowed_tools: tuple[str, ...] | None = None,
) -> LLMResult:
    if not query.strip(): raise ValueError("query must not be empty")
    if max_tool_calls < 0: raise ValueError("max_tool_calls must be non-negative")
    client = client or _client_from_env()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_tool_system_prompt()},
        {"role": "user", "content": build_tool_user_prompt(query, evidence)},
    ]
    traces: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_calls = 0
    seen_tool_calls: set[tuple[str, str]] = set()
    tool_specs = tool_definitions()
    if allowed_tools is not None:
        allowed = set(allowed_tools)
        tool_specs = [
            spec for spec in tool_specs
            if spec.get("function", {}).get("name") in allowed
        ]
    prompt_tokens = completion_tokens = 0
    started = time.perf_counter()
    while True:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.1}
        if total_calls < max_tool_calls and tool_specs:
            kwargs.update({"tools": tool_specs, "tool_choice": "auto"})
        else:
            messages.append({"role": "system", "content": "The tool-call budget is exhausted. Answer only from the evidence already collected, or refuse if it is insufficient."})
        response = client.chat.completions.create(**kwargs)
        usage = _value(response, "usage")
        prompt_tokens += int(_value(usage, "prompt_tokens", 0) or 0)
        completion_tokens += int(_value(usage, "completion_tokens", 0) or 0)
        message = _value(_value(response, "choices", [None])[0], "message")
        calls = _value(message, "tool_calls", []) or []
        if calls and total_calls >= max_tool_calls:
            warnings.append("Tool-call budget exhausted; model requested another tool.")
            return LLMResult("I could not verify the answer from the available evidence.", [], traces, warnings, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}, round((time.perf_counter() - started) * 1000, 3))
        if not calls:
            answer = (_value(message, "content", "") or "").strip()
            if not answer: raise ToolCallingError("OpenAI returned an empty answer")
            refs = _citations(answer, evidence)
            if evidence and not refs and "insufficient" not in answer.lower():
                warnings.append("Answer did not include a recognized evidence citation.")
            return LLMResult(answer, refs, traces, warnings, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}, round((time.perf_counter() - started) * 1000, 3))
        messages.append(_message_dict(message))
        for call in calls:
            name = _value(_value(call, "function"), "name", "")
            raw_args = _value(_value(call, "function"), "arguments", "{}") or "{}"
            trace = {"tool": name, "status": "completed"}
            call_key = (name, raw_args)
            if call_key in seen_tool_calls:
                trace["status"] = "duplicate_blocked"
                warnings.append(f"Duplicate tool request blocked: {name}.")
                result = {"error": "duplicate tool request blocked; use the existing result"}
                total_calls = max(total_calls, max_tool_calls)
            elif total_calls >= max_tool_calls:
                trace["status"] = "budget_exhausted"
                result = {"error": "tool-call budget exhausted"}
            else:
                seen_tool_calls.add(call_key)
                total_calls += 1
                try:
                    arguments = json.loads(raw_args)
                    result = executor(name, arguments)
                except Exception as exc:
                    trace["status"] = "failed"
                    result = {"error": str(exc)}
            traces.append(trace)
            messages.append({"role": "tool", "tool_call_id": _value(call, "id", ""), "content": json.dumps(result, default=str)})
