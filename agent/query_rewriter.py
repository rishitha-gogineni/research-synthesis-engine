"""Context-aware query rewriting for multi-turn research questions."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Literal

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator


DEFAULT_REWRITE_MODEL = "gpt-4o-mini"
MAX_HISTORY_TURNS = 8
AMBIGUOUS_PATTERNS = re.compile(
    r"\b(it|its|they|their|them|that|this|those|these|one|ones|method|approach|paper|model)\b",
    re.IGNORECASE,
)


class ChatTurn(BaseModel):
    """One prior chat turn used by the query rewriter."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2500)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("chat history content must not be blank")
        return cleaned


class QueryRewriteResult(BaseModel):
    """Result of rewriting a contextual follow-up into a standalone query."""

    original_query: str
    standalone_query: str
    rewrite_used: bool = False
    method: Literal["none", "llm", "heuristic"] = "none"
    reason: str = "No rewrite needed."


RewriteGenerator = Callable[[str], str]


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def needs_context(question: str) -> bool:
    lowered = question.lower().strip()
    return bool(AMBIGUOUS_PATTERNS.search(lowered)) or len(lowered.split()) <= 5


def compact_history(chat_history: list[ChatTurn]) -> list[ChatTurn]:
    return chat_history[-MAX_HISTORY_TURNS:]


def history_to_text(chat_history: list[ChatTurn]) -> str:
    lines = []
    for turn in compact_history(chat_history):
        lines.append(f"{turn.role}: {turn.content}")
    return "\n".join(lines)


def build_rewrite_prompt(question: str, chat_history: list[ChatTurn]) -> str:
    return f"""
Rewrite the current research question into a standalone search query for an indexed AI research paper corpus.

Rules:
- Do not answer the question.
- Resolve pronouns and references using only the chat history.
- Preserve the user's intent.
- If the current question is already standalone, return it unchanged.
- Do not invent paper titles, metrics, or claims.
- Return only JSON with: standalone_query, rewrite_used, reason.

Chat history:
{history_to_text(chat_history)}

Current question:
{question}
""".strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("rewriter did not return JSON") from None
        return json.loads(cleaned[start : end + 1])


def call_openai_rewriter(prompt: str, *, model: str = DEFAULT_REWRITE_MODEL) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You rewrite contextual research questions into standalone retrieval queries."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty rewrite")
    return content


def _last_user_topic(chat_history: list[ChatTurn]) -> str:
    for turn in reversed(chat_history):
        if turn.role == "user":
            content = clean_text(turn.content)
            content = re.sub(r"^(explain|describe|summarize|tell me about|what is|what are)\s+", "", content, flags=re.IGNORECASE)
            return content.rstrip("?.!")
    for turn in reversed(chat_history):
        content = clean_text(turn.content)
        if content:
            return content[:180].rstrip("?.!")
    return "the previously discussed research topic"


def heuristic_rewrite(question: str, chat_history: list[ChatTurn]) -> QueryRewriteResult:
    original = clean_text(question)
    if not chat_history or not needs_context(original):
        return QueryRewriteResult(original_query=original, standalone_query=original, rewrite_used=False, method="none")

    topic = _last_user_topic(chat_history)
    lowered = original.lower()
    if "limitation" in lowered or "challenge" in lowered:
        standalone = f"What are the limitations and challenges of {topic} in AI research papers?"
    elif "compare" in lowered or "versus" in lowered or " vs " in lowered or "different" in lowered:
        standalone = f"{original} Context: compare against {topic} in AI research papers."
    elif "read" in lowered or "paper" in lowered or "first" in lowered:
        standalone = f"Which papers should I read about {topic} in the indexed AI research corpus?"
    else:
        standalone = f"For {topic}, {original}"

    return QueryRewriteResult(
        original_query=original,
        standalone_query=standalone,
        rewrite_used=standalone != original,
        method="heuristic",
        reason="Resolved a contextual follow-up using recent chat history.",
    )


def rewrite_query(
    question: str,
    chat_history: list[ChatTurn] | None = None,
    *,
    generator: RewriteGenerator | None = None,
    model: str = DEFAULT_REWRITE_MODEL,
) -> QueryRewriteResult:
    """Rewrite a contextual follow-up question, using an LLM first and heuristics as fallback."""

    original = clean_text(question)
    history = compact_history(chat_history or [])
    if not history:
        return QueryRewriteResult(original_query=original, standalone_query=original, rewrite_used=False, method="none")
    if not needs_context(original):
        return QueryRewriteResult(
            original_query=original,
            standalone_query=original,
            rewrite_used=False,
            method="none",
            reason="Question is already standalone; rewrite skipped.",
        )

    prompt = build_rewrite_prompt(original, history)
    try:
        raw = generator(prompt) if generator is not None else call_openai_rewriter(prompt, model=model)
        payload = parse_json_object(raw)
        standalone = clean_text(payload.get("standalone_query"))
        if not standalone:
            raise ValueError("standalone_query is missing")
        rewrite_used = bool(payload.get("rewrite_used", standalone != original)) and standalone != original
        return QueryRewriteResult(
            original_query=original,
            standalone_query=standalone,
            rewrite_used=rewrite_used,
            method="llm",
            reason=clean_text(payload.get("reason")) or "Rewritten using chat history.",
        )
    except Exception:
        return heuristic_rewrite(original, history)
