"""Guardrail agent — validates input queries before the pipeline runs.

Runs in parallel with the planning step. Checks for:
- Prompt injection attempts
- Off-topic / non-research queries
- Unsafe or harmful content
- Queries that are too vague to be actionable
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GuardrailResult:
    safe: bool
    reason: str
    category: str  # "safe", "prompt_injection", "off_topic", "unsafe", "too_vague"


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+a",
    r"pretend\s+you\s+are",
    r"act\s+as\s+(a|an|if)",
    r"forget\s+(all\s+)?(your|previous)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*you",
    r"<\s*/?system\s*>",
    r"\]\s*\[\s*system",
]

NON_RESEARCH_PATTERNS = [
    r"\b(weather|forecast)\s+(in|for|today)",
    r"\b(stock|share)\s+price\b",
    r"\b(restaurant|hotel|flight|ticket)\s+(near|in|to|from)\b",
    r"\b(recipe|cook|bake)\s+(for|a|the)\b",
    r"\b(buy|purchase|order|shop)\s+(a|the|online)\b",
    # General non-research intent patterns
    r"\bwho\s+won\b",
    r"\bwhat\s+is\s+the\s+(score|price|cost|weather)\b",
    r"\bhow\s+much\s+(does|is|are|do)\b",
    r"\bwhere\s+(can\s+i|do\s+i|should\s+i)\s+(buy|eat|go|find|stay)\b",
    r"\b(sports?\s+score|game\s+result|match\s+result)\b",
    r"\b(movie|film|tv|show)\s+(review|rating|showtimes?|schedule)\b",
]

UNSAFE_PATTERNS = [
    r"\b(hack|exploit|attack)\s+(a|the|this)\s+(system|server|network|website)\b",
    r"\b(make|build|create)\s+(a\s+)?(bomb|weapon|virus|malware)\b",
    r"\bhow\s+to\s+(harm|hurt|kill|injure)\b",
]


def check_guardrails(query: str) -> GuardrailResult:
    """Run all guardrail checks on the input query."""
    normalized = " ".join(query.lower().split())

    if not normalized or len(normalized) < 3:
        return GuardrailResult(
            safe=False,
            reason="Query is empty or too short to process.",
            category="too_vague",
        )

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            return GuardrailResult(
                safe=False,
                reason="Query contains patterns consistent with prompt injection.",
                category="prompt_injection",
            )

    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, normalized):
            return GuardrailResult(
                safe=False,
                reason="Query requests potentially harmful content.",
                category="unsafe",
            )

    for pattern in NON_RESEARCH_PATTERNS:
        if re.search(pattern, normalized):
            return GuardrailResult(
                safe=False,
                reason="Query is not related to AI/ML research.",
                category="off_topic",
            )

    if len(normalized.split()) < 3 and "?" not in query:
        return GuardrailResult(
            safe=False,
            reason="Query is too vague — provide more context.",
            category="too_vague",
        )

    return GuardrailResult(safe=True, reason="Query passed all checks.", category="safe")
