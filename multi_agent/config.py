"""Effort scaling configuration for multi-agent research system."""

from dataclasses import dataclass


@dataclass
class EffortLevel:
    name: str
    max_subagents: int
    max_tool_calls_per_agent: int
    max_iterations: int


EFFORT_SIMPLE = EffortLevel(
    name="simple",
    max_subagents=1,
    max_tool_calls_per_agent=5,
    max_iterations=1,
)

EFFORT_MODERATE = EffortLevel(
    name="moderate",
    max_subagents=3,
    max_tool_calls_per_agent=10,
    max_iterations=2,
)

EFFORT_COMPLEX = EffortLevel(
    name="complex",
    max_subagents=5,
    max_tool_calls_per_agent=15,
    max_iterations=3,
)

SUBAGENT_TIMEOUT_SECONDS = 60
MAX_TOTAL_TOKENS_PER_SESSION = 100_000
DEFAULT_MODEL = "gpt-4o"
SUBAGENT_MODEL = "gpt-4o-mini"

COMPLEXITY_KEYWORDS_SIMPLE = [
    "what is", "define", "who", "when was", "name the",
]

COMPLEXITY_KEYWORDS_COMPLEX = [
    "compare", "contrast", "analyze", "comprehensive", "all",
    "relationship between", "how does", "survey", "overview of",
    "differences between", "trade-offs", "pros and cons",
]


def classify_effort(query: str) -> EffortLevel:
    """Classify query complexity to determine agent count and tool budget."""
    query_lower = query.lower()

    complex_signals = sum(
        1 for kw in COMPLEXITY_KEYWORDS_COMPLEX if kw in query_lower
    )
    simple_signals = sum(
        1 for kw in COMPLEXITY_KEYWORDS_SIMPLE if kw in query_lower
    )

    if complex_signals >= 2:
        return EFFORT_COMPLEX
    if complex_signals == 1 or len(query.split()) > 20:
        return EFFORT_MODERATE
    if simple_signals >= 1 and len(query.split()) <= 12:
        return EFFORT_SIMPLE
    return EFFORT_MODERATE
