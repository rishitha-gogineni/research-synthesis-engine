"""Deterministic, route-aware candidate promotion.

The top-20 diagnostic on the v2 fixture shows relevant-ID hit rate rising from
0.691 at k=10 to 0.882 at k=20, and true recall from 0.384 to 0.551. In other
words a large share of the remaining failures are *ranking* failures, not
indexing failures: the right evidence is already retrieved, just below the
visible cutoff. This module is the layer that tries to pull it up.

Three properties are non-negotiable here, because each one corresponds to a
measured regression this project already paid for:

1. **Pool invariance.** Every signal is an absolute function of a candidate's
   own rank and its own payload. Nothing is min-max normalized against the
   batch. Widening the candidate pool therefore cannot rescale a candidate that
   was already in it -- that batch-relative coupling is exactly what made naive
   oversampling drop Recall@10 from 0.76 to 0.69 (see docs/DECISIONS.md).

2. **No dependency on rerank scores.** ``rerank_and_blend`` now preserves
   retrieval order when the cross-encoder cannot load, which is the deployed
   free-tier configuration. In that path ``blended_score`` and ``rerank_score``
   do not exist on candidates at all. Promotion reads only rank position,
   ``section_hint``, and metadata fields that are always present.

3. **Clustering for evidence queries, diversity only for comparisons.** The
   MMR ablation showed a blanket redundancy penalty hurt recall, because 19 of
   36 labeled queries want several chunks from the *same* paper. Parent-paper
   caps here are therefore opt-in per query intent, and they demote rather than
   drop, so no candidate is ever lost.

Promotion is bounded: the total bonus a candidate can earn is capped at
MAX_PROMOTION_BONUS, which is roughly the rank-prior gap between rank 10 and
rank 15. A well-matched candidate can climb a few positions; nothing jumps
from rank 20 to rank 1.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


# Rank prior. Depends only on a candidate's own rank, never on the pool.
RANK_DECAY = 10.0

# Bonus budget. Sized against the rank prior: prior(10) - prior(15) == 0.10.
SECTION_BONUS = 0.05
VOCAB_BONUS = 0.03
PAPER_FIELD_BONUS = 0.04
MAX_PROMOTION_BONUS = 0.12

# Parent-paper caps, applied by demotion rather than removal.
DIVERSITY_CHUNKS_PER_PAPER = 2
EVIDENCE_CHUNKS_PER_PAPER = 4


INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dataset", re.compile(r"\b(?:datasets?|corpora|corpus|training data|trained on|evaluated on)\b", re.IGNORECASE)),
    ("benchmark", re.compile(r"\b(?:benchmarks?|evaluation suite|test sets?|leaderboards?)\b", re.IGNORECASE)),
    ("metric", re.compile(r"\b(?:metrics?|measured?|measure|accuracy|precision|recall|f1|perplexity)\b", re.IGNORECASE)),
    ("limitation", re.compile(r"\b(?:limitations?|weakness(?:es)?|drawbacks?|shortcomings?|failure modes?)\b", re.IGNORECASE)),
    ("method", re.compile(r"\b(?:methods?|methodology|approach(?:es)?|architectures?|techniques?|how does .+ work)\b", re.IGNORECASE)),
    ("result", re.compile(r"\b(?:results?|ablations?|how much|improvements?|outperforms?|reduces?|speedups?)\b", re.IGNORECASE)),
)

# section_hint values present in data/full_text_chunks.json:
# results, experiments, methodology, limitations, unknown, introduction,
# related_work, conclusion.
INTENT_SECTIONS: dict[str, frozenset[str]] = {
    "dataset": frozenset({"experiments", "methodology"}),
    "benchmark": frozenset({"experiments", "results"}),
    "metric": frozenset({"results", "experiments"}),
    "limitation": frozenset({"limitations", "conclusion"}),
    "method": frozenset({"methodology"}),
    "result": frozenset({"results", "experiments"}),
}

INTENT_VOCAB: dict[str, frozenset[str]] = {
    "dataset": frozenset({"dataset", "datasets", "corpus", "corpora", "training set", "test set", "annotated"}),
    "benchmark": frozenset({"benchmark", "benchmarks", "evaluation suite", "leaderboard", "test set"}),
    "metric": frozenset({"accuracy", "precision", "recall", "f1", "bleu", "rouge", "perplexity", "score", "metric"}),
    "limitation": frozenset({"limitation", "limitations", "however", "fails", "failure", "drawback", "future work"}),
    "method": frozenset({"architecture", "algorithm", "objective", "training procedure", "we propose", "our method"}),
    "result": frozenset({"ablation", "outperforms", "improves", "reduction", "baseline", "table", "compared to"}),
}

# Paper-level metadata fields that answer each intent directly.
INTENT_PAPER_FIELDS: dict[str, tuple[str, ...]] = {
    "dataset": ("dataset_used",),
    "benchmark": ("dataset_used", "key_result"),
    "metric": ("key_result",),
    "limitation": ("limitations",),
    "method": ("methodology",),
    "result": ("key_result",),
}

# Queries that genuinely want breadth across papers rather than depth in one.
DIVERSITY_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdifference between\b", re.IGNORECASE),
    re.compile(r"\bcompare(?:d|s)?\b", re.IGNORECASE),
    re.compile(r"\bcomparison\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\bsurvey\b", re.IGNORECASE),
    re.compile(r"\bwhat are the (?:main|key|common)\b", re.IGNORECASE),
    re.compile(r"\bapproaches to\b", re.IGNORECASE),
    re.compile(r"\bdiffer(?:s|ent)?\s+from\b", re.IGNORECASE),
)


def detect_query_intents(query: str) -> tuple[str, ...]:
    """Return every intent label whose pattern fires on the query.

    Intents are not mutually exclusive: "what datasets and metrics are used"
    legitimately carries both. Order follows INTENT_PATTERNS so the result is
    deterministic for a given query.
    """

    return tuple(name for name, pattern in INTENT_PATTERNS if pattern.search(query))


def is_diversity_query(query: str) -> bool:
    """True when the query wants breadth across papers rather than depth in one."""

    return any(pattern.search(query) for pattern in DIVERSITY_QUERY_PATTERNS)


def rank_prior(rank: int) -> float:
    """Score a candidate from its 1-based retrieval rank alone.

    ``1 / (1 + rank / RANK_DECAY)`` is a smooth, strictly decreasing function of
    a single candidate's own rank. It never reads any other candidate, so
    enlarging the pool leaves every existing candidate's prior untouched. With
    RANK_DECAY = 10 the priors are rank 1 -> 0.909, rank 10 -> 0.500,
    rank 15 -> 0.400, rank 20 -> 0.333.
    """

    if rank < 1:
        raise ValueError("rank must be 1-based and positive")
    return 1.0 / (1.0 + rank / RANK_DECAY)


def candidate_haystack(candidate: dict[str, Any]) -> str:
    fields = (
        candidate.get("text"),
        candidate.get("title"),
        candidate.get("abstract"),
        candidate.get("main_contribution"),
        candidate.get("methodology"),
        candidate.get("dataset_used"),
        candidate.get("key_result"),
        candidate.get("limitations"),
    )
    return " ".join(str(field).lower() for field in fields if field)


def section_bonus(candidate: dict[str, Any], intents: tuple[str, ...]) -> float:
    """Reward chunks whose section_hint matches what the query is asking for.

    section_hint is written at ingestion time and stored on every chunk payload,
    but nothing downstream has ever scored it. A "what are the limitations"
    query and a chunk tagged ``limitations`` is about as clean a deterministic
    relevance signal as this corpus offers.
    """

    hint = str(candidate.get("section_hint") or "").strip().lower()
    if not hint or hint == "unknown":
        return 0.0
    for intent in intents:
        if hint in INTENT_SECTIONS.get(intent, frozenset()):
            return SECTION_BONUS
    return 0.0


def vocab_bonus(candidate: dict[str, Any], intents: tuple[str, ...]) -> float:
    """Reward candidates whose own text carries the vocabulary of the intent."""

    if not intents:
        return 0.0
    haystack = candidate_haystack(candidate)
    if not haystack:
        return 0.0
    for intent in intents:
        if any(term in haystack for term in INTENT_VOCAB.get(intent, frozenset())):
            return VOCAB_BONUS
    return 0.0


def paper_field_bonus(candidate: dict[str, Any], intents: tuple[str, ...]) -> float:
    """Reward papers whose enriched metadata directly answers the intent.

    Paper records carry structured ``dataset_used`` / ``methodology`` /
    ``key_result`` / ``limitations`` fields from enrichment. A populated field
    matching the query intent is a stronger signal than the same words merely
    appearing somewhere in an abstract.
    """

    for intent in intents:
        for field in INTENT_PAPER_FIELDS.get(intent, ()):
            value = candidate.get(field)
            if value and str(value).strip():
                return PAPER_FIELD_BONUS
    return 0.0


def promotion_bonus(
    candidate: dict[str, Any],
    intents: tuple[str, ...],
    *,
    level: str,
) -> tuple[float, list[str]]:
    """Return the clamped bonus for one candidate plus the signals that fired."""

    signals: list[str] = []
    total = 0.0

    if level == "chunk":
        earned = section_bonus(candidate, intents)
        if earned:
            total += earned
            signals.append(f"section_hint:{candidate.get('section_hint')}")
    else:
        earned = paper_field_bonus(candidate, intents)
        if earned:
            total += earned
            signals.append("paper_field_match")

    earned = vocab_bonus(candidate, intents)
    if earned:
        total += earned
        signals.append("intent_vocab")

    return min(total, MAX_PROMOTION_BONUS), signals


def chunks_per_paper_cap(query: str, intents: tuple[str, ...]) -> int | None:
    """How many chunks from one parent paper may occupy the visible window.

    ``None`` means no cap, which is the default. Capping is only applied when
    the query itself asks for breadth, or when it asks for specific evidence
    (where a loose cap still leaves room for the multi-chunk answers that MMR
    used to destroy).
    """

    if is_diversity_query(query):
        return DIVERSITY_CHUNKS_PER_PAPER
    if intents:
        return EVIDENCE_CHUNKS_PER_PAPER
    return None


def apply_parent_paper_cap(candidates: list[dict[str, Any]], cap: int | None) -> list[dict[str, Any]]:
    """Demote chunks beyond ``cap`` per parent paper to the tail of the list.

    Nothing is dropped. A paper that legitimately owns the answer keeps all its
    chunks, just with the surplus pushed below other papers' first hits. This is
    the difference between this and MMR: no chunk text is ever compared, so a
    genuinely relevant near-duplicate is never penalized for being similar.
    """

    if cap is None or cap <= 0:
        return list(candidates)

    seen: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for candidate in candidates:
        paper_id = str(candidate.get("paper_id") or "")
        if not paper_id:
            kept.append(candidate)
            continue
        seen[paper_id] += 1
        if seen[paper_id] <= cap:
            kept.append(candidate)
        else:
            overflow.append(candidate)
    return kept + overflow


def promote_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    level: str = "chunk",
) -> list[dict[str, Any]]:
    """Re-order a candidate pool with bounded, pool-invariant signals.

    ``candidates`` is assumed to arrive in retrieval order. Each candidate is
    scored as ``rank_prior(its own rank) + bounded intent bonus``, so the
    ordering of any two candidates depends only on those two candidates. The
    result is truncated to ``top_k``.

    Candidates are annotated with ``promotion_score`` and ``promotion_signals``
    for debuggability. These are additive fields; nothing existing is
    overwritten, and no scoring field is required to be present on input.
    """

    if level not in {"chunk", "paper"}:
        raise ValueError("level must be 'chunk' or 'paper'")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if not candidates:
        return []

    intents = detect_query_intents(query)

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        bonus, signals = promotion_bonus(candidate, intents, level=level)
        score = rank_prior(index + 1) + bonus
        candidate["promotion_score"] = score
        candidate["promotion_signals"] = signals
        scored.append((score, index, candidate))

    # Ties break on original retrieval rank, so promotion is a stable sort.
    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered = [candidate for _, _, candidate in scored]

    if level == "chunk":
        ordered = apply_parent_paper_cap(ordered, chunks_per_paper_cap(query, intents))

    return ordered[:top_k]
