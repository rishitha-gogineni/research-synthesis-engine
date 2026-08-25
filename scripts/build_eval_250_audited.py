"""Build a corpus-aligned, auditable successor to the legacy 250-query fixture.

The legacy fixture is intentionally left unchanged.  This builder repairs only
issues that can be checked deterministically against the local metadata and
full-text corpus:

* canonicalize duplicate OpenAlex records by normalized title;
* restore one paper label for paper-specific generated questions;
* replace known topic-mismatched label sets with corpus-backed labels;
* remove labels that cannot be reached by a chunk-level collection;
* route metadata-only papers to paper-level retrieval;
* repair a small set of mechanically truncated questions; and
* emit a query-by-query audit manifest explaining every change.

Usage:
    python scripts/build_eval_250_audited.py
    python scripts/build_eval_250_audited.py --check
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIXTURE = ROOT / "tests" / "fixtures" / "eval_queries_250.json"
OUTPUT_FIXTURE = ROOT / "tests" / "fixtures" / "eval_queries_250_audited.json"
AUDIT_MANIFEST = ROOT / "tests" / "fixtures" / "eval_queries_250_audit_manifest.json"
ALIAS_OUTPUT = ROOT / "data" / "paper_id_aliases.json"
PAPERS_PATH = ROOT / "data" / "enriched_papers_final.json"
FULL_TEXT_PATH = ROOT / "data" / "full_text_papers_v2.json"
CHUNKS_PATH = ROOT / "data" / "full_text_chunks_v2.json"

# OpenAlex currently carries the canonical Transformer paper with an incorrect
# local year.  Keep the source corpus immutable in this benchmark-only repair,
# but do not allow that bad value to make it relevant to post-2024 filters.
YEAR_OVERRIDES = {
    "https://openalex.org/W2626778328": 2017,
}

AMBIGUOUS_PRIMARY_OVERRIDES = {
    "What are the benchmark results for assessment of retrieval, augmentation, and generation steps in rag?":
        "https://openalex.org/W4411203672",
    "Describe how Benchmarking Large Language Models in is implemented.":
        "https://openalex.org/W4386556635",
    "What are the quantitative improvements from Retrieval augmented generation for large?":
        "https://openalex.org/W4411203672",
}

EXPLICIT_LABEL_OVERRIDES = {
    "What papers evaluate models on commonsense reasoning datasets?": [
        "https://openalex.org/W4389524317",
        "https://openalex.org/W4404781224",
    ],
    "Which papers in the corpus use discharge summaries or clinical notes?": [
        "https://openalex.org/W4391640544",
        "https://openalex.org/W4399738410",
    ],
    "What is zero-shot learning in the context of LLMs?": [
        "https://openalex.org/W3194309076",
        "https://openalex.org/W4394579747",
        "https://openalex.org/W4389524317",
    ],
    "How do LLMs perform on theory of mind tasks?": [
        "https://openalex.org/W4319452268",
        "https://openalex.org/W4389523767",
    ],
}

EXPECTED_TOPIC_OVERRIDES = {
    "Which papers in the corpus use discharge summaries or clinical notes?": [
        "Fine-tuning (LoRA / PEFT)",
        "Retrieval-Augmented Generation (RAG)",
    ],
    "How do LLMs perform on theory of mind tasks?": [
        "LLM Evaluation & Hallucination Detection",
        "AI Agents & Tool Use",
    ],
    "Show me papers on code generation or evaluation.": [
        "LLM Evaluation & Hallucination Detection",
        "Retrieval-Augmented Generation (RAG)",
    ],
}

MALFORMED_QUERY_REPLACEMENTS = {
    "What are the evaluation results for Retrieval-Augmented-related LLM tasks?":
        "What quantitative results does the selected retrieval-augmented language-model study report?",
    "What are the evaluation results for Evaluating-related LLM tasks?":
        "What quantitative results does the selected LLM evaluation study report?",
    "Describe how a unified categorization criter is implemented.":
        "How is the unified categorization criterion implemented?",
    "What are the evaluation results for Interactive-related LLM tasks?":
        "What evaluation results does the selected interactive-LLM study report?",
    "What are the limitations of of chatsim for editable photo-r?":
        "What limitations does ChatSim report for editable photo-realistic scene simulation?",
    "What problem does a unified categorization criter address?":
        "What problem does the unified categorization criterion address?",
}

EXPECTED_KEYWORD_OVERRIDES = {
    "What are the evaluation results for Retrieval-Augmented-related LLM tasks?": [
        "code",
        "summarization",
        "retrieval",
    ],
    "What are the evaluation results for Evaluating-related LLM tasks?": [
        "radiation",
        "oncology",
        "evaluation",
    ],
    "Describe how a unified categorization criter is implemented.": [
        "parameter-efficient",
        "fine-tuning",
        "categorization",
    ],
    "What are the evaluation results for Interactive-related LLM tasks?": [
        "interactive",
        "networking",
        "retrieval",
    ],
    "What are the limitations of of chatsim for editable photo-r?": [
        "chatsim",
        "editable",
        "simulation",
    ],
    "What problem does a unified categorization criter address?": [
        "parameter-efficient",
        "fine-tuning",
        "categorization",
    ],
}

SPECIFIC_RATIONALE_PATTERNS = (
    re.compile(r"^\[factual\] numeric result from (.+)$"),
    re.compile(r"^\[methodology\] deep-dive into (.+)$"),
    re.compile(r"^\[limitation\] from (.+)$"),
    re.compile(r"^\[section:[^]]+\] from (.+)$"),
    re.compile(r"^\[abstract\] main contribution of (.+)$"),
)


def normalize_text(value: str) -> str:
    value = value.replace("'", "").replace("'", "")
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def paper_search_text(paper: dict[str, Any]) -> str:
    fields = (
        "title",
        "abstract",
        "main_contribution",
        "methodology",
        "dataset_used",
        "key_result",
        "limitations",
    )
    return " ".join(str(paper.get(field, "")).lower() for field in fields)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_aliases(
    papers: list[dict[str, Any]],
    *,
    full_text_ids: set[str],
    chunk_paper_ids: set[str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in papers:
        by_title[normalize_text(paper["title"])].append(paper)

    aliases: dict[str, str] = {}
    groups: list[dict[str, Any]] = []
    for title_key, records in sorted(by_title.items()):
        if len(records) <= 1:
            continue

        def preference(paper: dict[str, Any]) -> tuple[int, int, int, str]:
            paper_id = paper["paper_id"]
            return (
                int(paper_id in chunk_paper_ids),
                int(paper_id in full_text_ids),
                int(paper.get("citation_count") or 0),
                paper_id,
            )

        canonical = max(records, key=preference)
        canonical_id = canonical["paper_id"]
        alias_ids = sorted(paper["paper_id"] for paper in records if paper["paper_id"] != canonical_id)
        for alias_id in alias_ids:
            aliases[alias_id] = canonical_id
        groups.append(
            {
                "normalized_title": title_key,
                "title": canonical["title"],
                "canonical_id": canonical_id,
                "alias_ids": alias_ids,
            }
        )
    return aliases, groups


def canonicalize_ids(values: list[str], aliases: dict[str, str]) -> list[str]:
    return list(dict.fromkeys(aliases.get(value, value) for value in values))


def rationale_target(query: dict[str, Any]) -> str | None:
    rationale = query.get("rationale") or ""
    for pattern in SPECIFIC_RATIONALE_PATTERNS:
        if match := pattern.match(rationale):
            return match.group(1)
    return None


def specific_primary_id(
    query: dict[str, Any],
    paper_by_id: dict[str, dict[str, Any]],
    aliases: dict[str, str],
) -> str | None:
    if override := AMBIGUOUS_PRIMARY_OVERRIDES.get(query["query"]):
        return aliases.get(override, override)
    target = rationale_target(query)
    if target is None:
        return None
    target_key = normalize_text(target)
    matches = [
        aliases.get(paper_id, paper_id)
        for paper_id in query.get("expected_relevant_ids", [])
        if normalize_text(paper_by_id[paper_id]["title"]).startswith(target_key)
    ]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError(f"could not resolve one primary label for {query['query']!r}: {matches}")
    return matches[0]


def corrected_year(paper: dict[str, Any]) -> int | None:
    return YEAR_OVERRIDES.get(paper["paper_id"], paper.get("year"))


def ranked(papers: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[str]:
    matches = [paper for paper in papers if predicate(paper)]
    matches.sort(key=lambda paper: (paper.get("citation_count") or 0, paper["paper_id"]), reverse=True)
    return [paper["paper_id"] for paper in matches[:5]]


def metadata_labels(query: dict[str, Any], papers: list[dict[str, Any]]) -> list[str]:
    text = query["query"].lower()
    expected_topics = set(query.get("expected_topics", []))

    def topic_match(paper: dict[str, Any]) -> bool:
        return not expected_topics or paper.get("topic") in expected_topics

    if match := re.search(r"published in (\d{4}) or later", text):
        year = int(match.group(1))
        return ranked(papers, lambda paper: topic_match(paper) and (corrected_year(paper) or 0) >= year)
    if match := re.search(r"more than (\d+) citations", text):
        citations = int(match.group(1))
        return ranked(papers, lambda paper: topic_match(paper) and (paper.get("citation_count") or 0) > citations)
    if "highly cited rag papers from 2023 or later" in text:
        return ranked(
            papers,
            lambda paper: topic_match(paper) and (corrected_year(paper) or 0) >= 2023,
        )
    if match := re.search(r"over (\d+) citations", text):
        citations = int(match.group(1))
        return ranked(papers, lambda paper: topic_match(paper) and (paper.get("citation_count") or 0) > citations)
    if "most cited papers on hallucination detection" in text:
        return ranked(papers, lambda paper: topic_match(paper) and "hallucination" in paper_search_text(paper))
    if "recent agent papers from 2024" in text:
        return ranked(papers, lambda paper: topic_match(paper) and corrected_year(paper) == 2024)
    if "fine-tuning papers were published before 2023" in text:
        return ranked(papers, lambda paper: topic_match(paper) and (corrected_year(paper) or 9999) < 2023)
    if "all survey papers" in text:
        return ranked(papers, lambda paper: "survey" in paper["title"].lower())
    if "papers about lora specifically" in text:
        return ranked(papers, lambda paper: topic_match(paper) and "lora" in paper["title"].lower())
    if "papers discuss medical applications" in text:
        terms = ("medical", "medicine", "clinical", "healthcare", "biomed")
        return ranked(papers, lambda paper: any(term in paper_search_text(paper) for term in terms))
    if "papers on code generation or evaluation" in text:
        return ranked(
            papers,
            lambda paper: topic_match(paper)
            and "code" in paper_search_text(paper)
            and any(term in paper_search_text(paper) for term in ("generation", "evaluation", "repair", "summarization")),
        )
    if "papers about multi-agent systems" in text:
        return ranked(papers, lambda paper: topic_match(paper) and "multi-agent" in paper_search_text(paper))
    raise ValueError(f"unhandled metadata query: {query['query']}")


def record_change(
    changes: list[dict[str, Any]],
    *,
    query_before: dict[str, Any],
    query_after: dict[str, Any],
    reasons: list[str],
) -> None:
    if not reasons:
        return
    changes.append(
        {
            "query_before": query_before["query"],
            "query_after": query_after["query"],
            "route_before": query_before["expected_route"],
            "route_after": query_after["expected_route"],
            "labels_before": query_before.get("expected_relevant_ids", []),
            "labels_after": query_after.get("expected_relevant_ids", []),
            "reasons": reasons,
        }
    )


def build() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_queries: list[dict[str, Any]] = load_json(SOURCE_FIXTURE)
    papers: list[dict[str, Any]] = load_json(PAPERS_PATH)
    full_text_ids = {paper["paper_id"] for paper in load_json(FULL_TEXT_PATH)}
    chunk_paper_ids = {chunk["paper_id"] for chunk in load_json(CHUNKS_PATH)}
    paper_by_id = {paper["paper_id"]: paper for paper in papers}
    aliases, duplicate_groups = canonical_aliases(
        papers,
        full_text_ids=full_text_ids,
        chunk_paper_ids=chunk_paper_ids,
    )
    canonical_chunk_ids = {aliases.get(paper_id, paper_id) for paper_id in chunk_paper_ids}

    audited: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for source in source_queries:
        query = copy.deepcopy(source)
        reasons: list[str] = []

        if replacement := MALFORMED_QUERY_REPLACEMENTS.get(query["query"]):
            query["query"] = replacement
            reasons.append("repaired mechanically malformed query text")

        if keywords := EXPECTED_KEYWORD_OVERRIDES.get(source["query"]):
            query["expected_keywords"] = keywords
            reasons.append("repaired malformed expected keywords")

        if topics := EXPECTED_TOPIC_OVERRIDES.get(source["query"]):
            query["expected_topics"] = topics
            reasons.append("expanded expected topics for a genuinely cross-topic query")

        if query["evaluation_focus"] == "metadata_filter":
            labels = metadata_labels(query, papers)
            reasons.append("recomputed metadata labels from query filters and corrected local metadata")
        elif source["query"] in EXPLICIT_LABEL_OVERRIDES:
            labels = EXPLICIT_LABEL_OVERRIDES[source["query"]]
            reasons.append("replaced topic-mismatched labels with corpus-backed papers")
        elif primary_id := specific_primary_id(source, paper_by_id, aliases):
            labels = [primary_id]
            if canonicalize_ids(source.get("expected_relevant_ids", []), aliases) != labels:
                reasons.append("restored one target paper for a paper-specific question")
        else:
            labels = canonicalize_ids(source.get("expected_relevant_ids", []), aliases)

        labels = canonicalize_ids(labels, aliases)
        expected_topics = set(query.get("expected_topics", []))
        if expected_topics and query["evaluation_focus"] != "metadata_filter":
            topic_aligned = [paper_id for paper_id in labels if paper_by_id[paper_id]["topic"] in expected_topics]
            if topic_aligned != labels:
                labels = topic_aligned
                reasons.append("removed labels outside the query's expected topics")

        if query["expected_route"] == "chunk_level" and labels:
            reachable = [paper_id for paper_id in labels if paper_id in canonical_chunk_ids]
            if reachable:
                if reachable != labels:
                    labels = reachable
                    reasons.append("removed labels unreachable by the canonical chunk corpus")
            else:
                query["expected_route"] = "paper_level"
                reasons.append("reclassified metadata-only paper from chunk-level to paper-level retrieval")

        if source.get("expected_relevant_ids") and not labels:
            raise ValueError(f"repair removed every label from {source['query']!r}")

        query["expected_relevant_ids"] = labels
        # Route accuracy should measure the intended behavior, not accept nearly
        # every path after results are observed.
        query["acceptable_routes"] = [query["expected_route"]]
        if source.get("acceptable_routes") != query["acceptable_routes"]:
            reasons.append("restricted route labels to the intended route")

        audited.append(query)
        record_change(changes, query_before=source, query_after=query, reasons=reasons)

    alias_payload = {
        "description": "Canonical OpenAlex IDs for duplicate normalized paper titles in the local RSE corpus.",
        "source": str(PAPERS_PATH.relative_to(ROOT)),
        "aliases": dict(sorted(aliases.items())),
        "duplicate_groups": duplicate_groups,
    }
    manifest = {
        "source_fixture": str(SOURCE_FIXTURE.relative_to(ROOT)),
        "audited_fixture": str(OUTPUT_FIXTURE.relative_to(ROOT)),
        "query_count": len(audited),
        "changed_query_count": len(changes),
        "changes": changes,
    }
    return audited, alias_payload, manifest


def validate(
    queries: list[dict[str, Any]],
    alias_payload: dict[str, Any],
) -> dict[str, Any]:
    papers: list[dict[str, Any]] = load_json(PAPERS_PATH)
    paper_by_id = {paper["paper_id"]: paper for paper in papers}
    aliases: dict[str, str] = alias_payload["aliases"]
    chunk_ids = {aliases.get(chunk["paper_id"], chunk["paper_id"]) for chunk in load_json(CHUNKS_PATH)}

    errors: list[str] = []
    if len(queries) != 250:
        errors.append(f"expected 250 queries, got {len(queries)}")
    if len({query["query"] for query in queries}) != len(queries):
        errors.append("query text is not unique")

    for query in queries:
        labels = query.get("expected_relevant_ids", [])
        missing = [paper_id for paper_id in labels if paper_id not in paper_by_id]
        if missing:
            errors.append(f"{query['query']}: missing paper IDs {missing}")
        if any(paper_id in aliases for paper_id in labels):
            errors.append(f"{query['query']}: label uses a non-canonical alias")
        expected_topics = set(query.get("expected_topics", []))
        mismatched = [
            paper_id for paper_id in labels
            if expected_topics and paper_by_id[paper_id]["topic"] not in expected_topics
        ]
        if mismatched:
            errors.append(f"{query['query']}: topic-mismatched labels {mismatched}")
        if query["expected_route"] == "chunk_level" and any(paper_id not in chunk_ids for paper_id in labels):
            errors.append(f"{query['query']}: unreachable chunk-level labels")
        if query.get("acceptable_routes") != [query["expected_route"]]:
            errors.append(f"{query['query']}: route alternatives are not strict")

    if errors:
        raise ValueError("\n".join(errors))

    label_assignments = sum(len(query.get("expected_relevant_ids", [])) for query in queries)
    unique_labels = {paper_id for query in queries for paper_id in query.get("expected_relevant_ids", [])}
    return {
        "queries": len(queries),
        "labeled_queries": sum(bool(query.get("expected_relevant_ids")) for query in queries),
        "label_assignments": label_assignments,
        "unique_labeled_papers": len(unique_labels),
        "chunk_level_queries": sum(query["expected_route"] == "chunk_level" for query in queries),
        "unreachable_chunk_labels": 0,
        "topic_mismatched_queries": 0,
        "duplicate_alias_groups": len(alias_payload["duplicate_groups"]),
    }


def serialize(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate that committed outputs match a fresh build.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries, aliases, manifest = build()
    stats = validate(queries, aliases)
    manifest["validation"] = stats

    outputs = {
        OUTPUT_FIXTURE: serialize(queries),
        ALIAS_OUTPUT: serialize(aliases),
        AUDIT_MANIFEST: serialize(manifest),
    }
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, content in outputs.items() if not path.exists() or path.read_text() != content]
        if stale:
            raise SystemExit("Audited evaluation outputs are stale: " + ", ".join(stale))
        print(json.dumps(stats, indent=2))
        return

    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
