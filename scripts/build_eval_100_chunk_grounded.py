"""Build a compact, corpus-grounded 100-query RAG benchmark.

The existing 250-query fixture remains unchanged.  This builder selects a
stratified subset and replaces paper-level labels with exact chunk IDs for
full-text questions.  Metadata/reading-path questions intentionally retain
paper IDs because their expected evidence is the paper record or abstract.

Usage:
    python scripts/build_eval_100_chunk_grounded.py
    python scripts/build_eval_100_chunk_grounded.py --check
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "fixtures" / "eval_queries_250_audited.json"
CHUNKS = ROOT / "data" / "full_text_chunks_v2.json"
PAPERS = ROOT / "data" / "enriched_papers_final.json"
FULL_TEXT_PAPERS = ROOT / "data" / "full_text_papers_v2.json"
OUTPUT = ROOT / "tests" / "fixtures" / "eval_queries_100_chunk_grounded.json"
MANIFEST = ROOT / "tests" / "fixtures" / "eval_queries_100_chunk_grounded_manifest.json"

QUOTAS = {
    "metadata": 10,
    "abstract": 10,
    "numeric_results": 20,
    "methodology": 19,
    # Only six existing dataset/benchmark questions have complete chunk-level
    # coverage for every labeled paper.  Keep those six rather than silently
    # dropping papers from a multi-paper gold set.
    "dataset_benchmarks": 6,
    "comparison": 10,
    "limitations": 10,
    "section_specific": 5,
    "multi_turn": 5,
    "out_of_corpus": 5,
}

STOPWORDS = {
    "about", "after", "against", "also", "among", "and", "are", "been",
    "being", "between", "can", "compared", "does", "from", "have", "how",
    "into", "more", "most", "than", "that", "their", "these", "they", "this",
    "those", "what", "when", "where", "which", "with", "within", "would",
    "your", "paper", "papers", "study", "studies", "selected", "reported",
    "results", "method", "methods", "approach", "work", "use", "used",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def tokens(value: str) -> list[str]:
    return [token for token in norm(value).split() if len(token) >= 3 and token not in STOPWORDS]


def classify(query: dict[str, Any]) -> str:
    if query.get("category") == "out_of_corpus":
        return "out_of_corpus"
    if query.get("category") == "multi_turn" or query.get("evaluation_focus") == "contextual_rewrite":
        return "multi_turn"
    focus = query.get("evaluation_focus")
    if focus == "metadata_filter":
        return "metadata"
    if focus == "reading_path":
        return "abstract"
    if focus == "cross_topic_comparison":
        return "comparison"
    rationale = query.get("rationale") or ""
    lowered = query.get("query", "").lower()
    if rationale.startswith("[factual] numeric"):
        return "numeric_results"
    if rationale.startswith("[methodology]"):
        return "methodology"
    if rationale.startswith("[limitation]"):
        return "limitations"
    if rationale.startswith("[section:"):
        return "section_specific"
    if any(term in lowered for term in ("dataset", "benchmark", "imagenet", "coco", "pubmed", "humaneval", "ade20k")):
        return "dataset_benchmarks"
    return "methodology"


def section_bonus(category: str, hint: str | None) -> float:
    section = (hint or "").lower()
    if category == "numeric_results":
        return 4.0 if any(word in section for word in ("result", "evaluation", "experiment", "table")) else 0.0
    if category == "methodology":
        return 4.0 if any(word in section for word in ("method", "approach", "architecture", "model")) else 0.0
    if category == "limitations":
        return 4.0 if any(word in section for word in ("discussion", "conclusion", "limitation", "future")) else 0.0
    if category == "section_specific":
        return 2.0 if section not in ("", "front_matter", "abstract") else 0.0
    if category == "dataset_benchmarks":
        return 3.0 if any(word in section for word in ("data", "dataset", "experiment", "evaluation")) else 0.0
    return 0.0


def chunk_score(query: dict[str, Any], category: str, paper: dict[str, Any], chunk: dict[str, Any]) -> tuple[float, int]:
    text = norm(chunk.get("text", ""))
    title_terms = set(tokens(paper.get("title", "")))
    keyword_terms = set(tokens(" ".join(query.get("expected_keywords", []))))
    query_terms = set(tokens(query.get("query", "")))
    weighted_hits = sum(4 for term in keyword_terms if term in text)
    weighted_hits += sum(3 for term in title_terms if term in text)
    weighted_hits += sum(1 for term in query_terms if term in text)
    phrase_hits = sum(5 for phrase in query.get("expected_keywords", []) if norm(phrase) and norm(phrase) in text)
    score = weighted_hits + phrase_hits + section_bonus(category, chunk.get("section_hint"))
    anchors = len({term for term in keyword_terms | title_terms if term in text})
    return score, anchors


def choose_chunk_ids(query: dict[str, Any], category: str, by_paper: dict[str, list[dict[str, Any]]], papers: dict[str, dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    selected: list[str] = []
    audit: list[dict[str, Any]] = []
    # Preserve all papers for cross-paper questions, but use one conservative
    # evidence chunk per paper.  For single-paper questions this yields one
    # gold chunk, avoiding an inflated denominator from labeling whole papers.
    for paper_id in query.get("expected_relevant_ids", []):
        chunks = by_paper.get(paper_id, [])
        paper = papers.get(paper_id, {"title": ""})
        ranked = sorted(
            ((chunk_score(query, category, paper, chunk), chunk) for chunk in chunks),
            key=lambda item: (-item[0][0], -item[0][1], item[1].get("chunk_index", 0)),
        )
        if not ranked:
            audit.append({"paper_id": paper_id, "status": "no_chunks"})
            continue
        (score, anchors), chunk = ranked[0]
        selected.append(chunk["chunk_id"])
        audit.append({
            "paper_id": paper_id,
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk.get("chunk_index"),
            "section_hint": chunk.get("section_hint"),
            "score": score,
            "anchor_count": anchors,
        })
    return selected, audit


def add_paper_anchors(record: dict[str, Any], category: str, papers: dict[str, dict[str, Any]]) -> None:
    """Make full-text questions self-contained without changing their intent.

    The legacy questions often say only "What accuracy does it achieve?" while
    their gold label points at one paper.  A chunk-grounded benchmark must name
    that paper; otherwise a zero hit can mean either retrieval failure or an
    ambiguous query.  We keep the original wording and append the source title
    as an explicit anchor.  Multi-turn queries are left untouched so rewriting
    remains testable.
    """
    if category == "multi_turn":
        return
    titles = [papers[paper_id]["title"] for paper_id in record.get("source_paper_ids", []) if paper_id in papers]
    if not titles:
        return
    base = record["query"].rstrip()
    if base.endswith("?"):
        base = base[:-1]
    if len(titles) == 1:
        record["query"] = f'{base} (according to "{titles[0]}")?'
    else:
        joined = "; ".join(f'"{title}"' for title in titles)
        record["query"] = f"{base} (considering {joined})?"


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = load(SOURCE)
    chunks = load(CHUNKS)
    papers = {paper["paper_id"]: paper for paper in load(PAPERS)}
    full_text_papers = {paper["paper_id"]: paper for paper in load(FULL_TEXT_PAPERS)}
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_paper[chunk["paper_id"]].append(chunk)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, query in enumerate(source):
        enriched = dict(query)
        enriched["_source_index"] = index
        buckets[classify(query)].append(enriched)

    output: list[dict[str, Any]] = []
    selection_audit: list[dict[str, Any]] = []
    for category, quota in QUOTAS.items():
        candidates = buckets[category]
        # Prefer specific, single-paper questions first; retain original order
        # as a deterministic tie-breaker so the fixture is reproducible.
        def candidate_key(item: dict[str, Any]) -> tuple[int, int, int]:
            ids = item.get("expected_relevant_ids", [])
            has_chunks = all(paper_id in by_paper for paper_id in ids) if ids else category == "out_of_corpus"
            specific = int(item.get("rationale", "").startswith(("[factual]", "[methodology]", "[limitation]", "[section:")))
            return (-int(has_chunks), -specific, int(item["_source_index"]))

        chosen = 0
        for query in sorted(candidates, key=candidate_key):
            if chosen >= quota:
                break
            record = {key: value for key, value in query.items() if not key.startswith("_")}
            record["benchmark_category"] = category
            record["source_paper_ids"] = list(record.get("expected_relevant_ids", []))
            record["source_pdf_filenames"] = [
                full_text_papers[paper_id].get("pdf_filename")
                for paper_id in record.get("source_paper_ids", [])
                if paper_id in full_text_papers and full_text_papers[paper_id].get("pdf_filename")
            ]

            if category in {"numeric_results", "methodology", "dataset_benchmarks", "limitations", "section_specific", "comparison", "multi_turn"}:
                chunk_ids, audit = choose_chunk_ids(record, category, by_paper, papers)
                if not chunk_ids:
                    continue
                if len(chunk_ids) < len(record.get("expected_relevant_ids", [])):
                    continue
                record["expected_relevant_ids"] = chunk_ids
                record["gold_chunk_ids"] = chunk_ids
                record["expected_route"] = "chunk_level"
                # The production pipeline may return paper context alongside
                # the exact chunk evidence. Accept hybrid_both here so route
                # accuracy measures an actual routing error rather than
                # penalising the intended paper+passage retrieval path.
                record["acceptable_routes"] = ["chunk_level", "hybrid_both"]
                record["evaluation_focus"] = "full_text_evidence"
                add_paper_anchors(record, category, papers)
                # The label is now chunk-level, so the old paper-level rationale
                # remains only as provenance and is not used as ground truth.
                selection_audit.append({
                    "source_index": query["_source_index"],
                    "query": record["query"],
                    "benchmark_category": category,
                    "chunks": audit,
                })
            else:
                # Metadata/abstract and out-of-corpus labels intentionally remain
                # paper-level because those tasks are not chunk retrieval tasks.
                record["gold_chunk_ids"] = []
                if category == "out_of_corpus":
                    # The confidence gate intentionally probes both levels
                    # before refusing unsupported questions.
                    record["acceptable_routes"] = list(dict.fromkeys([
                        *record.get("acceptable_routes", []),
                        "hybrid_both",
                    ]))
                selection_audit.append({
                    "source_index": query["_source_index"],
                    "query": record["query"],
                    "benchmark_category": category,
                    "chunks": [],
                })
            output.append(record)
            chosen += 1
        if chosen != quota:
            raise RuntimeError(f"could only select {chosen}/{quota} queries for {category}")

    if len(output) != 100:
        raise RuntimeError(f"expected 100 queries, selected {len(output)}")
    # Stable ordering by benchmark category then source order makes diffs easy
    # to review while preserving deterministic selection.
    output.sort(key=lambda item: (list(QUOTAS).index(item["benchmark_category"]), item["query"]))
    for index, record in enumerate(output, start=1):
        record["benchmark_id"] = f"rse100-{index:03d}"

    manifest = {
        "source_fixture": str(SOURCE.relative_to(ROOT)),
        "output_fixture": str(OUTPUT.relative_to(ROOT)),
        "chunk_source": str(CHUNKS.relative_to(ROOT)),
        "paper_source": str(PAPERS.relative_to(ROOT)),
        "pdf_text_source": str(FULL_TEXT_PAPERS.relative_to(ROOT)),
        "query_count": len(output),
        "category_counts": dict(Counter(item["benchmark_category"] for item in output)),
        "chunk_labeled_queries": sum(bool(item.get("gold_chunk_ids")) for item in output),
        "paper_labeled_queries": sum(bool(item.get("source_paper_ids")) and not item.get("gold_chunk_ids") for item in output),
        "out_of_corpus_queries": sum(item["benchmark_category"] == "out_of_corpus" for item in output),
        "selection_audit": selection_audit,
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Build in memory and validate without writing files")
    args = parser.parse_args()
    output, manifest = build()
    if not args.check:
        OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "selection_audit"}, indent=2))


if __name__ == "__main__":
    main()
