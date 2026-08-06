"""Lightweight corpus indexes for fast query-pattern paths.

These indexes do not replace vector/BM25 retrieval. They handle questions where
structured lookup is a better fit: ranked metadata lists and specific-paper
lookup before falling back to the normal RAG pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

QuestionPatternName = Literal[
    "concept_explanation",
    "comparison",
    "paper_lookup",
    "ranked_list",
    "dataset_method",
    "reading_path",
    "follow_up",
    "out_of_corpus",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS_PATH = PROJECT_ROOT / "data" / "enriched_papers_final.json"
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "full_text_chunks.json"

TOPIC_ALIASES: dict[str, str] = {
    "rag": "Retrieval-Augmented Generation (RAG)",
    "retrieval augmented": "Retrieval-Augmented Generation (RAG)",
    "retrieval-augmented": "Retrieval-Augmented Generation (RAG)",
    "hallucination": "LLM Evaluation & Hallucination Detection",
    "evaluation": "LLM Evaluation & Hallucination Detection",
    "lora": "Fine-tuning (LoRA / PEFT)",
    "peft": "Fine-tuning (LoRA / PEFT)",
    "fine tuning": "Fine-tuning (LoRA / PEFT)",
    "fine-tuning": "Fine-tuning (LoRA / PEFT)",
    "agent": "AI Agents & Tool Use",
    "agents": "AI Agents & Tool Use",
    "tool use": "AI Agents & Tool Use",
    "transformer": "Transformers / Attention Mechanisms",
    "transformers": "Transformers / Attention Mechanisms",
    "attention": "Transformers / Attention Mechanisms",
}

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "based",
    "between",
    "can",
    "explain",
    "for",
    "from",
    "give",
    "highly",
    "into",
    "list",
    "most",
    "paper",
    "papers",
    "published",
    "read",
    "show",
    "survey",
    "that",
    "the",
    "this",
    "what",
    "which",
    "with",
}


@dataclass(frozen=True)
class PaperMatch:
    """Resolved paper lookup candidate."""

    paper_id: str
    title: str
    score: float
    reason: str


@dataclass
class CorpusIndex:
    """In-memory maps built from the local paper and chunk artifacts."""

    paper_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    title_to_paper_id: dict[str, str] = field(default_factory=dict)
    topic_to_papers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    chunks_by_paper_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_records(cls, papers: list[dict[str, Any]], chunks: list[dict[str, Any]] | None = None) -> "CorpusIndex":
        paper_by_id: dict[str, dict[str, Any]] = {}
        title_to_paper_id: dict[str, str] = {}
        topic_to_papers: dict[str, list[dict[str, Any]]] = {}
        chunks_by_paper_id: dict[str, list[dict[str, Any]]] = {}

        for paper in papers:
            paper_id = str(paper.get("paper_id") or "").strip()
            title = str(paper.get("title") or "").strip()
            if not paper_id or not title:
                continue
            normalized_title = normalize_text(title)
            paper_by_id[paper_id] = paper
            title_to_paper_id[normalized_title] = paper_id
            topic = paper.get("topic") or "Unknown topic"
            topic_to_papers.setdefault(topic, []).append(paper)

        for topic, topic_papers in topic_to_papers.items():
            topic_to_papers[topic] = sorted(topic_papers, key=paper_sort_key, reverse=True)

        for chunk in chunks or []:
            paper_id = str(chunk.get("paper_id") or "").strip()
            if not paper_id:
                continue
            chunks_by_paper_id.setdefault(paper_id, []).append(chunk)

        for paper_id, paper_chunks in chunks_by_paper_id.items():
            chunks_by_paper_id[paper_id] = sorted(
                paper_chunks,
                key=lambda row: (row.get("chunk_index") is None, row.get("chunk_index") or 0),
            )

        return cls(
            paper_by_id=paper_by_id,
            title_to_paper_id=title_to_paper_id,
            topic_to_papers=topic_to_papers,
            chunks_by_paper_id=chunks_by_paper_id,
        )

    @classmethod
    def from_bm25_artifact(cls, artifact: dict[str, Any]) -> "CorpusIndex":
        papers = []
        for paper in artifact.get("papers", []) or []:
            metadata = paper.get("metadata", {}) or {}
            papers.append({**metadata, **paper})
        return cls.from_records(papers, [])

    def topic_for_query(self, query: str) -> str | None:
        lowered = normalize_text(query)
        for topic in self.topic_to_papers:
            if normalize_text(topic) in lowered:
                return topic
        for alias, topic in TOPIC_ALIASES.items():
            if alias in lowered and topic in self.topic_to_papers:
                return topic
        return None

    def resolve_paper(self, query: str, *, min_score: float = 0.42) -> PaperMatch | None:
        normalized_query = normalize_text(query)
        if not normalized_query:
            return None

        quoted = re.findall(r"""["\']([^"\']{8,})["\']""", query)
        for phrase in quoted:
            normalized_phrase = normalize_text(phrase)
            if normalized_phrase in self.title_to_paper_id:
                paper_id = self.title_to_paper_id[normalized_phrase]
                return PaperMatch(paper_id, self.paper_by_id[paper_id]["title"], 1.0, "exact quoted title match")

        best: PaperMatch | None = None
        query_tokens = significant_tokens(normalized_query)
        for normalized_title, paper_id in self.title_to_paper_id.items():
            paper = self.paper_by_id[paper_id]
            if normalized_title and normalized_title in normalized_query:
                match = PaperMatch(paper_id, paper["title"], 1.0, "title appears in query")
                if best is None or match.score > best.score:
                    best = match
                continue

            title_tokens = significant_tokens(normalized_title)
            if not title_tokens or not query_tokens:
                continue
            overlap = query_tokens & title_tokens
            score = max(len(overlap) / max(4, len(title_tokens)), len(overlap) / max(1, len(query_tokens)))
            if score >= min_score and (best is None or score > best.score):
                best = PaperMatch(paper_id, paper["title"], round(score, 6), f"title token overlap: {', '.join(sorted(overlap))}")
        return best

    def ranked_papers(self, query: str, *, top_k: int, matched_by: list[str] | None = None) -> list[dict[str, Any]]:
        after_year, before_year = parse_year_filters(query)
        topic_filter = self.topic_for_query(query)
        lowered = normalize_text(query)
        require_survey = "survey" in lowered or "review" in lowered

        candidates = list(self.paper_by_id.values())
        if topic_filter:
            candidates = list(self.topic_to_papers.get(topic_filter, []))

        filtered = []
        for paper in candidates:
            year = safe_int(paper.get("year"))
            if after_year is not None and (year is None or year <= after_year):
                continue
            if before_year is not None and (year is None or year >= before_year):
                continue
            if require_survey and not is_survey_like(paper):
                continue
            filtered.append(paper)

        markers = matched_by or ["metadata_filter", "corpus_index"]
        return [paper_to_candidate(paper, markers) for paper in sorted(filtered, key=paper_sort_key, reverse=True)[:top_k]]

    def paper_candidate(self, paper_id: str, matched_by: list[str] | None = None) -> dict[str, Any] | None:
        paper = self.paper_by_id.get(paper_id)
        if not paper:
            return None
        return paper_to_candidate(paper, matched_by or ["paper_lookup", "corpus_index"])

    def chunk_candidates_for_paper(self, paper_id: str, *, top_k: int) -> list[dict[str, Any]]:
        chunks = self.chunks_by_paper_id.get(paper_id, [])[:top_k]
        return [chunk_to_candidate(chunk) for chunk in chunks]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def significant_tokens(value: str) -> set[str]:
    return {token for token in normalize_text(value).split() if len(token) > 2 and token not in STOPWORDS}


def safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_year_filters(query: str) -> tuple[int | None, int | None]:
    lowered = query.lower()
    after = None
    before = None
    after_match = re.search(r"\b(?:after|since)\s+(20\d{2}|19\d{2})", lowered)
    before_match = re.search(r"\bbefore\s+(20\d{2}|19\d{2})", lowered)
    range_match = re.search(r"\bbetween\s+(20\d{2}|19\d{2})\s+and\s+(20\d{2}|19\d{2})", lowered)
    if after_match:
        after = int(after_match.group(1))
    if before_match:
        before = int(before_match.group(1))
    if range_match:
        after = int(range_match.group(1)) - 1
        before = int(range_match.group(2)) + 1
    return after, before


def paper_sort_key(paper: dict[str, Any]) -> tuple[int, int, str]:
    return (safe_int(paper.get("citation_count")) or 0, safe_int(paper.get("year")) or 0, str(paper.get("title") or ""))


def is_survey_like(paper: dict[str, Any]) -> bool:
    text = " ".join(
        str(paper.get(field) or "")
        for field in ("title", "abstract", "main_contribution", "methodology")
    ).lower()
    return "survey" in text or "review" in text


def classify_question_pattern(query: str, *, has_chat_history: bool = False) -> QuestionPatternName:
    lowered = normalize_text(query)
    if not lowered:
        return "follow_up"
    if re.search(r"\b(that|this|it|they|them|its)\b", lowered) and has_chat_history:
        return "follow_up"
    comparison_tokens = ("compare", "versus", " vs ", "difference between", "contrast", "tradeoff")
    if any(token in lowered for token in comparison_tokens) or re.search(r"\bhow does\b.*\bdiffer\b", lowered):
        return "comparison"
    conceptual_phrases = (
        "making things up",
        "make things up",
        "stop a chatbot from making",
        "look up facts",
        "grounding llm outputs",
        "grounded in retrieved evidence",
    )
    if any(phrase in lowered for phrase in conceptual_phrases):
        return "concept_explanation"
    if any(token in lowered for token in ("top", "most cited", "highly cited", "latest", "recent", "newest", "published", "after ", "before ", "between ", "list", "show me")):
        return "ranked_list"
    if (("explain" in lowered or "summarize" in lowered) and "paper" in lowered) or any(token in lowered for token in ("that paper", "this paper")) or re.search(r"""["\'][^"\']{8,}["\']""", query):
        return "paper_lookup"
    if any(token in lowered for token in ("dataset", "benchmark", "metric", "method", "methodology", "experiment", "result", "evaluate", "evaluation")):
        return "dataset_method"
    if any(token in lowered for token in ("read first", "reading path", "what should i read", "papers should i read", "start with")):
        return "reading_path"
    if not any(anchor in lowered for anchor in TOPIC_ALIASES):
        return "out_of_corpus"
    return "concept_explanation"


def paper_to_candidate(paper: dict[str, Any], matched_by: list[str]) -> dict[str, Any]:
    return {
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title") or "Untitled paper",
        "topic": paper.get("topic") or "Unknown topic",
        "year": safe_int(paper.get("year")),
        "citation_count": safe_int(paper.get("citation_count")) or 0,
        "authors": paper.get("authors") or [],
        "abstract": paper.get("abstract"),
        "arxiv_id": paper.get("arxiv_id"),
        "url": paper.get("url"),
        "main_contribution": paper.get("main_contribution"),
        "methodology": paper.get("methodology"),
        "dataset_used": paper.get("dataset_used"),
        "key_result": paper.get("key_result"),
        "limitations": paper.get("limitations"),
        "hybrid_score": 0.0,
        "matched_by": matched_by,
    }


def chunk_to_candidate(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "paper_id": chunk.get("paper_id"),
        "title": chunk.get("title") or "Untitled paper",
        "topic": chunk.get("topic") or "Unknown topic",
        "year": safe_int(chunk.get("year")),
        "citation_count": safe_int(chunk.get("citation_count")) or 0,
        "chunk_index": chunk.get("chunk_index"),
        "total_chunks": chunk.get("total_chunks"),
        "section_hint": chunk.get("section_hint"),
        "word_count": chunk.get("word_count"),
        "text": chunk.get("text") or "No chunk text available.",
        "pdf_url": chunk.get("pdf_url"),
        "source_type": chunk.get("source_type"),
        "page_count": chunk.get("page_count"),
        "dense_score": 0.0,
        "matched_by": ["paper_lookup", "corpus_index"],
    }


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {path}")
    return [item for item in data if isinstance(item, dict)]


@lru_cache(maxsize=4)
def load_corpus_index(
    papers_path: str | Path = DEFAULT_PAPERS_PATH,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
) -> CorpusIndex:
    papers = load_json_list(Path(papers_path))
    chunks = load_json_list(Path(chunks_path))
    return CorpusIndex.from_records(papers, chunks)
