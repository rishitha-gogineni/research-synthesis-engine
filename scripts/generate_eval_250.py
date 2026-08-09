"""Generate a 250-query evaluation fixture covering 13 categories.

Ground truth is deterministic from corpus metadata (paper_id, topic, title,
citation_count, year, key_result, dataset_used, limitations, methodology).
Works with ANY chunking strategy — no chunk_ids in expected_relevant_ids.

Usage:
    python scripts/generate_eval_250.py

Outputs:
    tests/fixtures/eval_queries_250.json
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
PAPERS_PATH = ROOT / "data" / "enriched_papers_final.json"
CHUNKS_PATH = ROOT / "data" / "full_text_chunks_v2.json"
OUTPUT_PATH = ROOT / "tests" / "fixtures" / "eval_queries_250.json"

random.seed(42)


def load_papers() -> list[dict]:
    return json.load(PAPERS_PATH.open(encoding="utf-8"))


def load_chunks() -> list[dict]:
    return json.load(CHUNKS_PATH.open(encoding="utf-8"))


def papers_by_topic(papers: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in papers:
        out.setdefault(p["topic"], []).append(p)
    for v in out.values():
        v.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
    return out


def has_field(paper: dict, field: str) -> bool:
    val = paper.get(field, "")
    return bool(val) and val.lower() != "not specified"


def short_title(title: str) -> str:
    if len(title) > 60:
        return title[:57] + "..."
    return title


# Valid schema values for evaluation_focus (from shared/schemas.py EvaluationFocusName)
VALID_FOCUS = {
    "route_selection", "full_text_evidence", "metadata_filter",
    "cross_topic_comparison", "contextual_rewrite", "confidence_gate", "reading_path",
}

# Map our semantic categories to valid schema values
FOCUS_MAP = {
    "factual_recall": "full_text_evidence",
    "methodology_evidence": "full_text_evidence",
    "dataset_discovery": "full_text_evidence",
    "limitation_analysis": "full_text_evidence",
    "cross_topic_comparison": "cross_topic_comparison",
    "temporal_evolution": "full_text_evidence",
    "metadata_filter": "metadata_filter",
    "reading_path": "reading_path",
    "section_specific": "full_text_evidence",
    "abstract_overview": "route_selection",
    "confidence_gate": "confidence_gate",
    "multi_turn": "contextual_rewrite",
    "adversarial": "route_selection",
}


def make_query(
    query: str,
    expected_route: str,
    expected_relevant_ids: list[str],
    category: str,
    evaluation_focus: str,
    rationale: str,
    *,
    expected_topics: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    acceptable_routes: list[str] | None = None,
    expected_confidence_decision: str | None = None,
    chat_history: list[dict] | None = None,
    expected_standalone_keywords: list[str] | None = None,
) -> dict:
    mapped_focus = FOCUS_MAP.get(evaluation_focus, evaluation_focus)
    if mapped_focus not in VALID_FOCUS:
        mapped_focus = "route_selection"
    return {
        "query": query,
        "expected_route": expected_route,
        "expected_topics": expected_topics or [],
        "expected_keywords": expected_keywords or [],
        "expected_relevant_ids": expected_relevant_ids,
        "category": category,
        "evaluation_focus": mapped_focus,
        "rationale": rationale,
        "chat_history": chat_history or [],
        "acceptable_routes": acceptable_routes or [expected_route],
        "expected_standalone_keywords": expected_standalone_keywords or [],
        "expected_confidence_decision": expected_confidence_decision,
    }


# ---------------------------------------------------------------------------
# Category 1: Factual Recall — Numerical Results (30 queries)
# ---------------------------------------------------------------------------

def generate_factual_recall(papers: list[dict]) -> list[dict]:
    queries = []
    candidates = [p for p in papers if has_field(p, "key_result") and any(c.isdigit() for c in p["key_result"])]
    random.shuffle(candidates)

    templates = [
        "What quantitative results does '{title}' report?",
        "What are the key numerical findings in '{title}'?",
        "What performance metrics does the paper '{title}' achieve?",
        "What benchmark scores does '{title}' report?",
        "How well does the method in '{title}' perform?",
    ]

    for i, p in enumerate(candidates[:30]):
        template = templates[i % len(templates)]
        query = template.format(title=short_title(p["title"]))
        keywords = []
        for num in re.findall(r"\d+\.?\d*%?", p["key_result"]):
            keywords.append(num)
        if not keywords:
            keywords = [p["title"].split()[0].lower()]

        queries.append(make_query(
            query=query,
            expected_route="chunk_level",
            expected_relevant_ids=[p["paper_id"]],
            category="single_turn",
            evaluation_focus="factual_recall",
            rationale=f"[factual] numeric result from {p['title'][:40]}",
            expected_topics=[p["topic"]],
            expected_keywords=keywords[:3],
            acceptable_routes=["chunk_level", "paper_level"],
        ))
    return queries


# ---------------------------------------------------------------------------
# Category 2: Methodology Deep-Dive (25 queries)
# ---------------------------------------------------------------------------

def generate_methodology(papers: list[dict]) -> list[dict]:
    queries = []
    candidates = [p for p in papers if has_field(p, "methodology")]
    random.shuffle(candidates)

    templates = [
        "How does the method proposed in '{title}' work?",
        "What methodology does '{title}' use?",
        "Explain the approach used in '{title}'.",
        "What technique does '{title}' introduce?",
        "Describe the architecture proposed in '{title}'.",
    ]

    for i, p in enumerate(candidates[:25]):
        template = templates[i % len(templates)]
        query = template.format(title=short_title(p["title"]))
        method_words = [w.lower() for w in p["methodology"].split()[:5] if len(w) > 4]

        queries.append(make_query(
            query=query,
            expected_route="chunk_level",
            expected_relevant_ids=[p["paper_id"]],
            category="single_turn",
            evaluation_focus="methodology_evidence",
            rationale=f"[methodology] deep-dive into {p['title'][:40]}",
            expected_topics=[p["topic"]],
            expected_keywords=method_words[:2] if method_words else [p["title"].split()[0].lower()],
            acceptable_routes=["chunk_level", "paper_level"],
        ))
    return queries


# ---------------------------------------------------------------------------
# Category 3: Dataset/Benchmark Discovery (20 queries)
# ---------------------------------------------------------------------------

def generate_dataset_discovery(papers: list[dict], by_topic: dict[str, list[dict]]) -> list[dict]:
    queries = []

    dataset_papers = [p for p in papers if has_field(p, "dataset_used")]

    dataset_keywords = {}
    for p in dataset_papers:
        ds = p["dataset_used"].lower()
        for kw in ["imagenet", "squad", "mmlu", "truthfulqa", "beir", "coco", "wmt",
                   "medical", "pubmed", "alfred", "hotpotqa", "natural questions",
                   "spider", "humaneval", "ade20k"]:
            if kw in ds:
                dataset_keywords.setdefault(kw, []).append(p)

    templates = [
        "Which papers evaluate on {dataset}?",
        "What papers use the {dataset} benchmark?",
        "Show me papers that use {dataset} for evaluation.",
        "Which studies in the corpus use {dataset}?",
    ]

    used_datasets = []
    for ds, ds_papers in sorted(dataset_keywords.items(), key=lambda x: -len(x[1])):
        if len(ds_papers) >= 2 and len(used_datasets) < 10:
            used_datasets.append((ds, ds_papers))

    for i, (ds, ds_papers) in enumerate(used_datasets):
        template = templates[i % len(templates)]
        query = template.format(dataset=ds.title())
        queries.append(make_query(
            query=query,
            expected_route="paper_level",
            expected_relevant_ids=[p["paper_id"] for p in ds_papers[:5]],
            category="single_turn",
            evaluation_focus="dataset_discovery",
            rationale=f"[dataset] papers using {ds}",
            expected_keywords=[ds],
            acceptable_routes=["paper_level", "hybrid_both"],
        ))

    topic_dataset_templates = [
        "What benchmarks are used to evaluate {topic_short}?",
        "Which datasets are commonly used in {topic_short} research?",
        "What evaluation benchmarks exist for {topic_short}?",
    ]

    topic_short_names = {
        "Retrieval-Augmented Generation (RAG)": "RAG",
        "Transformers / Attention Mechanisms": "vision transformers",
        "LLM Evaluation & Hallucination Detection": "LLM evaluation",
        "AI Agents & Tool Use": "AI agents",
        "Fine-tuning (LoRA / PEFT)": "parameter-efficient fine-tuning",
    }

    for i, (topic, topic_papers) in enumerate(by_topic.items()):
        if len(queries) >= 20:
            break
        ds_in_topic = [p for p in topic_papers if has_field(p, "dataset_used")]
        if ds_in_topic:
            template = topic_dataset_templates[i % len(topic_dataset_templates)]
            short = topic_short_names.get(topic, topic)
            query = template.format(topic_short=short)
            queries.append(make_query(
                query=query,
                expected_route="paper_level",
                expected_relevant_ids=[p["paper_id"] for p in ds_in_topic[:5]],
                category="single_turn",
                evaluation_focus="dataset_discovery",
                rationale=f"[dataset] benchmarks for {short}",
                expected_topics=[topic],
                expected_keywords=["dataset", "benchmark", "evaluat"],
                acceptable_routes=["paper_level", "hybrid_both"],
            ))

    return queries[:20]


# ---------------------------------------------------------------------------
# Category 4: Limitation & Failure Analysis (20 queries)
# ---------------------------------------------------------------------------

def generate_limitations(papers: list[dict], by_topic: dict[str, list[dict]]) -> list[dict]:
    queries = []

    papers_with_lim = [p for p in papers if has_field(p, "limitations")]
    random.shuffle(papers_with_lim)

    paper_templates = [
        "What are the limitations discussed in '{title}'?",
        "What problems or weaknesses does '{title}' identify?",
        "What challenges are mentioned in '{title}'?",
    ]

    for i, p in enumerate(papers_with_lim[:10]):
        template = paper_templates[i % len(paper_templates)]
        query = template.format(title=short_title(p["title"]))
        queries.append(make_query(
            query=query,
            expected_route="chunk_level",
            expected_relevant_ids=[p["paper_id"]],
            category="single_turn",
            evaluation_focus="limitation_analysis",
            rationale=f"[limitation] from {p['title'][:40]}",
            expected_topics=[p["topic"]],
            expected_keywords=["limitation", "challenge", "problem"],
            acceptable_routes=["chunk_level", "paper_level"],
        ))

    topic_short_names = {
        "Retrieval-Augmented Generation (RAG)": "RAG systems",
        "Transformers / Attention Mechanisms": "transformer architectures",
        "LLM Evaluation & Hallucination Detection": "LLM evaluation methods",
        "AI Agents & Tool Use": "LLM-based agents",
        "Fine-tuning (LoRA / PEFT)": "parameter-efficient fine-tuning",
    }

    topic_templates = [
        "What are the main limitations of {topic_short}?",
        "What doesn't work well about {topic_short}?",
        "What are the known failure modes of {topic_short}?",
        "What challenges remain unsolved in {topic_short}?",
    ]

    for i, (topic, topic_papers) in enumerate(by_topic.items()):
        short = topic_short_names.get(topic, topic)
        template = topic_templates[i % len(topic_templates)]
        lim_papers = [p for p in topic_papers if has_field(p, "limitations")]
        if not lim_papers:
            lim_papers = topic_papers[:3]
        queries.append(make_query(
            query=template.format(topic_short=short),
            expected_route="chunk_level",
            expected_relevant_ids=[p["paper_id"] for p in lim_papers[:4]],
            category="single_turn",
            evaluation_focus="limitation_analysis",
            rationale=f"[limitation] topic-level for {short}",
            expected_topics=[topic],
            expected_keywords=["limitation", "challenge", "weakness"],
            acceptable_routes=["chunk_level", "hybrid_both"],
        ))

    # General limitation questions
    general_lim = [
        ("What are the main problems with large language models?", ["LLM Evaluation & Hallucination Detection", "Fine-tuning (LoRA / PEFT)"]),
        ("What causes hallucinations in LLMs?", ["LLM Evaluation & Hallucination Detection", "Retrieval-Augmented Generation (RAG)"]),
        ("Why is evaluation of LLMs difficult?", ["LLM Evaluation & Hallucination Detection"]),
        ("What are the computational challenges of fine-tuning large models?", ["Fine-tuning (LoRA / PEFT)"]),
        ("What are the failure modes when using RAG in production?", ["Retrieval-Augmented Generation (RAG)"]),
    ]

    for query_text, topics in general_lim:
        if len(queries) >= 20:
            break
        relevant = [p for p in papers if p["topic"] in topics and has_field(p, "limitations")]
        if not relevant:
            relevant = [p for p in papers if p["topic"] in topics][:3]
        queries.append(make_query(
            query=query_text,
            expected_route="chunk_level",
            expected_relevant_ids=[p["paper_id"] for p in relevant[:5]],
            category="single_turn",
            evaluation_focus="limitation_analysis",
            rationale=f"[limitation] general: {query_text[:40]}",
            expected_topics=topics,
            expected_keywords=["limitation", "challenge", "problem", "failure"],
            acceptable_routes=["chunk_level", "hybrid_both"],
        ))

    return queries[:20]


# ---------------------------------------------------------------------------
# Category 5: Cross-Topic Comparison (20 queries)
# ---------------------------------------------------------------------------

def generate_comparisons(papers: list[dict], by_topic: dict[str, list[dict]]) -> list[dict]:
    queries = []

    comparison_pairs = [
        ("Retrieval-Augmented Generation (RAG)", "Fine-tuning (LoRA / PEFT)",
         "How does RAG compare to fine-tuning for injecting new knowledge into LLMs?"),
        ("Retrieval-Augmented Generation (RAG)", "Fine-tuning (LoRA / PEFT)",
         "What are the trade-offs between RAG and parameter-efficient fine-tuning?"),
        ("Retrieval-Augmented Generation (RAG)", "LLM Evaluation & Hallucination Detection",
         "How effective is RAG at reducing hallucinations compared to other methods?"),
        ("AI Agents & Tool Use", "Retrieval-Augmented Generation (RAG)",
         "How do agentic RAG systems differ from standard RAG pipelines?"),
        ("AI Agents & Tool Use", "Fine-tuning (LoRA / PEFT)",
         "Is it better to fine-tune an LLM or give it tools for domain-specific tasks?"),
        ("Transformers / Attention Mechanisms", "Fine-tuning (LoRA / PEFT)",
         "How does LoRA interact with attention mechanisms in transformers?"),
        ("LLM Evaluation & Hallucination Detection", "AI Agents & Tool Use",
         "How do you evaluate the reliability of LLM-based agents?"),
        ("Transformers / Attention Mechanisms", "LLM Evaluation & Hallucination Detection",
         "How are transformer-based models evaluated for code generation?"),
        ("Retrieval-Augmented Generation (RAG)", "AI Agents & Tool Use",
         "What's the difference between a RAG system and an AI agent with search tools?"),
        ("Fine-tuning (LoRA / PEFT)", "LLM Evaluation & Hallucination Detection",
         "Does fine-tuning increase or decrease hallucination rates?"),
        ("Retrieval-Augmented Generation (RAG)", "Transformers / Attention Mechanisms",
         "How do attention mechanisms help in retrieval-augmented generation?"),
        ("AI Agents & Tool Use", "LLM Evaluation & Hallucination Detection",
         "What benchmarks exist for evaluating autonomous AI agents?"),
        ("Transformers / Attention Mechanisms", "AI Agents & Tool Use",
         "How do vision transformers enable embodied AI agents?"),
        ("Fine-tuning (LoRA / PEFT)", "Retrieval-Augmented Generation (RAG)",
         "When should you use LoRA versus RAG for a domain-specific application?"),
        ("LLM Evaluation & Hallucination Detection", "Retrieval-Augmented Generation (RAG)",
         "How is faithfulness measured in RAG systems versus standalone LLMs?"),
        ("AI Agents & Tool Use", "Transformers / Attention Mechanisms",
         "What role does the transformer architecture play in multi-agent systems?"),
        ("Fine-tuning (LoRA / PEFT)", "AI Agents & Tool Use",
         "Can you fine-tune an LLM to be a better agent?"),
        ("LLM Evaluation & Hallucination Detection", "Fine-tuning (LoRA / PEFT)",
         "How do you evaluate whether fine-tuning improved a model?"),
        ("Transformers / Attention Mechanisms", "Retrieval-Augmented Generation (RAG)",
         "How are cross-attention mechanisms used in retrieval-augmented models?"),
        ("AI Agents & Tool Use", "Fine-tuning (LoRA / PEFT)",
         "Compare prompt engineering versus fine-tuning for building AI agents."),
    ]

    for topic1, topic2, query_text in comparison_pairs[:20]:
        t1_papers = by_topic[topic1][:3]
        t2_papers = by_topic[topic2][:3]
        relevant_ids = [p["paper_id"] for p in t1_papers + t2_papers]

        queries.append(make_query(
            query=query_text,
            expected_route="hybrid_both",
            expected_relevant_ids=relevant_ids,
            category="single_turn",
            evaluation_focus="cross_topic_comparison",
            rationale=f"[comparison] {topic1[:20]} vs {topic2[:20]}",
            expected_topics=[topic1, topic2],
            expected_keywords=["compare", "difference", "trade-off", "versus"],
            acceptable_routes=["hybrid_both", "paper_level"],
        ))

    return queries[:20]


# ---------------------------------------------------------------------------
# Category 6: Temporal/Evolution (15 queries)
# ---------------------------------------------------------------------------

def generate_temporal(papers: list[dict], by_topic: dict[str, list[dict]]) -> list[dict]:
    queries = []

    temporal_questions = [
        ("How has retrieval-augmented generation evolved since 2020?",
         "Retrieval-Augmented Generation (RAG)", ["rag", "retrieval"]),
        ("What RAG improvements were proposed after 2022?",
         "Retrieval-Augmented Generation (RAG)", ["rag", "retrieval", "improv"]),
        ("How has the Transformer architecture evolved since Attention Is All You Need?",
         "Transformers / Attention Mechanisms", ["transformer", "attention"]),
        ("What vision transformer variants emerged after 2021?",
         "Transformers / Attention Mechanisms", ["vision", "transformer", "vit"]),
        ("How has LLM evaluation changed since GPT-3?",
         "LLM Evaluation & Hallucination Detection", ["evaluation", "benchmark"]),
        ("What hallucination detection methods were developed after 2023?",
         "LLM Evaluation & Hallucination Detection", ["hallucination", "detection"]),
        ("How have AI agent architectures evolved since 2023?",
         "AI Agents & Tool Use", ["agent", "autonomous"]),
        ("What tool-use methods for LLMs emerged after ReAct?",
         "AI Agents & Tool Use", ["tool", "agent", "react"]),
        ("How has parameter-efficient fine-tuning evolved from adapters to LoRA?",
         "Fine-tuning (LoRA / PEFT)", ["lora", "adapter", "parameter"]),
        ("What fine-tuning methods came after LoRA?",
         "Fine-tuning (LoRA / PEFT)", ["lora", "qlora", "fine-tun"]),
        ("What are the most recent developments in RAG evaluation?",
         "Retrieval-Augmented Generation (RAG)", ["evaluation", "rag", "benchmark"]),
        ("How has attention efficiency improved over time?",
         "Transformers / Attention Mechanisms", ["efficient", "attention", "linear"]),
        ("What's the progression from single-agent to multi-agent systems?",
         "AI Agents & Tool Use", ["multi-agent", "agent"]),
        ("How has prompt tuning evolved as an alternative to full fine-tuning?",
         "Fine-tuning (LoRA / PEFT)", ["prompt", "tuning"]),
        ("What jailbreaking and safety evaluation methods appeared after 2023?",
         "LLM Evaluation & Hallucination Detection", ["jailbreak", "safety", "red-team"]),
    ]

    for query_text, topic, keywords in temporal_questions:
        topic_papers = by_topic[topic]
        queries.append(make_query(
            query=query_text,
            expected_route="paper_level",
            expected_relevant_ids=[p["paper_id"] for p in topic_papers[:5]],
            category="single_turn",
            evaluation_focus="temporal_evolution",
            rationale=f"[temporal] evolution in {topic[:30]}",
            expected_topics=[topic],
            expected_keywords=keywords,
            acceptable_routes=["paper_level", "hybrid_both"],
        ))

    return queries[:15]


# ---------------------------------------------------------------------------
# Category 7: Metadata Filter (20 queries)
# ---------------------------------------------------------------------------

def generate_metadata_filter(papers: list[dict], by_topic: dict[str, list[dict]]) -> list[dict]:
    queries = []

    topic_short = {
        "Retrieval-Augmented Generation (RAG)": "RAG",
        "Transformers / Attention Mechanisms": "transformer",
        "LLM Evaluation & Hallucination Detection": "LLM evaluation",
        "AI Agents & Tool Use": "AI agent",
        "Fine-tuning (LoRA / PEFT)": "fine-tuning",
    }

    # Topic + year filters
    for topic, short in topic_short.items():
        topic_papers = by_topic[topic]
        recent = [p for p in topic_papers if p.get("year", 0) >= 2024]
        if recent:
            queries.append(make_query(
                query=f"Show me {short} papers published in 2024 or later.",
                expected_route="metadata_filter",
                expected_relevant_ids=[p["paper_id"] for p in recent[:5]],
                category="single_turn",
                evaluation_focus="metadata_filter",
                rationale=f"[metadata] {short} after 2024",
                expected_topics=[topic],
                expected_keywords=[short.lower()],
                acceptable_routes=["metadata_filter", "paper_level"],
            ))

    # Topic + citation filters
    for topic, short in topic_short.items():
        topic_papers = by_topic[topic]
        high_cite = [p for p in topic_papers if p.get("citation_count", 0) >= 200]
        if high_cite:
            queries.append(make_query(
                query=f"Which {short} papers have more than 200 citations?",
                expected_route="metadata_filter",
                expected_relevant_ids=[p["paper_id"] for p in high_cite[:5]],
                category="single_turn",
                evaluation_focus="metadata_filter",
                rationale=f"[metadata] high-cited {short}",
                expected_topics=[topic],
                expected_keywords=[short.lower(), "citation"],
                acceptable_routes=["metadata_filter", "paper_level"],
            ))

    # Combined filters
    combined = [
        ("Show me highly cited RAG papers from 2023 or later.",
         "Retrieval-Augmented Generation (RAG)", lambda p: p.get("year", 0) >= 2023 and p.get("citation_count", 0) >= 100),
        ("List transformer papers with over 1000 citations.",
         "Transformers / Attention Mechanisms", lambda p: p.get("citation_count", 0) >= 1000),
        ("What are the most cited papers on hallucination detection?",
         "LLM Evaluation & Hallucination Detection", lambda p: p.get("citation_count", 0) >= 200),
        ("Show me recent agent papers from 2024.",
         "AI Agents & Tool Use", lambda p: p.get("year", 0) >= 2024),
        ("Which fine-tuning papers were published before 2023?",
         "Fine-tuning (LoRA / PEFT)", lambda p: p.get("year", 0) < 2023),
        ("List all survey papers in the corpus.",
         None, lambda p: "survey" in p.get("title", "").lower()),
        ("Show me papers about LoRA specifically.",
         "Fine-tuning (LoRA / PEFT)", lambda p: "lora" in p.get("title", "").lower()),
        ("Which papers discuss medical applications?",
         None, lambda p: "medic" in p.get("title", "").lower() or "medic" in p.get("dataset_used", "").lower()),
        ("Show me papers on code generation or evaluation.",
         "LLM Evaluation & Hallucination Detection", lambda p: "code" in p.get("title", "").lower()),
        ("List papers about multi-agent systems.",
         "AI Agents & Tool Use", lambda p: "multi-agent" in p.get("title", "").lower() or "multi-agent" in p.get("main_contribution", "").lower()),
    ]

    for query_text, topic, filter_fn in combined:
        matching = [p for p in papers if filter_fn(p)]
        if not matching:
            continue
        queries.append(make_query(
            query=query_text,
            expected_route="metadata_filter",
            expected_relevant_ids=[p["paper_id"] for p in matching[:5]],
            category="single_turn",
            evaluation_focus="metadata_filter",
            rationale=f"[metadata] {query_text[:40]}",
            expected_topics=[topic] if topic else [],
            expected_keywords=[],
            acceptable_routes=["metadata_filter", "paper_level"],
        ))

    return queries[:20]


# ---------------------------------------------------------------------------
# Category 8: Reading Path / Recommendation (15 queries)
# ---------------------------------------------------------------------------

def generate_reading_path(papers: list[dict], by_topic: dict[str, list[dict]]) -> list[dict]:
    queries = []

    topic_short = {
        "Retrieval-Augmented Generation (RAG)": "retrieval-augmented generation",
        "Transformers / Attention Mechanisms": "transformer architectures",
        "LLM Evaluation & Hallucination Detection": "LLM evaluation",
        "AI Agents & Tool Use": "LLM-based autonomous agents",
        "Fine-tuning (LoRA / PEFT)": "parameter-efficient fine-tuning",
    }

    templates = [
        "What papers should I read first to understand {topic}?",
        "What are the foundational papers on {topic}?",
        "Recommend key papers for someone new to {topic}.",
    ]

    for i, (topic, short) in enumerate(topic_short.items()):
        template = templates[i % len(templates)]
        query = template.format(topic=short)
        top_papers = by_topic[topic][:5]
        queries.append(make_query(
            query=query,
            expected_route="paper_level",
            expected_relevant_ids=[p["paper_id"] for p in top_papers],
            category="single_turn",
            evaluation_focus="reading_path",
            rationale=f"[reading_path] foundational for {short[:30]}",
            expected_topics=[topic],
            expected_keywords=["foundational", "read", "start", "recommend"],
            acceptable_routes=["paper_level", "hybrid_both"],
        ))

    specific_reading = [
        ("Where should I start to understand LoRA and adapter methods?",
         "Fine-tuning (LoRA / PEFT)", ["lora", "adapter"]),
        ("What surveys cover the RAG landscape?",
         "Retrieval-Augmented Generation (RAG)", ["survey", "rag"]),
        ("What are the must-read papers on attention mechanisms?",
         "Transformers / Attention Mechanisms", ["attention"]),
        ("Recommend papers on hallucination mitigation strategies.",
         "LLM Evaluation & Hallucination Detection", ["hallucination", "mitigation"]),
        ("What should I read to understand tool-use in LLMs?",
         "AI Agents & Tool Use", ["tool", "llm"]),
        ("What are the seminal papers in this research area?",
         None, []),
        ("Give me a reading list for understanding RAG evaluation.",
         "Retrieval-Augmented Generation (RAG)", ["evaluation", "rag"]),
        ("What background papers do I need for vision transformers?",
         "Transformers / Attention Mechanisms", ["vision", "transformer"]),
        ("Suggest key papers on LLM benchmarking.",
         "LLM Evaluation & Hallucination Detection", ["benchmark", "evaluation"]),
        ("What are the most important multi-agent papers?",
         "AI Agents & Tool Use", ["multi-agent", "agent"]),
    ]

    for query_text, topic, keywords in specific_reading:
        if len(queries) >= 15:
            break
        if topic:
            relevant = by_topic[topic][:5]
            topics = [topic]
        else:
            relevant = sorted(papers, key=lambda p: p.get("citation_count", 0), reverse=True)[:5]
            topics = []
        queries.append(make_query(
            query=query_text,
            expected_route="paper_level",
            expected_relevant_ids=[p["paper_id"] for p in relevant],
            category="single_turn",
            evaluation_focus="reading_path",
            rationale=f"[reading_path] {query_text[:40]}",
            expected_topics=topics,
            expected_keywords=keywords,
            acceptable_routes=["paper_level", "hybrid_both"],
        ))

    return queries[:15]


# ---------------------------------------------------------------------------
# Category 9: Section-Specific Extraction (20 queries)
# ---------------------------------------------------------------------------

def generate_section_specific(papers: list[dict], chunks: list[dict]) -> list[dict]:
    queries = []

    papers_with_chunks = {c["paper_id"] for c in chunks}
    chunk_papers = [p for p in papers if p["paper_id"] in papers_with_chunks]

    section_templates = {
        "experiments": [
            "What experiments did '{title}' run?",
            "How was '{title}' evaluated experimentally?",
            "What experimental setup does '{title}' describe?",
        ],
        "methodology": [
            "What is the method proposed in '{title}'?",
            "Describe the technical approach in '{title}'.",
            "What architecture does '{title}' propose?",
        ],
        "conclusion": [
            "What does '{title}' conclude?",
            "What are the main conclusions of '{title}'?",
            "What future work does '{title}' suggest?",
        ],
        "results": [
            "What results does '{title}' report?",
            "What are the main findings in '{title}'?",
        ],
        "limitations": [
            "What limitations does '{title}' acknowledge?",
            "What are the weaknesses discussed in '{title}'?",
        ],
    }

    chunks_by_paper_section: dict[str, dict[str, int]] = {}
    for c in chunks:
        pid = c["paper_id"]
        sec = c.get("section_hint", "")
        if sec:
            chunks_by_paper_section.setdefault(pid, {})
            chunks_by_paper_section[pid][sec] = chunks_by_paper_section[pid].get(sec, 0) + 1

    count_per_section = {"experiments": 0, "methodology": 0, "conclusion": 0, "results": 0, "limitations": 0}
    random.shuffle(chunk_papers)

    for p in chunk_papers:
        if len(queries) >= 20:
            break
        pid = p["paper_id"]
        if pid not in chunks_by_paper_section:
            continue
        for section, templates in section_templates.items():
            if count_per_section[section] >= 5:
                continue
            if section in chunks_by_paper_section.get(pid, {}):
                template = templates[count_per_section[section] % len(templates)]
                query = template.format(title=short_title(p["title"]))
                queries.append(make_query(
                    query=query,
                    expected_route="chunk_level",
                    expected_relevant_ids=[pid],
                    category="single_turn",
                    evaluation_focus="section_specific",
                    rationale=f"[section:{section}] from {p['title'][:40]}",
                    expected_topics=[p["topic"]],
                    expected_keywords=[section.replace("_", " ")],
                    acceptable_routes=["chunk_level", "paper_level"],
                ))
                count_per_section[section] += 1
                break

    return queries[:20]


# ---------------------------------------------------------------------------
# Category 10: Abstract-Level Overview (20 queries)
# ---------------------------------------------------------------------------

def generate_abstract_overview(papers: list[dict], by_topic: dict[str, list[dict]]) -> list[dict]:
    queries = []

    templates = [
        "What is the main contribution of '{title}'?",
        "What does '{title}' propose?",
        "What problem does '{title}' solve?",
        "Summarize '{title}' in a few sentences.",
    ]

    selected = []
    for topic, topic_papers in by_topic.items():
        selected.extend(topic_papers[:4])
    random.shuffle(selected)

    for i, p in enumerate(selected[:20]):
        template = templates[i % len(templates)]
        query = template.format(title=short_title(p["title"]))
        contribution_words = []
        if has_field(p, "main_contribution"):
            contribution_words = [w.lower() for w in p["main_contribution"].split()[:4] if len(w) > 4]

        queries.append(make_query(
            query=query,
            expected_route="paper_level",
            expected_relevant_ids=[p["paper_id"]],
            category="single_turn",
            evaluation_focus="abstract_overview",
            rationale=f"[abstract] main contribution of {p['title'][:40]}",
            expected_topics=[p["topic"]],
            expected_keywords=contribution_words[:2] if contribution_words else [p["title"].split()[0].lower()],
            acceptable_routes=["paper_level", "hybrid_both"],
        ))

    return queries[:20]


# ---------------------------------------------------------------------------
# Category 11: Out-of-Corpus / Confidence Gate (15 queries)
# ---------------------------------------------------------------------------

def generate_out_of_corpus() -> list[dict]:
    out_of_corpus_questions = [
        "What does this system know about quantum computing?",
        "Explain how DALL-E generates images from text.",
        "What are the latest developments in autonomous driving?",
        "How does AlphaFold predict protein structure?",
        "What is the current state of nuclear fusion research?",
        "Explain the physics behind gravitational waves.",
        "What machine learning methods are used in climate modeling?",
        "How do recommendation systems work at Netflix?",
        "What is the role of AI in drug discovery?",
        "Explain reinforcement learning for robotics control.",
        "What are the ethical concerns with facial recognition?",
        "How does blockchain consensus work?",
        "What is the current state of quantum machine learning?",
        "How do self-driving cars handle edge cases?",
        "What is neurosymbolic AI and how does it work?",
    ]

    queries = []
    for q in out_of_corpus_questions:
        queries.append(make_query(
            query=q,
            expected_route="paper_level",
            expected_relevant_ids=[],
            category="out_of_corpus",
            evaluation_focus="confidence_gate",
            rationale=f"[confidence] out-of-corpus: {q[:40]}",
            expected_confidence_decision="insufficient_evidence",
            acceptable_routes=["paper_level", "chunk_level", "hybrid_both", "metadata_filter"],
        ))

    return queries[:15]


# ---------------------------------------------------------------------------
# Category 12: Multi-Turn Conversational (15 queries)
# ---------------------------------------------------------------------------

def generate_multi_turn(papers: list[dict], by_topic: dict[str, list[dict]]) -> list[dict]:
    queries = []

    conversations = [
        {
            "turns": [
                {"role": "user", "content": "What is LoRA?"},
                {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that injects trainable low-rank matrices into transformer layers."},
            ],
            "query": "How does it compare to full fine-tuning in terms of memory?",
            "topic": "Fine-tuning (LoRA / PEFT)",
            "keywords": ["lora", "memory", "parameter"],
        },
        {
            "turns": [
                {"role": "user", "content": "Tell me about RAG systems."},
                {"role": "assistant", "content": "RAG systems combine retrieval with generation to ground LLM responses in external knowledge."},
            ],
            "query": "What are their main limitations?",
            "topic": "Retrieval-Augmented Generation (RAG)",
            "keywords": ["rag", "limitation"],
        },
        {
            "turns": [
                {"role": "user", "content": "What is the Transformer architecture?"},
                {"role": "assistant", "content": "The Transformer uses self-attention mechanisms instead of recurrence for sequence modeling."},
            ],
            "query": "What variants have been proposed for computer vision?",
            "topic": "Transformers / Attention Mechanisms",
            "keywords": ["vision", "transformer", "vit"],
        },
        {
            "turns": [
                {"role": "user", "content": "How do LLM-based agents work?"},
                {"role": "assistant", "content": "They use LLMs as reasoning engines that can plan, use tools, and take actions."},
            ],
            "query": "What benchmarks evaluate them?",
            "topic": "AI Agents & Tool Use",
            "keywords": ["benchmark", "evaluat", "agent"],
        },
        {
            "turns": [
                {"role": "user", "content": "What causes hallucinations in LLMs?"},
                {"role": "assistant", "content": "Hallucinations arise from training data noise, knowledge gaps, and the generative nature of language models."},
            ],
            "query": "How can RAG help reduce them?",
            "topic": "Retrieval-Augmented Generation (RAG)",
            "keywords": ["rag", "hallucination", "reduc"],
        },
        {
            "turns": [
                {"role": "user", "content": "What is BitFit?"},
                {"role": "assistant", "content": "BitFit is a parameter-efficient method that only fine-tunes the bias terms in a pretrained model."},
            ],
            "query": "How does it compare to LoRA and adapters?",
            "topic": "Fine-tuning (LoRA / PEFT)",
            "keywords": ["bitfit", "lora", "adapter", "compare"],
        },
        {
            "turns": [
                {"role": "user", "content": "What is TruthfulQA?"},
                {"role": "assistant", "content": "TruthfulQA is a benchmark that measures whether language models generate truthful answers."},
            ],
            "query": "What did they find about model size and truthfulness?",
            "topic": "LLM Evaluation & Hallucination Detection",
            "keywords": ["truthful", "model", "size"],
        },
        {
            "turns": [
                {"role": "user", "content": "Explain efficient attention mechanisms."},
                {"role": "assistant", "content": "Efficient attention reduces the quadratic complexity of standard self-attention through approximations or sparsity."},
            ],
            "query": "Which papers achieve linear complexity?",
            "topic": "Transformers / Attention Mechanisms",
            "keywords": ["linear", "attention", "efficient"],
        },
        {
            "turns": [
                {"role": "user", "content": "What are generative agents?"},
                {"role": "assistant", "content": "Generative agents are computational agents that simulate believable human behavior using LLMs."},
            ],
            "query": "How do they store and retrieve memories?",
            "topic": "AI Agents & Tool Use",
            "keywords": ["memory", "agent", "retriev"],
        },
        {
            "turns": [
                {"role": "user", "content": "What is RAGAS?"},
                {"role": "assistant", "content": "RAGAS is a framework for automated evaluation of RAG systems measuring faithfulness, relevance, and context recall."},
            ],
            "query": "What metrics does it compute?",
            "topic": "Retrieval-Augmented Generation (RAG)",
            "keywords": ["ragas", "metric", "faithfulness"],
        },
        {
            "turns": [
                {"role": "user", "content": "Tell me about LLaMA-Adapter."},
                {"role": "assistant", "content": "LLaMA-Adapter adds learnable adapters with zero-init attention to efficiently fine-tune LLaMA."},
            ],
            "query": "How many parameters does it add?",
            "topic": "Fine-tuning (LoRA / PEFT)",
            "keywords": ["llama", "adapter", "parameter"],
        },
        {
            "turns": [
                {"role": "user", "content": "What are the main RAG evaluation challenges?"},
                {"role": "assistant", "content": "Key challenges include measuring faithfulness, handling ambiguous queries, and evaluating retrieval quality separately from generation."},
            ],
            "query": "Which papers propose solutions to these?",
            "topic": "Retrieval-Augmented Generation (RAG)",
            "keywords": ["evaluation", "rag"],
        },
        {
            "turns": [
                {"role": "user", "content": "What is semantic entropy for hallucination detection?"},
                {"role": "assistant", "content": "Semantic entropy clusters LLM outputs by meaning and measures the entropy across clusters to detect uncertainty."},
            ],
            "query": "How effective is it compared to other detection methods?",
            "topic": "LLM Evaluation & Hallucination Detection",
            "keywords": ["semantic entropy", "detection", "hallucination"],
        },
        {
            "turns": [
                {"role": "user", "content": "What is cross-attention in transformers?"},
                {"role": "assistant", "content": "Cross-attention allows one sequence to attend to another, commonly used between encoder and decoder."},
            ],
            "query": "Which vision transformer papers use it?",
            "topic": "Transformers / Attention Mechanisms",
            "keywords": ["cross-attention", "vision"],
        },
        {
            "turns": [
                {"role": "user", "content": "How do embodied AI agents navigate environments?"},
                {"role": "assistant", "content": "They typically use a combination of visual perception, language understanding, and planning modules."},
            ],
            "query": "What datasets are used to test them?",
            "topic": "AI Agents & Tool Use",
            "keywords": ["dataset", "embodied", "navigation"],
        },
    ]

    for conv in conversations[:15]:
        topic = conv["topic"]
        relevant = by_topic[topic][:4]
        queries.append(make_query(
            query=conv["query"],
            expected_route="paper_level",
            expected_relevant_ids=[p["paper_id"] for p in relevant],
            category="multi_turn",
            evaluation_focus="multi_turn",
            rationale=f"[multi_turn] follow-up on {topic[:30]}",
            expected_topics=[topic],
            expected_keywords=conv["keywords"],
            chat_history=conv["turns"],
            expected_standalone_keywords=conv["keywords"],
            acceptable_routes=["paper_level", "chunk_level", "hybrid_both"],
        ))

    return queries[:15]


# ---------------------------------------------------------------------------
# Category 13: Adversarial / Edge Cases (15 queries)
# ---------------------------------------------------------------------------

def generate_adversarial(papers: list[dict], by_topic: dict[str, list[dict]]) -> list[dict]:
    queries = []

    adversarial_cases = [
        # Typos — should still retrieve
        ("What is LoRa fine tuning?", "Fine-tuning (LoRA / PEFT)", ["lora"], True),
        ("Tell me about retrevial augmented generation", "Retrieval-Augmented Generation (RAG)", ["retrieval", "rag"], True),
        ("How does atention mechanism work?", "Transformers / Attention Mechanisms", ["attention"], True),
        # Ambiguous — could go multiple ways
        ("What is attention?", "Transformers / Attention Mechanisms", ["attention"], True),
        ("Tell me about agents.", "AI Agents & Tool Use", ["agent"], True),
        # Too vague — system should still try but results may be noisy
        ("Tell me everything about AI.", None, ["ai"], True),
        ("What's interesting in this corpus?", None, [], True),
        # Paper exists but question targets wrong content
        ("What does Attention Is All You Need say about LoRA?", "Transformers / Attention Mechanisms", [], False),
        ("What does the LoRA paper say about hallucinations?", "Fine-tuning (LoRA / PEFT)", [], False),
        # Very specific — tests precision
        ("What is the exact BLEU score on WMT 2014 English-to-German from the original Transformer paper?",
         "Transformers / Attention Mechanisms", ["28.4", "bleu"], True),
        # Abbreviations
        ("What is PEFT?", "Fine-tuning (LoRA / PEFT)", ["parameter", "efficient"], True),
        ("Explain RAG.", "Retrieval-Augmented Generation (RAG)", ["retrieval", "augmented"], True),
        # Negation
        ("What methods do NOT use attention?", "Transformers / Attention Mechanisms", ["attention"], True),
        # Subjective
        ("Which is better, LoRA or full fine-tuning?", "Fine-tuning (LoRA / PEFT)", ["lora", "fine-tun"], True),
        # Counting/listing
        ("How many papers in this corpus discuss RAG?", "Retrieval-Augmented Generation (RAG)", ["rag"], True),
    ]

    for query_text, topic, keywords, should_retrieve in adversarial_cases:
        if topic:
            relevant = by_topic[topic][:3] if should_retrieve else []
        else:
            relevant = sorted(papers, key=lambda p: p.get("citation_count", 0), reverse=True)[:3] if should_retrieve else []

        queries.append(make_query(
            query=query_text,
            expected_route="paper_level",
            expected_relevant_ids=[p["paper_id"] for p in relevant],
            category="adversarial",
            evaluation_focus="adversarial",
            rationale=f"[adversarial] {query_text[:40]}",
            expected_topics=[topic] if topic else [],
            expected_keywords=keywords,
            expected_confidence_decision="insufficient_evidence" if not should_retrieve else None,
            acceptable_routes=["paper_level", "chunk_level", "hybrid_both", "metadata_filter"],
        ))

    return queries[:15]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading corpus data...")
    papers = load_papers()
    chunks = load_chunks()
    by_topic = papers_by_topic(papers)

    print(f"Papers: {len(papers)}, Chunks: {len(chunks)}, Topics: {len(by_topic)}")

    print("\nGenerating queries by category...")
    all_queries: list[dict] = []

    categories = [
        ("Factual Recall (numbers)", generate_factual_recall(papers)),
        ("Methodology Deep-Dive", generate_methodology(papers)),
        ("Dataset Discovery", generate_dataset_discovery(papers, by_topic)),
        ("Limitation Analysis", generate_limitations(papers, by_topic)),
        ("Cross-Topic Comparison", generate_comparisons(papers, by_topic)),
        ("Temporal Evolution", generate_temporal(papers, by_topic)),
        ("Metadata Filter", generate_metadata_filter(papers, by_topic)),
        ("Reading Path", generate_reading_path(papers, by_topic)),
        ("Section-Specific", generate_section_specific(papers, chunks)),
        ("Abstract Overview", generate_abstract_overview(papers, by_topic)),
        ("Out-of-Corpus", generate_out_of_corpus()),
        ("Multi-Turn", generate_multi_turn(papers, by_topic)),
        ("Adversarial", generate_adversarial(papers, by_topic)),
    ]

    for name, queries in categories:
        print(f"  {name}: {len(queries)} queries")
        all_queries.extend(queries)

    print(f"\nTotal: {len(all_queries)} queries")

    # Validate
    routes = {}
    focuses = {}
    cats = {}
    for q in all_queries:
        routes[q["expected_route"]] = routes.get(q["expected_route"], 0) + 1
        focuses[q["evaluation_focus"]] = focuses.get(q["evaluation_focus"], 0) + 1
        cats[q["category"]] = cats.get(q["category"], 0) + 1

    print(f"\nBy route: {dict(sorted(routes.items()))}")
    print(f"By focus: {dict(sorted(focuses.items()))}")
    print(f"By category: {dict(sorted(cats.items()))}")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_queries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Written to {OUTPUT_PATH}")
    print(f"   {len(all_queries)} queries across {len(categories)} categories")


if __name__ == "__main__":
    main()
