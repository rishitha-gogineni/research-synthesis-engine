"""Generate a natural-language research-paper eval fixture from corpus metadata.

Reads data/full_text_papers.json and produces tests/fixtures/eval_queries_natural.json
with ~150 queries covering how humans actually ask about research papers:

  - Concept overview      : "What is X?"
  - Method-specific       : "How does LoRA reduce memory?"
  - Comparison            : "How do X and Y differ?"
  - Dataset/benchmark     : "What benchmarks measure X?"
  - Paper-anchored        : "What does '{title}' propose?"
  - Section-specific      : "What are the limitations of X?"

Ground truth is deterministic from paper metadata:
  - `expected_relevant_ids` : paper_ids whose titles/topics match the query intent
  - `expected_keywords`     : semantic keywords the retrieved chunk should contain
  - `expected_topics`       : topics the retrieved paper should belong to

This eval is deliberately CHUNKING-AGNOSTIC — it does not encode any specific
chunk_id, only paper_id + keyword presence. Any chunking strategy that retrieves
the correct paper AND surfaces the right content will score well.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


PAPERS_PATH = Path("data/full_text_papers.json")
OUT_PATH = Path("tests/fixtures/eval_queries_natural.json")


# Topic → route mapping (matches RSE router expectations)
TOPIC_TO_ROUTE = {
    "Retrieval-Augmented Generation (RAG)": "paper_level",
    "Transformers / Attention Mechanisms": "paper_level",
    "LLM Evaluation & Hallucination Detection": "paper_level",
    "AI Agents & Tool Use": "paper_level",
    "Fine-tuning (LoRA / PEFT)": "paper_level",
}


def normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def find_papers_with_keyword(papers: list[dict], keyword: str, field: str = "title") -> list[str]:
    """Return paper_ids whose given field contains the keyword (case-insensitive)."""
    key = keyword.lower()
    return [p["paper_id"] for p in papers if key in (p.get(field) or "").lower()]


def find_papers_by_topic(papers: list[dict], topic: str) -> list[str]:
    return [p["paper_id"] for p in papers if p.get("topic") == topic]


def concept_overview_queries(papers: list[dict]) -> list[dict]:
    """'What is X?' / 'How does Y work?' style queries — topic-level intent."""
    queries = []
    concepts = [
        ("retrieval-augmented generation", "Retrieval-Augmented Generation (RAG)", ["retrieval-augmented", "rag", "retrieval"]),
        ("RAG", "Retrieval-Augmented Generation (RAG)", ["retrieval-augmented", "rag"]),
        ("transformer attention", "Transformers / Attention Mechanisms", ["attention", "transformer"]),
        ("self-attention", "Transformers / Attention Mechanisms", ["attention", "self-attention"]),
        ("hallucination detection", "LLM Evaluation & Hallucination Detection", ["hallucination", "detection"]),
        ("LLM evaluation", "LLM Evaluation & Hallucination Detection", ["evaluation", "benchmark"]),
        ("AI agents", "AI Agents & Tool Use", ["agent", "autonomous"]),
        ("tool use in LLMs", "AI Agents & Tool Use", ["tool", "agent"]),
        ("parameter-efficient fine-tuning", "Fine-tuning (LoRA / PEFT)", ["fine-tuning", "parameter-efficient", "peft"]),
        ("LoRA", "Fine-tuning (LoRA / PEFT)", ["lora", "low-rank"]),
    ]
    for concept, topic, kws in concepts:
        ids = find_papers_by_topic(papers, topic)[:5]
        if not ids:
            continue
        queries.append({
            "query": f"What is {concept}?",
            "expected_route": "paper_level",
            "expected_topics": [topic],
            "expected_keywords": kws,
            "expected_relevant_ids": ids,
            "category": "single_turn",
            "evaluation_focus": "concept_overview",
            "rationale": f"[factual] concept overview for {concept}",
        })
        queries.append({
            "query": f"How does {concept} work?",
            "expected_route": "paper_level",
            "expected_topics": [topic],
            "expected_keywords": kws,
            "expected_relevant_ids": ids,
            "category": "single_turn",
            "evaluation_focus": "concept_overview",
            "rationale": f"[factual] how-does-X-work for {concept}",
        })
    return queries


def method_specific_queries(papers: list[dict]) -> list[dict]:
    """Named-method questions — pull from papers whose title contains the method name."""
    methods = [
        ("LoRA", "reduce GPU memory during fine-tuning", ["lora", "low-rank", "memory"]),
        ("QLoRA", "quantize models for fine-tuning", ["qlora", "quantization", "4-bit"]),
        ("BitFit", "fine-tune with bias parameters only", ["bitfit", "bias", "fine-tuning"]),
        ("prefix tuning", "fine-tune with prefix parameters", ["prefix", "tuning"]),
        ("prompt tuning", "adapt models with soft prompts", ["prompt", "tuning", "soft"]),
        ("chain-of-thought", "improve reasoning in LLMs", ["chain-of-thought", "cot", "reasoning"]),
        ("ReAct", "combine reasoning and acting", ["react", "reasoning", "acting"]),
        ("Toolformer", "teach LLMs to use tools", ["toolformer", "tool", "api"]),
        ("BERT", "pre-train bidirectional transformers", ["bert", "bidirectional", "masked"]),
        ("TruthfulQA", "measure LLM truthfulness", ["truthfulqa", "truth", "falsehood"]),
        ("HaluEval", "evaluate hallucinations", ["halueval", "hallucination", "benchmark"]),
        ("attention", "compute contextual representations", ["attention", "context", "query"]),
        ("MiniLM", "distill transformers", ["minilm", "distillation", "compression"]),
        ("dense retrieval", "retrieve documents with embeddings", ["dense", "retrieval", "embedding"]),
        ("adapter", "add trainable modules to frozen models", ["adapter", "trainable", "modular"]),
    ]
    queries = []
    for method, purpose, kws in methods:
        ids = find_papers_with_keyword(papers, method, "title")
        if not ids:
            continue
        queries.append({
            "query": f"How does {method} {purpose}?",
            "expected_route": "paper_level",
            "expected_topics": [],
            "expected_keywords": kws,
            "expected_relevant_ids": ids[:5],
            "category": "single_turn",
            "evaluation_focus": "method_mechanism",
            "rationale": f"[factual] how {method} achieves its stated purpose",
        })
        queries.append({
            "query": f"What is {method}?",
            "expected_route": "paper_level",
            "expected_topics": [],
            "expected_keywords": kws,
            "expected_relevant_ids": ids[:5],
            "category": "single_turn",
            "evaluation_focus": "method_definition",
            "rationale": f"[factual] method definition for {method}",
        })
    return queries


def comparison_queries(papers: list[dict]) -> list[dict]:
    """Cross-topic comparison queries — natural 'X vs Y' questions."""
    comparisons = [
        (
            "How does RAG compare to fine-tuning for injecting new knowledge into LLMs?",
            ["Retrieval-Augmented Generation (RAG)", "Fine-tuning (LoRA / PEFT)"],
            ["rag", "fine-tuning", "knowledge"],
        ),
        (
            "What are the tradeoffs between prompt tuning and full fine-tuning?",
            ["Fine-tuning (LoRA / PEFT)"],
            ["prompt", "tuning", "fine-tuning"],
        ),
        (
            "How do dense retrieval and sparse retrieval differ?",
            ["Retrieval-Augmented Generation (RAG)"],
            ["dense", "sparse", "retrieval"],
        ),
        (
            "What is the difference between chain-of-thought and ReAct prompting?",
            ["AI Agents & Tool Use"],
            ["chain-of-thought", "react", "reasoning"],
        ),
        (
            "How do different hallucination detection methods compare?",
            ["LLM Evaluation & Hallucination Detection"],
            ["hallucination", "detection", "evaluation"],
        ),
        (
            "How do LoRA and QLoRA differ in parameter efficiency?",
            ["Fine-tuning (LoRA / PEFT)"],
            ["lora", "qlora", "quantization"],
        ),
        (
            "How do transformers compare to RNNs for sequence modeling?",
            ["Transformers / Attention Mechanisms"],
            ["transformer", "rnn", "sequence"],
        ),
        (
            "What are the differences between adapter and prompt-based tuning?",
            ["Fine-tuning (LoRA / PEFT)"],
            ["adapter", "prompt", "tuning"],
        ),
    ]
    queries = []
    for q, topics, kws in comparisons:
        ids: list[str] = []
        for topic in topics:
            ids.extend(find_papers_by_topic(papers, topic)[:3])
        if not ids:
            continue
        queries.append({
            "query": q,
            "expected_route": "hybrid_both",
            "expected_topics": topics,
            "expected_keywords": kws,
            "expected_relevant_ids": list(dict.fromkeys(ids))[:6],
            "category": "single_turn",
            "evaluation_focus": "cross_topic_comparison",
            "rationale": "[synthesis] cross-topic comparison",
        })
    return queries


def dataset_benchmark_queries(papers: list[dict]) -> list[dict]:
    """Dataset / benchmark discovery — 'what datasets are used for X'."""
    dataset_queries = [
        (
            "What datasets are used to evaluate hallucination in LLMs?",
            "LLM Evaluation & Hallucination Detection",
            ["hallucination", "dataset", "benchmark", "evaluation"],
        ),
        (
            "What benchmarks measure LLM reasoning capability?",
            "LLM Evaluation & Hallucination Detection",
            ["reasoning", "benchmark", "evaluation"],
        ),
        (
            "What datasets are used to train and evaluate RAG systems?",
            "Retrieval-Augmented Generation (RAG)",
            ["dataset", "benchmark", "rag"],
        ),
        (
            "Which benchmarks evaluate autonomous agent performance?",
            "AI Agents & Tool Use",
            ["benchmark", "agent", "evaluation"],
        ),
        (
            "What datasets are used for parameter-efficient fine-tuning studies?",
            "Fine-tuning (LoRA / PEFT)",
            ["dataset", "fine-tuning", "benchmark"],
        ),
        (
            "Which vision benchmarks evaluate transformer architectures?",
            "Transformers / Attention Mechanisms",
            ["vision", "benchmark", "imagenet"],
        ),
    ]
    queries = []
    for q, topic, kws in dataset_queries:
        ids = find_papers_by_topic(papers, topic)[:5]
        if not ids:
            continue
        queries.append({
            "query": q,
            "expected_route": "paper_level",
            "expected_topics": [topic],
            "expected_keywords": kws,
            "expected_relevant_ids": ids,
            "category": "single_turn",
            "evaluation_focus": "dataset_discovery",
            "rationale": "[factual] dataset / benchmark discovery",
        })
    return queries


def paper_anchored_queries(papers: list[dict]) -> list[dict]:
    """'What does <paper title> propose?' — precise paper_id ground truth."""
    # Pick well-known / high-citation papers (top 40 by citation)
    ranked = sorted(papers, key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
    picks = ranked[:40]
    queries = []
    templates = [
        ("What does the paper '{title}' propose?", "paper_contribution", ["propose", "method", "novel"]),
        ("What is the main contribution of '{title}'?", "paper_contribution", ["contribution", "novel"]),
    ]
    for p in picks:
        title = normalize_title(p.get("title") or "")
        if not title or len(title) < 15:
            continue
        for tpl, focus, kws in templates:
            queries.append({
                "query": tpl.format(title=title),
                "expected_route": "paper_level",
                "expected_topics": [p.get("topic")] if p.get("topic") else [],
                "expected_keywords": kws,
                "expected_relevant_ids": [p["paper_id"]],
                "category": "single_turn",
                "evaluation_focus": focus,
                "rationale": f"[factual] paper-anchored recall for {title[:40]}",
            })
    return queries


def section_specific_queries(papers: list[dict]) -> list[dict]:
    """Section-specific questions — 'what are the limitations of X?' etc."""
    queries = []
    section_templates = [
        ("What are the limitations of {topic}?", ["limitation", "future work", "shortcoming"], "limitations"),
        ("What results do {topic} papers report?", ["result", "performance", "achieve"], "results"),
        ("What methodology do {topic} papers use?", ["method", "approach", "architecture"], "methodology"),
    ]
    for topic in [
        "Retrieval-Augmented Generation (RAG)",
        "AI Agents & Tool Use",
        "Fine-tuning (LoRA / PEFT)",
        "LLM Evaluation & Hallucination Detection",
        "Transformers / Attention Mechanisms",
    ]:
        ids = find_papers_by_topic(papers, topic)[:5]
        if not ids:
            continue
        # Simplified topic name for query
        simple = topic.split("(")[0].strip().lower()
        for tpl, kws, focus in section_templates:
            queries.append({
                "query": tpl.format(topic=simple),
                "expected_route": "chunk_level",
                "expected_topics": [topic],
                "expected_keywords": kws,
                "expected_relevant_ids": ids,
                "category": "single_turn",
                "evaluation_focus": focus,
                "rationale": f"[factual] section-specific recall on {topic}",
            })
    return queries


def out_of_corpus_queries() -> list[dict]:
    """Confidence-gate stressors: questions the corpus can't answer."""
    ooc = [
        "What are the latest developments in marine biology and coral bleaching?",
        "How do I bake sourdough bread at high altitude?",
        "What is the current stock price of NVIDIA?",
        "Explain the history of the Roman Empire.",
        "What are the best practices for treating type 2 diabetes?",
    ]
    return [
        {
            "query": q,
            "expected_route": "paper_level",
            "expected_topics": [],
            "expected_keywords": [],
            "expected_relevant_ids": [],
            "expected_confidence_decision": "insufficient_evidence",
            "category": "out_of_corpus",
            "evaluation_focus": "confidence_gate",
            "rationale": "[gate] out-of-corpus, gate must decline",
        }
        for q in ooc
    ]


def add_defaults(queries: list[dict]) -> list[dict]:
    """Fill required schema fields the RSE fixture expects."""
    for q in queries:
        q.setdefault("chat_history", [])
        q.setdefault("acceptable_routes", [])
        q.setdefault("expected_standalone_keywords", [])
        q.setdefault("expected_confidence_decision", None)
    return queries


def main() -> None:
    papers = json.loads(PAPERS_PATH.read_text())
    print(f"Loaded {len(papers)} papers")

    all_queries: list[dict] = []
    all_queries.extend(concept_overview_queries(papers))
    all_queries.extend(method_specific_queries(papers))
    all_queries.extend(comparison_queries(papers))
    all_queries.extend(dataset_benchmark_queries(papers))
    all_queries.extend(paper_anchored_queries(papers))
    all_queries.extend(section_specific_queries(papers))
    all_queries.extend(out_of_corpus_queries())

    add_defaults(all_queries)

    # Breakdown
    focus_counts: dict[str, int] = defaultdict(int)
    for q in all_queries:
        focus_counts[q["evaluation_focus"]] += 1

    print(f"\n=== EVAL FIXTURE SUMMARY ===")
    print(f"Total queries: {len(all_queries)}")
    print(f"\nBy evaluation focus:")
    for focus, count in sorted(focus_counts.items(), key=lambda x: -x[1]):
        print(f"  {focus:<28} {count:>4}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_queries, indent=2, ensure_ascii=False))
    print(f"\nWritten to: {OUT_PATH}")


if __name__ == "__main__":
    main()
