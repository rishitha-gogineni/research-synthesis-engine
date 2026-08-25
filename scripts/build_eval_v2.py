"""
Build a stronger golden eval set (eval_queries_v2.json) grounded in the REAL corpus.

Integrity rules:
- Every gold ID (paper OpenAlex URL or chunk-id) is chosen by matching the query's
  keywords against real corpus content within the correct topic. Nothing is fabricated.
- Out-of-scope / refusal queries intentionally carry NO relevant IDs (they test the
  system's ability to refuse). They still assert expected behavior via route.

Run:  python scripts/build_eval_v2.py
Output: tests/fixtures/eval_queries_v2.json  (+ a human-readable audit to stdout)
"""
import json, re, collections, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "tests" / "fixtures" / "eval_queries_v2.json"

papers = json.load(open(DATA / "enriched_papers_final.json"))
chunks = json.load(open(DATA / "full_text_chunks.json"))

TOPICS = sorted({p["topic"] for p in papers})

def paper_text(p):
    parts = [p.get(k) or "" for k in
             ("title", "abstract", "key_result", "main_contribution",
              "methodology", "dataset_used", "limitations")]
    return " ".join(parts).lower()

def chunk_text(c):
    return (c.get("title", "") + " " + c.get("text", "")).lower()

PAPER_TEXT = {p["paper_id"]: paper_text(p) for p in papers}
PAPER_TOPIC = {p["paper_id"]: p["topic"] for p in papers}
PAPER_YEAR = {p["paper_id"]: p.get("year") for p in papers}
PAPER_CITES = {p["paper_id"]: p.get("citation_count") or 0 for p in papers}
PAPER_TITLE = {p["paper_id"]: p.get("title") or "" for p in papers}

def score(text, keywords):
    return sum(text.count(k.lower()) for k in keywords)

def pick_papers(keywords, topics, n=3, min_hits=2, year_min=None, rank_by="score"):
    """rank_by='score' -> most keyword-relevant (tie-break citations).
       rank_by='citation' -> most-cited among on-topic keyword matches
       (for 'most-cited'/'foundational'/'reading-path' queries)."""
    cands = []
    for pid, txt in PAPER_TEXT.items():
        if topics and PAPER_TOPIC[pid] not in topics:
            continue
        if year_min and (PAPER_YEAR[pid] or 0) < year_min:
            continue
        s = score(txt, keywords)
        if s >= min_hits:
            if rank_by == "citation":
                cands.append((PAPER_CITES[pid], s, pid))
            else:
                cands.append((s, PAPER_CITES[pid], pid))
    cands.sort(reverse=True)
    return [pid for _, _, pid in cands[:n]]

def pick_chunks(keywords, topics, n=3, min_hits=2, restrict_papers=None):
    cands = []
    for c in chunks:
        if topics and c.get("topic") not in topics:
            continue
        if restrict_papers and c["paper_id"] not in restrict_papers:
            continue
        s = score(chunk_text(c), keywords)
        if s >= min_hits:
            cands.append((s, c["citation_count"] or 0, c["chunk_id"]))
    cands.sort(reverse=True)
    seen, out = set(), []
    for _, _, cid in cands:
        if cid not in seen:
            seen.add(cid); out.append(cid)
        if len(out) >= n: break
    return out

RAG   = "Retrieval-Augmented Generation (RAG)"
TFMR  = "Transformers / Attention Mechanisms"
EVAL  = "LLM Evaluation & Hallucination Detection"
AGENT = "AI Agents & Tool Use"
LORA  = "Fine-tuning (LoRA / PEFT)"

# Each spec: (query, route, topics, keywords, focus, category, extra)
# route in: paper_level | chunk_level | hybrid_both | metadata_filter | refuse
# extra: dict overrides - acceptable_routes, year_min, chunk_kw, restrict_to_matched_papers
SPECS = [
 # ---------- Simple factual / paper-level (15) ----------
 ("What are the main approaches to retrieval-augmented generation?", "paper_level", [RAG],
   ["retrieval-augmented","rag","retrieval","generation"], "route_selection", "factual", {}),
 ("Give an overview of parameter-efficient fine-tuning methods.", "paper_level", [LORA],
   ["parameter-efficient","peft","fine-tuning","adapter"], "route_selection", "factual", {}),
 ("What is LoRA and how does it reduce trainable parameters?", "paper_level", [LORA],
   ["lora","low-rank","trainable parameters","rank"], "route_selection", "factual", {}),
 ("Summarize research on LLM-based autonomous agents.", "paper_level", [AGENT],
   ["autonomous agents","agent","llm-based","survey"], "route_selection", "factual", {}),
 ("What are the key ideas behind the transformer architecture?", "paper_level", [TFMR],
   ["transformer","attention","self-attention","architecture"], "route_selection", "factual", {}),
 ("How is hallucination in large language models defined and categorized?", "paper_level", [EVAL],
   ["hallucination","factuality","faithfulness","categoriz"], "route_selection", "factual", {}),
 ("What methods exist for grounding LLM outputs in retrieved evidence?", "paper_level", [RAG,EVAL],
   ["grounding","retrieval","evidence","hallucination"], "route_selection", "factual", {}),
 ("Overview of attention mechanism variants for efficiency.", "paper_level", [TFMR],
   ["attention","efficient","sparse","linear attention"], "route_selection", "factual", {}),
 ("What are common tool-use frameworks for LLM agents?", "paper_level", [AGENT],
   ["tool","tool-use","framework","agent"], "route_selection", "factual", {}),
 ("Describe adapter-based fine-tuning approaches.", "paper_level", [LORA],
   ["adapter","fine-tuning","module","insert"], "route_selection", "factual", {}),
 ("What is retrieval-augmented generation used for in knowledge-intensive tasks?", "paper_level", [RAG],
   ["knowledge-intensive","retrieval","rag","question answering"], "route_selection", "factual", {}),
 ("How do LLMs perform planning for embodied or multi-step tasks?", "paper_level", [AGENT],
   ["planning","embodied","multi-step","plan"], "route_selection", "factual", {}),
 ("What benchmarks measure factual accuracy of language models?", "paper_level", [EVAL],
   ["truthfulqa","benchmark","factual","hallucination"], "route_selection", "factual", {}),
 ("What are positional encoding methods in transformers?", "paper_level", [TFMR],
   ["positional","encoding","rotary","position"], "route_selection", "factual", {}),
 ("Overview of quantization for efficient LLM fine-tuning.", "paper_level", [LORA],
   ["quantization","qlora","4-bit","memory"], "route_selection", "factual", {}),

 # ---------- Specific detail / chunk-level (15) ----------
 ("Which datasets are used to evaluate hallucination detection methods?", "chunk_level", [EVAL],
   ["dataset","truthfulqa","halueval","benchmark"], "full_text_evidence", "detail", {}),
 ("What evaluation metrics are reported for RAG systems?", "chunk_level", [RAG],
   ["metric","recall","precision","evaluation"], "full_text_evidence", "detail", {}),
 ("What benchmarks are used for tool-use agents?", "chunk_level", [AGENT],
   ["benchmark","tool","evaluation","agent"], "full_text_evidence", "detail", {}),
 ("How much does LoRA reduce GPU memory during fine-tuning?", "chunk_level", [LORA],
   ["gpu memory","memory","reduce","lora"], "full_text_evidence", "detail", {}),
 ("What datasets are used in parameter-efficient fine-tuning experiments?", "chunk_level", [LORA],
   ["dataset","glue","benchmark","evaluation"], "full_text_evidence", "detail", {}),
 ("What are reported limitations of LLM agent tool-use systems?", "chunk_level", [AGENT],
   ["limitation","tool","fail","challenge"], "full_text_evidence", "detail", {}),
 ("How is retrieval quality measured in RAG evaluation papers?", "chunk_level", [RAG],
   ["retrieval","recall","relevance","quality"], "full_text_evidence", "detail", {}),
 ("What computational complexity do attention mechanisms have?", "chunk_level", [TFMR],
   ["complexity","quadratic","sequence length","attention"], "full_text_evidence", "detail", {}),
 ("What ablation studies are reported for transformer components?", "chunk_level", [TFMR],
   ["ablation","component","head","layer"], "full_text_evidence", "detail", {}),
 ("Which reasoning benchmarks do fine-tuned adapters evaluate on?", "chunk_level", [LORA],
   ["reasoning","arithmetic","commonsense","benchmark"], "full_text_evidence", "detail", {}),
 ("What failure modes cause hallucinations according to the papers?", "chunk_level", [EVAL],
   ["hallucination","failure","cause","factual"], "full_text_evidence", "detail", {}),
 ("What memory or planning modules do autonomous agents use?", "chunk_level", [AGENT],
   ["memory","planning","module","reflection"], "full_text_evidence", "detail", {}),
 ("How do re-ranking methods improve retrieval results?", "chunk_level", [RAG,AGENT],
   ["re-rank","rerank","ranking","retrieval"], "full_text_evidence", "detail", {}),
 ("What training data scale is discussed for embodied agent planning?", "chunk_level", [AGENT],
   ["training data","few-shot","planning","embodied"], "full_text_evidence", "detail", {}),
 ("What quantization precision levels are evaluated for fine-tuning?", "chunk_level", [LORA],
   ["4-bit","8-bit","precision","quantization"], "full_text_evidence", "detail", {}),

 # ---------- Multi-hop / cross-topic (12) ----------
 ("Compare RAG and fine-tuning as ways to inject domain knowledge into LLMs.", "hybrid_both", [RAG,LORA],
   ["retrieval","fine-tuning","knowledge","adapt"], "cross_topic_comparison", "multihop", {}),
 ("How do retrieval and self-verification each reduce hallucinations?", "hybrid_both", [RAG,EVAL],
   ["retrieval","verification","hallucination","grounding"], "cross_topic_comparison", "multihop", {}),
 ("Contrast transformer attention efficiency with agent tool-calling overhead.", "hybrid_both", [TFMR,AGENT],
   ["attention","efficient","tool","agent"], "cross_topic_comparison", "multihop", {}),
 ("How do LoRA-style methods relate to transformer architecture components?", "hybrid_both", [LORA,TFMR],
   ["lora","transformer","weight","attention"], "cross_topic_comparison", "multihop", {}),
 ("Compare evaluation strategies for RAG systems versus autonomous agents.", "hybrid_both", [RAG,AGENT],
   ["evaluation","benchmark","retrieval","agent"], "cross_topic_comparison", "multihop", {}),
 ("How do hallucination benchmarks differ from agent task benchmarks?", "hybrid_both", [EVAL,AGENT],
   ["benchmark","hallucination","agent","task"], "cross_topic_comparison", "multihop", {}),
 ("Relate parameter-efficient fine-tuning to reducing hallucinations.", "hybrid_both", [LORA,EVAL],
   ["fine-tuning","hallucination","factual","adapt"], "cross_topic_comparison", "multihop", {}),
 ("Compare how RAG and long-context transformers handle large inputs.", "hybrid_both", [RAG,TFMR],
   ["retrieval","context","long","attention"], "cross_topic_comparison", "multihop", {}),
 ("How do multi-agent debate methods relate to hallucination reduction?", "hybrid_both", [AGENT,EVAL],
   ["multi-agent","debate","hallucination","verification"], "cross_topic_comparison", "multihop", {}),
 ("Contrast adapter tuning and prompt-based methods for task adaptation.", "hybrid_both", [LORA],
   ["adapter","prompt","tuning","task"], "cross_topic_comparison", "multihop", {}),
 ("How do retrieval quality and generation faithfulness interact in RAG?", "hybrid_both", [RAG,EVAL],
   ["retrieval","faithfulness","generation","grounding"], "cross_topic_comparison", "multihop", {}),
 ("Compare attention-efficiency techniques across transformer papers.", "hybrid_both", [TFMR],
   ["efficient","attention","sparse","linear"], "cross_topic_comparison", "multihop", {}),

 # ---------- Metadata filter (8) ----------
 ("Show the most-cited LoRA and PEFT papers.", "metadata_filter", [LORA],
   ["lora","peft","fine-tuning"], "metadata_filter", "metadata", {"year_min": None}),
 ("List recent RAG papers published after 2022.", "metadata_filter", [RAG],
   ["retrieval","rag","generation"], "metadata_filter", "metadata", {"year_min": 2023}),
 ("Show top-cited transformer architecture papers.", "metadata_filter", [TFMR],
   ["transformer","attention","architecture"], "metadata_filter", "metadata", {}),
 ("Find highly cited autonomous agent papers.", "metadata_filter", [AGENT],
   ["agent","autonomous","survey"], "metadata_filter", "metadata", {}),
 ("List hallucination evaluation papers from 2023 onward.", "metadata_filter", [EVAL],
   ["hallucination","evaluation","factual"], "metadata_filter", "metadata", {"year_min": 2023}),
 ("Show recent multi-agent system papers after 2023.", "metadata_filter", [AGENT],
   ["multi-agent","agent","system"], "metadata_filter", "metadata", {"year_min": 2024}),
 ("Find the most influential attention-mechanism papers.", "metadata_filter", [TFMR],
   ["attention","self-attention","transformer"], "metadata_filter", "metadata", {}),
 ("List recent parameter-efficient fine-tuning papers after 2022.", "metadata_filter", [LORA],
   ["parameter-efficient","fine-tuning","adapter"], "metadata_filter", "metadata", {"year_min": 2023}),

 # ---------- Paraphrase / vocab-mismatch (10) ----------
 ("How can I stop a chatbot from making things up?", "hybrid_both", [EVAL,RAG],
   ["hallucination","grounding","retrieval","factual"], "route_selection", "paraphrase", {}),
 ("What's the cheapest way to adapt a giant model to a new task?", "paper_level", [LORA],
   ["parameter-efficient","fine-tuning","lora","memory"], "route_selection", "paraphrase", {}),
 ("How do AI programs decide which external tool to call?", "hybrid_both", [AGENT],
   ["tool","tool-use","agent","selection"], "route_selection", "paraphrase", {}),
 ("How do models look up facts before answering?", "paper_level", [RAG],
   ["retrieval","retrieval-augmented","knowledge","generation"], "route_selection", "paraphrase", {}),
 ("Why do transformers get slow on very long documents?", "chunk_level", [TFMR],
   ["complexity","quadratic","sequence length","attention"], "full_text_evidence", "paraphrase", {}),
 ("How can software agents remember earlier steps?", "chunk_level", [AGENT],
   ["memory","reflection","history","agent"], "full_text_evidence", "paraphrase", {}),
 ("What tricks shrink the memory needed to train big models?", "chunk_level", [LORA],
   ["memory","low-rank","quantization","reduce"], "full_text_evidence", "paraphrase", {}),
 ("How do you check if a model's answer is actually true?", "paper_level", [EVAL],
   ["factual","truthfulness","evaluation","hallucination"], "route_selection", "paraphrase", {}),
 ("What lets a language model act on its own over many steps?", "paper_level", [AGENT],
   ["autonomous","planning","agent","multi-step"], "route_selection", "paraphrase", {}),
 ("How do systems combine searching and writing an answer?", "hybrid_both", [RAG],
   ["retrieval","generation","augmented","rag"], "route_selection", "paraphrase", {}),

 # ---------- Out-of-scope / refusal (12) - NO relevant IDs on purpose ----------
 ("What are the best convolutional architectures for image classification?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("How does reinforcement learning train robotic arms for grasping?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("Explain diffusion models for text-to-image generation.", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("What is the best approach for time-series forecasting of stock prices?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("How do graph neural networks work for molecular property prediction?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("What are speech recognition architectures for low-resource languages?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("How do recommender systems use collaborative filtering?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("What are the current world records in competitive Pokemon?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("How do I fine-tune a model to predict the weather next week?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("What are the health benefits of a Mediterranean diet?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("Which GPUs are cheapest to buy for gaming in 2025?", "refuse", [],
   [], "refusal", "out_of_scope", {}),
 ("How does federated learning preserve privacy on mobile devices?", "refuse", [],
   [], "refusal", "out_of_scope", {}),

 # ---------- Reading-path / recommendation (8) ----------
 ("Which LoRA and PEFT papers should I read first and why?", "paper_level", [LORA],
   ["lora","peft","parameter-efficient","fine-tuning"], "reading_path", "reading_path",
   {"acceptable_routes": ["hybrid_both"]}),
 ("What are the foundational papers to understand RAG?", "paper_level", [RAG],
   ["retrieval-augmented","rag","retrieval","survey"], "reading_path", "reading_path",
   {"acceptable_routes": ["hybrid_both"]}),
 ("Where should a beginner start with LLM agents?", "paper_level", [AGENT],
   ["agent","autonomous","survey","llm"], "reading_path", "reading_path",
   {"acceptable_routes": ["hybrid_both"]}),
 ("Which transformer papers are essential background reading?", "paper_level", [TFMR],
   ["transformer","attention","architecture"], "reading_path", "reading_path",
   {"acceptable_routes": ["hybrid_both"]}),
 ("What should I read to understand LLM hallucination evaluation?", "paper_level", [EVAL],
   ["hallucination","evaluation","factual","benchmark"], "reading_path", "reading_path",
   {"acceptable_routes": ["hybrid_both"]}),
 ("Recommend key surveys covering autonomous agents.", "paper_level", [AGENT],
   ["survey","agent","autonomous"], "reading_path", "reading_path",
   {"acceptable_routes": ["hybrid_both"]}),
 ("Which papers best introduce parameter-efficient adaptation?", "paper_level", [LORA],
   ["parameter-efficient","adapter","fine-tuning"], "reading_path", "reading_path",
   {"acceptable_routes": ["hybrid_both"]}),
 ("What are the most important RAG evaluation papers to study?", "paper_level", [RAG,EVAL],
   ["rag","evaluation","retrieval","metric"], "reading_path", "reading_path",
   {"acceptable_routes": ["hybrid_both"]}),
]

ALL_ROUTES = ["paper_level", "chunk_level", "hybrid_both", "metadata_filter"]

def build():
    out = []
    audit = []
    for query, route, topics, keywords, focus, category, extra in SPECS:
        tag = category  # my fine-grained taxonomy, kept for audit + rationale
        is_oos = (category == "out_of_scope")
        entry = {
            "query": query,
            "expected_route": "paper_level" if is_oos else route,
            "expected_topics": topics,
            "expected_keywords": keywords,
            "expected_relevant_ids": [],
            "category": "out_of_corpus" if is_oos else "single_turn",
            "evaluation_focus": "confidence_gate" if is_oos else focus,
            "rationale": f"[{tag}] {focus} behavior.",
        }
        if is_oos:
            entry["expected_confidence_decision"] = "insufficient_evidence"
            entry["acceptable_routes"] = ALL_ROUTES  # routing is not what we test here
            entry["expected_relevant_ids"] = []
            entry["rationale"] = ("[out_of_scope] Topic not covered by the corpus; the "
                                  "confidence gate should return insufficient_evidence and refuse.")
            audit.append((query, tag, "refuse", []))
            out.append(entry)
            continue

        if extra.get("acceptable_routes"):
            entry["acceptable_routes"] = extra["acceptable_routes"]

        ids = []
        if route == "metadata_filter":
            ids = pick_papers(keywords, topics, n=3, min_hits=1,
                              year_min=extra.get("year_min"), rank_by="citation")
        elif route == "paper_level" and tag == "reading_path":
            ids = pick_papers(keywords, topics, n=3, min_hits=1, rank_by="citation")
        elif route == "paper_level":
            ids = pick_papers(keywords, topics, n=3, min_hits=2,
                              year_min=extra.get("year_min"))
        elif route == "chunk_level":
            ids = pick_chunks(keywords, topics, n=3, min_hits=2)
        elif route == "hybrid_both":
            p = pick_papers(keywords, topics, n=2, min_hits=2)
            c = pick_chunks(keywords, topics, n=2, min_hits=2)
            ids = p + c
        entry["expected_relevant_ids"] = ids
        audit.append((query, tag, route, ids))
        out.append(entry)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    # ----- audit report -----
    print(f"WROTE {len(out)} queries -> {OUT.relative_to(ROOT)}\n")
    cat = collections.Counter(c for _, c, _, _ in audit)
    print("Category distribution:", dict(cat))
    empties = [q for q, c, r, ids in audit if r != "refuse" and not ids]
    print(f"\nNon-refusal queries with EMPTY ids (need attention): {len(empties)}")
    for q in empties: print("   !", q)
    print("\n--- SAMPLE GROUNDING (first 3 per category) ---")
    per = collections.Counter()
    for q, c, r, ids in audit:
        if r == "refuse": continue
        if per[c] >= 3: continue
        per[c] += 1
        print(f"\n[{c}/{r}] {q}")
        for i in ids:
            if i.startswith("chunk-"):
                ct = next((x for x in chunks if x['chunk_id'] == i), None)
                print(f"    {i}  ::  {(ct['title'] if ct else '?')[:70]}")
            else:
                print(f"    {i.split('/')[-1]}  ::  {PAPER_TITLE.get(i,'?')[:70]}")

if __name__ == "__main__":
    build()
