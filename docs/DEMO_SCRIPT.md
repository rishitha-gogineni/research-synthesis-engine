# Demo Script

This walkthrough is designed for a short project demo or interview screen share. It focuses on what makes the system stronger than a basic chatbot: route-aware retrieval, full-text evidence, confidence gating, contextual follow-ups, and inspectable research outputs.

## Before The Demo

Start the local services:

```bash
docker compose up -d qdrant
uvicorn api.main:app --reload
RSE_API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

Confirm the backend is healthy:

```bash
curl http://localhost:8000/health
```

Expected dependency state:

```text
qdrant: available
paper_collection: available
chunk_collection: available
```

Optional evaluation check:

```bash
python -m retrieval.evaluate --queries tests/fixtures/eval_queries.json
```

## Best Demo Questions

### 1. Broad Synthesis

```text
What are the main approaches for reducing hallucinations in LLMs?
```

Shows: fast-first research brief, confidence gate, themes, evidence matrix, and source inspection.

What to point out:

- The system retrieves evidence before answering.
- The first answer is grounded and concise, not just a paper list.
- The evidence matrix makes claims inspectable.

### 2. Cross-Topic Comparison

```text
Compare RAG and self-verification methods for reducing hallucinations.
```

Shows: `hybrid_both` routing and comparison-style synthesis.

What to point out:

- The router uses both broad paper retrieval and detailed chunk retrieval.
- The answer should explain tradeoffs, not only rank papers.
- Sources come from the indexed literature corpus.

### 3. Agentic RAG Question

```text
What is the difference between AI agents and RAG?
```

Shows: cross-topic retrieval between `AI Agents & Tool Use` and `Retrieval-Augmented Generation (RAG)`.

What to point out:

- RAG is mainly an evidence-grounding pattern.
- Agents add planning, tool use, and multi-step task execution.
- This question is a good way to explain the project itself.

### 4. Full-Text Evidence Question

```text
What evidence do the papers give about SelfCheckGPT for hallucination detection?
```

Shows: chunk-level retrieval over full-text papers.

What to point out:

- This should use detailed full-text snippets, not only abstracts.
- The source tab should show paper titles, chunk labels, and snippets.
- Full-text evidence is useful for methods, datasets, metrics, and limitations.

### 5. Contextual Follow-Up

First ask:

```text
Explain SelfCheckGPT for hallucination detection.
```

Then ask the follow-up:

```text
What are its evaluation methods?
```

Shows: context-aware query rewriting.

What to point out:

- The follow-up is rewritten into a standalone retrieval query.
- The system uses chat history for retrieval, not just generation.
- Diagnostics can show the original question and rewritten query.

### 6. Confidence Gate / Refusal

```text
What does the indexed corpus say about Kubernetes autoscaling policies?
```

Shows: out-of-corpus handling.

What to point out:

- The system should avoid unsupported synthesis when corpus evidence is weak.
- The confidence gate checks query support, not just vector score strength.
- This is a safety feature, not a failure.

### 7. Reading Path

```text
Which LoRA and PEFT papers should I read first and why?
```

Shows: staged reading recommendation.

What to point out:

- The output is not just top-cited papers.
- Papers are organized into a learning sequence.
- Each recommendation includes a reason grounded in retrieved evidence.

## UI Walkthrough

1. Open the Streamlit workspace.
2. Select a research area only if you want a narrower query.
3. Type or select a question.
4. Use `Preview route` to show the route without retrieval.
5. Use `Run analysis` to run retrieval, confidence checking, and synthesis.
6. Start with the direct answer and evidence matrix.
7. Open sources to show snippets and citation counts.
8. Generate reading path or open problems only when useful.
9. Turn on diagnostics when explaining route, rewritten query, timing, or confidence signals.

## One-Minute Architecture Explanation

```text
OpenAlex corpus + legal full-text PDFs
→ abstract-level paper index + full-text chunk index
→ query router
→ paper retrieval / chunk retrieval / metadata filter
→ reranking and citation-aware scoring
→ CRAG confidence gate
→ grounded brief, evidence matrix, reading path, open problems
→ FastAPI backend and Streamlit workspace
```

## Strong Closing Line

Research Synthesis Engine does not just search papers. It turns an indexed AI literature corpus into a confidence-gated analyst brief with inspectable evidence, source snippets, and a staged reading path.
