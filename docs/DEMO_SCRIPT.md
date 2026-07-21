# Demo Script

This walkthrough is designed for a short project demo or interview screen share. It focuses on the parts that make the system more than a basic chatbot: route-aware retrieval, confidence gating, evidence inspection, and reading-path generation.

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

## Three Strong Questions

### 1. Hallucination Reduction

```text
What are the main approaches for reducing hallucinations in LLMs?
```

Use this question to show the full analyst brief: direct answer, themes, evidence matrix, top supporting evidence, and open problems.

What to point out:

- The system does not answer directly from memory.
- It retrieves from the indexed corpus first.
- The confidence gate determines whether a grounded answer is shown.
- The evidence matrix makes claims inspectable against sources.

### 2. RAG Versus Self-Verification

```text
Compare RAG and self-verification methods for reducing hallucinations.
```

Use this question to show route-aware retrieval and comparison-style synthesis.

What to point out:

- The router should prefer a broader route because this is a comparison question.
- Paper-level and chunk-level evidence can both be useful.
- The answer should separate tradeoffs rather than listing papers.

### 3. Reading Path

```text
Which LoRA and PEFT papers should I read first?
```

Use this question to show the staged reading path.

What to point out:

- The output is not just top-cited papers.
- Papers are staged into a learning sequence.
- Each recommendation includes a reason to read it.

## UI Walkthrough

1. Open the Streamlit workspace.
2. In the sidebar, select optional research areas, year range, top K, full-text-only mode, or diagnostics.
3. Pick a suggested question or type a custom research question.
4. Use `Preview route` to show the route without retrieval.
5. Use `Run analysis` to run route preview, retrieval, confidence check, and synthesis.
6. On the results page, explain the evidence gate first.
7. Read the direct answer only if the gate passed.
8. Open the evidence matrix and source tabs to show how claims can be inspected.

## One-Minute Architecture Explanation

```text
OpenAlex papers + full-text chunks
→ OpenAI embeddings and BM25 index
→ Qdrant paper and chunk collections
→ rule-based query router
→ unified retrieval and reranking
→ confidence gate
→ analyst brief, evidence matrix, reading path, open problems
→ FastAPI backend and Streamlit workspace
```

## If Evidence Is Weak

If the evidence gate does not pass, the UI intentionally avoids showing a direct answer. In a demo, treat this as a strength: the system is designed to say when the indexed corpus is not strong enough, while still letting the user inspect retrieved sources.
