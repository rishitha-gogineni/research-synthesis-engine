# Research Synthesis Engine

Research Synthesis Engine is a grounded research assistant for asking questions across a curated corpus of 250 AI research papers. It combines paper-level metadata search, full-text passage retrieval, hybrid dense and BM25 search, intent-aware routing, confidence checks, and citation-aware synthesis.

[Live Streamlit app](https://research-synthesis-engine-mhupghrfkhudtzy4uvqzq6.streamlit.app/) | [Live API](https://research-synthesis-engine-api.onrender.com)

The live services run on Render, Streamlit Community Cloud, and Qdrant Cloud. Free-tier services may take up to 60 seconds to wake after inactivity.

## Why this project

Research questions often require more than semantic similarity. A useful system must find the right paper or passage, preserve enough context, identify when the corpus is insufficient, and show where the answer came from. This project treats retrieval quality, refusal behavior, and evidence attribution as first-class engineering concerns.

## Core capabilities

- Routes questions to paper-level, full-text, hybrid, or metadata retrieval.
- Combines OpenAI dense embeddings with BM25 keyword search.
- Retrieves section-aware passages with paper, section, and page metadata.
- Applies route-aware promotion and optional cross-encoder reranking.
- Uses a confidence gate to decline unsupported questions instead of guessing.
- Generates grounded research briefs, evidence matrices, reading paths, and open-problem summaries.
- Provides an optional agentic route with Arxiv, Semantic Scholar, Tavily, and MCP-backed research tools.
- Records request IDs, tool calls, evidence, citations, confidence decisions, latency, and token usage.

## Example questions

- What datasets are used to evaluate hallucination detection methods?
- Compare retrieval-augmented generation with fine-tuning for adding domain knowledge.
- Show highly cited AI agent survey papers published after 2023.
- How much does LoRA reduce GPU memory during fine-tuning?

For questions outside the indexed corpus, the confidence layer returns a refusal or explicitly uses an enabled external source. It does not silently fill gaps with unsupported claims.

## Evaluation

The canonical benchmark contains 100 queries:

- 95 queries with labeled relevant paper or passage IDs
- 5 out-of-corpus confidence checks
- factual, methodology, results, comparison, metadata, reading-path, and refusal cases

Retrieval results with the fixed production configuration:

| Metric | Result |
| --- | ---: |
| Route accuracy | 1.00 |
| Keyword Hit@10 | 0.967 |
| Relevant-ID Hit@10 | 0.705 |
| Recall@10 | 0.573 |
| MRR | 0.459 |
| Confidence decision accuracy | 1.00 |

The optional 46-case agentic benchmark measures planning, route selection, tool plans, external-source coverage, answer or refusal behavior, and citation support. The latest recorded run reported 1.00 route and tool-plan accuracy, 1.00 external-source coverage across 15 cases, 0.978 answer or refusal accuracy, 0.872 citation validity, and 1.00 citation coverage. Provider failures are surfaced as warnings and are not hidden from the metrics.

The repository includes 422 automated tests with external services mocked for deterministic regression testing.

## Architecture

~~~mermaid
flowchart LR
    User --> Streamlit
    Streamlit --> FastAPI
    FastAPI --> Router
    Router --> PaperSearch[Paper search]
    Router --> ChunkSearch[Full-text search]
    Router --> HybridSearch[Paper and passage search]
    Router --> Metadata[Metadata filters]
    PaperSearch --> Qdrant
    ChunkSearch --> Qdrant
    PaperSearch --> BM25
    ChunkSearch --> BM25
    Qdrant --> Confidence[Confidence gate]
    BM25 --> Confidence
    Confidence --> Synthesis[Grounded synthesis]
    Confidence --> Refusal[Refusal with evidence status]
    Synthesis --> Citations[Citations and research brief]
~~~

## Data and retrieval design

- 250 paper records are indexed for metadata and abstract retrieval.
- 152 papers have extracted full text.
- 4,909 full-text passages are indexed with section and page metadata.
- Paper and passage retrieval use OpenAI text-embedding-3-large embeddings reduced to 1,024 dimensions and BM25.
- Retrieval candidates are fused, promoted using query intent and citation signals, and optionally reranked.
- The canonical vector store is Qdrant. BM25 indexes are local artifacts under data/.

## Technology

| Area | Tools |
| --- | --- |
| Backend | Python, FastAPI, Pydantic |
| Retrieval | Qdrant, BM25, OpenAI embeddings, optional cross-encoder |
| Synthesis | OpenAI GPT-4o-mini, confidence gate, query rewriting |
| External research | Arxiv, Semantic Scholar, Tavily, MCP tools |
| Data pipeline | OpenAlex, PyMuPDF, section-aware chunking |
| Frontend | Streamlit |
| Deployment | Render, Streamlit Community Cloud, Qdrant Cloud |
| Testing | Pytest with mocked external dependencies |

## Local setup

Requirements: Python 3.11+, Docker, and access to an OpenAI-compatible API key. Qdrant Cloud can be used instead of local Qdrant.

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
~~~

Set the required values in .env:

~~~text
OPENAI_API_KEY=
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
RSE_APPLY_RERANKING=false
TAVILY_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
~~~

Start the local services:

~~~bash
docker compose up -d qdrant
uvicorn api.main:app --reload
streamlit run ui/streamlit_app.py
~~~

Run the regression suite:

~~~bash
python -m pytest -q
~~~

Run the canonical retrieval benchmark:

~~~bash
python -m retrieval.evaluate \
  --queries tests/fixtures/eval_queries_100_chunk_grounded.json \
  --qdrant-url http://localhost:6333
~~~

Validate agentic planning and recorded responses:

~~~bash
python scripts/evaluate_agentic.py
~~~

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | /health | Service health |
| GET | /corpus/stats | Corpus and index statistics |
| POST | /route | Query route preview |
| POST | /retrieve | Retrieval results |
| POST | /confidence | Evidence confidence decision |
| POST | /brief | Grounded research brief |
| POST | /agentic/research | Corpus, live, or hybrid research workflow |

Example request:

~~~bash
curl -X POST http://localhost:8000/agentic/research \
  -H 'Content-Type: application/json' \
  -d '{"question":"What datasets are used for hallucination detection?","top_k":8,"max_tool_calls":3}'
~~~

## Screenshots

![Query workspace](docs/images/query-workspace-live.png)

![Research brief](docs/images/research-brief-live.png)

![Evidence matrix](docs/images/evidence-matrix-live.png)

## Repository layout

~~~text
ingestion/     Paper collection, extraction, and embeddings
full_text/     PDF extraction, section-aware chunking, and passage indexing
retrieval/     Routing, hybrid search, ranking, confidence, and evaluation
agent/         Research outputs and evidence-grounded synthesis
agentic/       Planner, external tools, tool execution, and traces
api/           FastAPI service
ui/            Streamlit application
mcp_servers/   MCP research tools
tests/         Automated regression tests
docs/          Evaluation methodology and design decisions
~~~

## Known limitations

- Some relevant passages still rank below the top-10 cutoff.
- Multi-paper comparison questions are harder than single-paper factual questions.
- The corpus is curated and does not update automatically.
- Synthesis quality depends on the configured language model and provider availability.
- Free-tier deployments have cold starts.

## Deployment

The API is containerized with api/Dockerfile and deployed on Render. The Streamlit application calls the API through RSE_API_URL. Qdrant Cloud stores the production paper and passage collections. Keep secrets in environment variables and never commit .env.
