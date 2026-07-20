# Decision Log

This file records project decisions while the implementation is still fresh. Every major design choice should be tied to the project scope, data scale, demo goals, and measurable outcomes.

## 2026-07-10: Start With 5 Topics

We will start with a small, curated set of about 5 research topics instead of trying to ingest every possible area.

Reasoning:
- A focused topic set keeps ingestion, evaluation, and demo review manageable within the 20-25 day build window.
- Retrieval metrics such as Recall@K and MRR are easier to inspect honestly when the corpus is small enough to manually review.
- Five topics provide enough variety to demonstrate routing, comparison, and timeline behavior without turning Day 1 into a data collection project.
- This scope still supports the main technical story: hybrid retrieval, reranking, CRAG confidence checks, and grounded synthesis.

## 2026-07-10: Use Batch Ingestion Instead of Kafka

We will use Python batch ingestion scripts and local persisted records instead of Kafka or any event streaming system.

Reasoning:
- The initial dataset is small and finite: academic papers fetched per topic from Semantic Scholar.
- The system does not need continuous real-time event processing to satisfy the core product goal.
- Batch ingestion is easier to test, rerun, debug, and explain for this project size.
- Idempotent re-ingestion and upsert behavior provide the practical replayability story this project needs.
- Avoiding Kafka keeps the architecture honest; Kafka can be reserved for a separate project with a genuinely continuous data source.

## 2026-07-10: Use a Local Cross-Encoder Instead of Cohere Rerank

We will use an open-source local cross-encoder from `sentence-transformers` instead of Cohere Rerank.

Reasoning:
- A local reranker avoids extra API cost and rate limits.
- It keeps the project mostly free apart from OpenAI embedding and synthesis usage.
- It makes the ranking stage reproducible during tests and demos.
- It demonstrates a real two-stage retrieval architecture: first retrieve broadly with dense and sparse search, then rerank a smaller candidate set with a stronger local model.
- Cohere can be mentioned as an alternative, but the project does not need another paid external dependency to prove the technical point.

## 2026-07-10: Skip Papers Without Abstracts During Raw Ingestion

We will skip raw paper records that do not have both a title and a reconstructable abstract.

Reasoning:
- The abstract is the minimum text field needed for embedding, BM25 indexing, extraction, and grounded synthesis.
- Keeping abstract-free records would create noisy retrieval candidates and force special-case handling immediately in Days 3-5.
- The ingestion script reports how many records were skipped per topic so this filtering remains visible.
- This is a raw-ingestion quality filter, not an enrichment step; the saved `data/raw_papers.json` still contains ungenerated paper metadata from OpenAlex.

## 2026-07-10: Use These 5 Initial Topics

The Day 2 ingestion corpus will use these topics:
- Retrieval-Augmented Generation (RAG)
- Transformers / Attention Mechanisms
- LLM Evaluation & Hallucination Detection
- AI Agents & Tool Use
- Fine-tuning (LoRA / PEFT)

Reasoning:
- Together they cover retrieval, model architecture, evaluation, agentic systems, and adaptation techniques.
- They align tightly with the final demo story for a research synthesis engine.
- Each topic is broad enough to retrieve around 50 relevant papers while still being specific enough for manual review and evaluation.

## 2026-07-10: Switch Day 2 Ingestion From Semantic Scholar to OpenAlex

We will use OpenAlex for the Day 2 raw paper ingestion source instead of Semantic Scholar.

Reasoning:
- The first live Semantic Scholar request returned `429 Too Many Requests` before yielding even one paper.
- OpenAlex returned a valid paper search response immediately for the first RAG topic test.
- OpenAlex provides the fields this project needs: title, authors, reconstructed abstract, cited-by count, publication year, DOI, arXiv IDs, and source URLs.
- OpenAlex's abstract format requires a small reconstruction step from `abstract_inverted_index`, but that is simpler than fighting early ingestion rate limits.
- The script supports `OPENALEX_API_KEY` if we add one later, while still allowing public test calls when OpenAlex permits them.

## 2026-07-10: Use Curated Title Queries and Sort by Citation Count

We will fetch OpenAlex works with curated `title.search` query aliases per topic and `sort=cited_by_count:desc` by default.

Reasoning:
- The project should start from influential research papers, not just the first relevance-ranked search results.
- Sorting broad full-text search by citation count can pull in irrelevant but highly cited works, so the script searches titles with topic-specific aliases before ranking by citations.
- Citation count is an imperfect proxy for quality, but it is transparent, measurable, and already part of the project schema.
- Later retrieval will still use semantic relevance; this ingestion sort only controls which 50 papers per topic enter the initial corpus.
- The saved corpus remains auditable because every paper keeps its OpenAlex ID and citation count.

## 2026-07-11: Use gpt-4o-mini for Abstract Extraction

We will use `gpt-4o-mini` for Day 3 extraction from abstracts.

Reasoning:
- The task is structured extraction from short abstracts, so a small model is sufficient and cost-aware.
- The prompt explicitly requires `"not stated in abstract"` for missing datasets, results, or limitations to reduce hallucinated metadata.
- Every output is validated with Pydantic before being saved.
- Malformed or invalid outputs are retried once, then logged and skipped so one bad paper does not crash the batch.
- The script saves progress after each paper and skips already enriched records on rerun to avoid accidental duplicate API spend.

## 2026-07-11: Truncate OpenAI Embeddings From 3072 to 1024 Dimensions

We will create embeddings with `text-embedding-3-large` and store the first 1024 dimensions.

Reasoning:
- `text-embedding-3-large` returns high-quality 3072-dimensional embeddings.
- Matryoshka-style truncation lets us reduce storage and Qdrant memory while preserving a meaningful prefix representation.
- The 1024-dimensional vector is a practical size for the project corpus and local Qdrant.
- We will measure the retrieval tradeoff later instead of claiming an unverified accuracy-retention percentage.

## 2026-07-12: Change Final Output to a Research Analyst Brief

We will make the final product output a research analyst brief instead of a generic synthesis/matrix/timeline package.

Reasoning:
- A generic summary can feel like an ordinary RAG demo, while a research brief feels like a practical decision-support tool.
- Users should not need to know the exact papers in the corpus; the UI will show available research areas, suggested questions, and a free-text question box.
- The primary outputs will be a direct answer, research themes, an evidence matrix, a recommended reading path, and open problems.
- The timeline remains useful, but it becomes a secondary supporting view instead of the main product promise.
- This change does not require redoing Days 1-4 because the existing metadata fields support the stronger output format.

## 2026-07-12: Use Stable Qdrant Point IDs for Idempotent Upserts

We will derive Qdrant point IDs from each paper's source ID using UUIDv5.

Reasoning:
- Qdrant point IDs must be valid UUID strings or unsigned integers, while OpenAlex IDs are URLs.
- UUIDv5 gives a deterministic ID for each paper, so rerunning the index script updates existing points rather than creating duplicates.
- The original OpenAlex paper ID is still stored in the payload for traceability.
- This keeps local indexing replayable without introducing Kafka or another event replay system.

## 2026-07-13: Keep Missing Extraction Values Explicitly Abstract-Scoped

We will use `"not stated in abstract"` for missing `dataset_used`, `key_result`, and `limitations` fields instead of the shorter `"not specified"`.

Reasoning:
- The extraction source for Day 3 is only the title and abstract, not the full paper.
- `"not stated in abstract"` is more precise and makes it clear that the system is not claiming the full paper lacks that information.
- Survey-style papers naturally have more missing experiment fields because their abstracts summarize a research area rather than report one dataset, metric, or limitation.
- This wording makes the structured metadata easier to defend during demos and interviews.
- Existing enriched data does not need to be reprocessed just for this label; future extraction runs will use the clearer wording.

## 2026-07-13: Keep Generated Data Local and Keep Git Focused on Reproducible Code

We will keep large or regenerated artifacts such as `data/raw_papers.json`, `data/enriched_papers_final.json`, `data/embedded_papers.json`, `data/bm25_index.pkl`, and local Qdrant storage out of git.

Reasoning:
- The repository should remain lightweight and easy for reviewers to clone.
- The code, schemas, tests, documentation, and commands are the durable project assets.
- Data files can be regenerated from the ingestion pipeline when API keys are available.
- Keeping secrets and local artifacts out of git avoids accidental exposure of API keys or machine-specific state.
- The README documents the expected artifact names and current local counts so the pipeline remains understandable without committing generated data.

## 2026-07-13: Present the Public README as a Product-Style Project, Not a Day-by-Day Assignment

We will describe the public project status as phases, such as `Phase 1: Ingestion & Indexing`, instead of framing the README around day numbers.

Reasoning:
- The day-by-day plan is useful for execution, but a public GitHub README should read like an engineering project.
- Phase language makes the work easier for recruiters, interviewers, and collaborators to understand quickly.
- The detailed build plan still lives in `docs/research-synthesis-engine-build-plan.md` for planning traceability.
- The README should highlight architecture, data flow, validation, and next-phase work rather than internal scheduling.

## 2026-07-13: Expand Toward a Research Intelligence Product After the Core Plan

After the core day-by-day plan is complete, the project should expand from a retrieval pipeline into a research intelligence tool.

Reasoning:
- A simple RAG chatbot would undersell the work already done in ingestion, extraction, indexing, and evaluation.
- The stronger product direction is an evidence-grounded research synthesis assistant that can compare methods, surface datasets and limitations, recommend reading paths, and identify research gaps.
- This expansion builds naturally on the existing schema fields: methodology, dataset, key result, limitation, citation count, topic, and source IDs.
- The best demo output should be a research analyst brief with citations and evidence tables, not a generic paragraph summary.
- This path gives the project a clearer internship story across data engineering, LLM extraction, vector search, retrieval evaluation, backend APIs, and user-facing product design.

## 2026-07-13: Start Live Retrieval With a Hybrid Qdrant + BM25 Wrapper

We will expose Day 6 retrieval through a reusable Python module that accepts any free-text research question, embeds it, searches Qdrant and BM25, and returns merged candidate papers.

Reasoning:
- User questions should be dynamic; answers should not be hardcoded to sample prompts.
- Qdrant provides semantic matching, while BM25 preserves exact keyword and phrase matches for terms such as `hallucination`, `LoRA`, or `tool use`.
- Returning dense, sparse, and hybrid scores makes later debugging, reranking, and evaluation easier.
- The wrapper keeps the LLM out of the first retrieval step; generation will only see selected evidence papers instead of the full 250-paper corpus.
- This creates a clean foundation for the next phase: local cross-encoder reranking and evidence-backed answer generation.

## 2026-07-13: Add a Tool-Style Retrieval Interface Before API Work

We will wrap hybrid retrieval in `tools.research_retrieval` with Pydantic request and response schemas before building FastAPI or UI endpoints.

Reasoning:
- The retrieval layer needs a stable contract before it is exposed through an API, agent, or dashboard.
- Pydantic schemas make the tool output predictable for later synthesis, reranking, and UI rendering.
- Tests can mock the retrieval function and validate the tool contract without spending OpenAI credits or requiring Qdrant to be running.
- A JSON CLI gives a simple manual debugging path while the backend and UI are still under construction.
- Keeping the wrapper separate from `retrieval.hybrid_search` preserves a clean boundary between retrieval mechanics and tool-facing behavior.

## 2026-07-13: Build Full-Text Retrieval From a Legal Open-Access Subset

We will not force full-text coverage for all 250 papers. Instead, we will build a 100-150 paper full-text subset from papers with legal open PDFs.

Reasoning:
- The existing 250-paper abstract index remains useful for broad discovery across all topics.
- Full-text retrieval should only use legal open sources such as arXiv and OpenAlex open-access PDF locations.
- The first source discovery run found 173 available full-text PDF sources: 49 arXiv sources and 124 OpenAlex open-access PDF sources.
- This is enough to select a high-citation, topic-balanced full-text subset without depending on closed-access papers.
- The next full-text steps should download, extract, chunk, and embed only the available subset, while unavailable papers stay abstract-only.

## 2026-07-13: Select 125 Full-Text Papers With Topic Balance

We will start full-text extraction with 125 selected papers: the top 25 available full-text sources per topic, ranked by citation count.

Reasoning:
- The user target is 100-150 full-text papers, and 125 lands in the middle of that range.
- A fixed 25-per-topic selection prevents the full-text index from being dominated by whichever topic has the most open PDFs.
- Citation count remains a transparent ranking signal for selecting influential papers from the 173 available sources.
- The selected subset contains 35 arXiv PDFs and 90 OpenAlex open-access PDFs.
- This subset should be downloaded and extracted first; the remaining available sources can be added later if the extraction quality is good.

## 2026-07-13: Extract as Many Legal Full-Text PDFs as Practical

After the initial 125-paper subset yielded 92 successful extractions, we expanded to all 173 discovered legal PDF sources and extracted 131 full-text papers.

Reasoning:
- The user wanted the strongest possible full-text set, not a fixed 100-150 cap.
- Attempting all discovered legal PDF sources maximizes evidence coverage while keeping closed or blocked papers out of the full-text index.
- The final local extraction set contains 131 successful papers, 2533 pages, and about 10.3M text characters.
- The 42 failures were mostly publisher-side access blocks such as `403 Forbidden`; those papers remain usable in the abstract-level index.
- The next step should chunk and embed the 131 successful full-text papers into a separate chunk-level retrieval collection.

## 2026-07-13: Store Full-Text Retrieval as a Separate Chunk Collection

We will store full-text evidence in a separate Qdrant collection named `research_paper_chunks` instead of mixing chunks into the paper-level `research_papers` collection.

Reasoning:
- Paper-level retrieval and chunk-level retrieval answer different kinds of questions.
- The `research_papers` collection remains one vector per paper for broad discovery across all 250 papers.
- The `research_paper_chunks` collection stores 4,170 vectors from 131 extracted full-text papers for detailed evidence questions.
- Chunk payloads keep paper metadata, section hints, chunk text, PDF source, citation count, and embedding metadata for traceability.
- Keeping collections separate makes routing easier: broad questions use paper retrieval, detailed dataset/method/result/limitation questions use chunk retrieval, and comparison questions can use both.

## 2026-07-14: Use a Rule-Based Query Router Before Agent Orchestration

We will route user questions with a deterministic rule-based router before adding any LLM-based planning.

Reasoning:
- The system currently has two real retrieval granularities: paper-level records and full-text chunks.
- A lightweight router is enough to decide whether a query needs broad paper discovery, detailed full-text evidence, both, or metadata filtering.
- Rule-based routing is easy to test, inspect, and explain during demos.
- Ambiguous or low-confidence queries default to `hybrid_both` so the system gathers both broad context and detailed evidence instead of guessing one narrow path.
- This keeps Day 11 focused on routing without adding new indexes, new retrieval layers, or live web tools.

## 2026-07-14: Keep `hybrid_both` Results as Separate Paper and Chunk Sets

When a query routes to `hybrid_both`, the retrieval service will return paper-level results and chunk-level results as two separate result sets.

Reasoning:
- Paper records and full-text chunks are different granularities and should not be forced into one ranked list before reranking/context assembly.
- Paper-level results are best for broad coverage, topic framing, citation context, and candidate paper selection.
- Chunk-level results are best for specific evidence such as datasets, metrics, methods, results, and limitations.
- Keeping them separate preserves score interpretability because paper hybrid scores and chunk vector scores are not directly comparable yet.
- Day 12+ can assemble context by linking chunks back to their parent papers, then Day 13-14 can rerank and blend scores with a clearer candidate contract.

## 2026-07-14: Rerank Candidates With a Local Cross-Encoder and Blend Citation Signal Transparently

We will use a local `sentence-transformers` cross-encoder to rerank retrieved candidates, then apply citation-aware blended scoring with a visible score breakdown.

Reasoning:
- Dense and BM25 retrieval are good first-stage retrieval methods, but a cross-encoder can compare the query and candidate text more directly.
- The cross-encoder runs locally, avoiding another paid reranking API and keeping tests mockable.
- Rerank scores are normalized within the candidate set so they can be combined with other signals.
- Citation count is log-normalized with `log1p(citation_count)` so highly cited papers help ranking without completely dominating relevance.
- The default score is `0.75 * rerank_score + 0.25 * normalized_citation_score`, and each output keeps `score_breakdown` for debugging and demo transparency.
- Paper-level and chunk-level candidate sets should be reranked and blended separately until the unified retrieval service defines a common context assembly contract.

## 2026-07-15: Use a Unified Retrieval Service as the Query-Time Entry Point

We will use `retrieval.unified_search` as the main query-time retrieval entry point before answer generation.

Reasoning:
- The project now has multiple retrieval paths, so callers need one stable interface instead of manually deciding which module to call.
- The unified service executes the Day 11 router decision and returns a structured response containing the route, route reason, paper results, and chunk results.
- `paper_level` uses the existing Qdrant + BM25 hybrid paper retriever.
- `chunk_level` searches the `research_paper_chunks` Qdrant collection for detailed full-text evidence.
- `hybrid_both` executes both retrieval paths but keeps paper and chunk results as separate ranked lists.
- `metadata_filter` uses local paper metadata from the BM25 artifact, so citation/year/topic style questions do not require OpenAI or Qdrant.
- Reranking and citation-aware blended scoring are applied within each result set, preserving score interpretability across different granularities.

## 2026-07-15: Treat `expected_relevant_ids` as Optional Partial Labels in Evaluation

We will include `expected_relevant_ids` in the retrieval evaluation schema, but it starts as an optional partial label field that defaults to an empty list.

Reasoning:
- Exact relevant paper/chunk IDs are the most rigorous retrieval labels, but they require manual inspection to avoid noisy or fake ground truth.
- The first evaluation set can still measure route accuracy, topic hit rate, and keyword presence across all queries while exact ID labels are filled in gradually.
- Recall@5, Recall@10, and MRR are computed only over queries with non-empty `expected_relevant_ids`.
- Queries with empty `expected_relevant_ids` are not counted as retrieval failures, because that would silently distort the rigorous metrics.
- CLI output reports both `queries_with_relevant_ids` and `queries_topic_keyword_only` so reviewers can see which metrics are strict ID-based scores and which are broader sanity checks.
- This is intentional incremental rigor: start transparent, label more queries as results are manually inspected, and keep the metric denominator honest.

## 2026-07-17: Use Basic CRAG Before Synthesis and Defer Advanced Agentic RAG Extensions

We will add a basic CRAG-style confidence assessment before answer generation, using the unified retrieval response as input.

Reasoning:
- The first synthesis guardrail should be deterministic, inspectable, and testable before adding multi-round agentic loops.
- Confidence is based on visible retrieval signals: top score, route confidence, result count, score consistency, topic agreement, and paper/chunk agreement for `hybrid_both`.
- The guardrail can allow synthesis, broaden retrieval, ask a clarifying question, or state insufficient evidence.
- This prevents weak retrieval from flowing directly into generated research briefs.
- MA-RAG style multi-round refinement and KidnapRAG-style adversarial defenses remain valuable advanced extensions, but they are intentionally deferred until the initial end-to-end research synthesis system is complete.

## 2026-07-17: Gate Research Brief Generation on Retrieval Confidence

We will generate research briefs only after the CRAG confidence assessment returns `sufficient_evidence`.

Reasoning:
- The brief is user-facing, so weak retrieval should not be converted into confident prose.
- The generator prompt is restricted to retrieved sources and requires source IDs, keeping claims tied to evidence.
- Low-confidence responses return a guarded `skipped_low_confidence` brief with the recommended next action instead of calling the LLM.
- This keeps synthesis cost-aware because failed retrieval does not spend generation tokens.
- Tests mock the generator so the behavior is validated without live API calls.

## 2026-07-17: Build the Evidence Matrix Deterministically From Retrieved Evidence

We will build the first evidence matrix directly from retrieval outputs rather than asking an LLM to invent the table structure.

Reasoning:
- Retrieval results already contain the fields needed for matrix rows: title, topic, source IDs, methodology, dataset, key result, limitation, snippets, and scores.
- Deterministic matrix construction makes the output easier to test and debug.
- Missing values are labeled as `not stated in retrieved evidence`, which is more precise than implying the field is globally unknown.
- Evidence strength is derived from retrieval/rerank/blended scores, so reviewers can inspect why a source appears strong or weak.
- Later UI work can render the same matrix as JSON or Markdown without changing retrieval semantics.

## 2026-07-18: Build Reading Paths With Deterministic Selection and Grounded Explanations

We will select reading-path candidates deterministically before asking the LLM to explain the sequence.

Reasoning:
- Paper recommendation should not be delegated entirely to generation because the system must avoid recommending papers outside the retrieved corpus.
- Candidate selection combines retrieval score, citation count, publication year, evidence coverage, methodology diversity, and paper/chunk support.
- Citation count is useful for identifying foundational work, but it is not the only ranking signal; recent highly relevant papers are preserved for later stages.
- The LLM receives only validated candidate IDs and source IDs, then writes reasons, focus points, prerequisites, and transitions.
- Every generated paper ID and source ID is validated against retrieved evidence before the reading path is accepted.

## 2026-07-18: Derive Open Problems Only From Retrieved Limitation Evidence

We will generate open-problems reports only from retrieved limitations, future-work/discussion chunks, evaluation gaps, conflicts, and corpus limitations.

Reasoning:
- Open problems can easily become generic, so unsupported problems are rejected during validation.
- Evidence strength is based on the number of distinct supporting papers and sources, not citation count alone.
- One-paper evidence is never labeled `strong`; it is presented as weak or moderate depending on source support.
- If retrieval confidence is low, or if retrieved evidence contains no limitation/future-work signals, the report returns a guarded empty result instead of speculating.
- The combined Day 19 service reuses one `UnifiedSearchResponse` and one `ConfidenceAssessment` for reading paths and open-problems reports, avoiding duplicate retrieval calls.

## 2026-07-18: Keep FastAPI Thin Over Existing Retrieval and Agent Services

We will expose the completed pipeline through `api.main` without moving business logic into API route handlers.

Reasoning:
- Retrieval, confidence, synthesis, evidence matrix, reading path, and open-problems logic already have tested module boundaries.
- The API should validate requests, orchestrate existing services, shape responses, and map errors to HTTP responses.
- `/guidance` runs unified retrieval once and confidence assessment once, then reuses those objects for all downstream outputs to avoid duplicated retrieval and inconsistent evidence.
- Endpoint tests monkeypatch service calls so they do not call OpenAI, Qdrant, OpenAlex, or the local cross-encoder.
- This keeps the backend ready for Streamlit while preserving the same service contracts used by CLI tests.

## 2026-07-20: Polish the API In Place Before Building the Streamlit UI

We will harden `api.main` without splitting the backend into a larger route/dependency package yet.

Reasoning:
- The Day 20 backend already works, so Day 20.5 should reduce UI risk without rewriting stable code.
- `question` is the canonical public request field, while `query` remains accepted as a backward-compatible alias for existing tests and callers.
- `/route` gives the UI a cheap preview/debug path without running retrieval.
- Request IDs, structured errors, debug controls, CORS, and health checks make the backend easier to inspect during demos without adding authentication or external infrastructure.
- Filters are currently applied after retrieval and return an explicit warning; this is honest for the first UI version and avoids pretending retrieval is already filter-aware.
- Detailed debug signals and timing metrics are returned only when `include_debug=true` so normal UI responses stay focused.
