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

