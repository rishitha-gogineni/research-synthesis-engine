# Evaluation Failure Analysis - v2 Fixture

This report analyzes the 82-query v2 evaluation fixture after the Matryoshka embedding fix, router phrase fixes, cross-encoder-unavailable fallback fix, and lightweight query expansion for colloquial research questions.

## Current Metrics

| Metric | Previous best | Current | Delta |
|---|---:|---:|---:|
| route_accuracy | 0.976 | 1.000 | +0.024 |
| hit@5 | 0.559 | 0.559 | +0.000 |
| hit@10 | 0.662 | 0.691 | +0.029 |
| recall@5 | 0.286 | 0.287 | +0.001 |
| recall@10 | 0.374 | 0.384 | +0.010 |
| mrr | 0.403 | 0.408 | +0.004 |

## Top-20 Diagnostic

The top-20 probe is not a product setting; it shows whether relevant evidence exists nearby but is being ranked below the visible cutoff.

| Metric | @5 | @10 | @20 |
|---|---:|---:|---:|
| Relevant-ID hit rate | 0.559 | 0.691 | 0.882 |
| True recall | 0.287 | 0.384 | 0.551 |

Interpretation: many relevant sources are present between ranks 11 and 20. That means the next improvement should focus on safer candidate reranking/promotion rather than new indexing.

## Remaining Failure Types

| Failure type | Count | Meaning | Best next fix |
|---|---:|---|---|
| partial_recall_top10 | 35 | Some expected sources appear in top 10, but not all. | Route-specific candidate promotion and better multi-source scoring. |
| cutoff_recoverable_top20 | 13 | No expected source appears in top 10, but at least one appears by top 20. | Fetch 20 internally and promote with stable, query-aware scoring. |
| deep_retrieval_miss | 8 | No expected source appears even by top 20. | Query expansion or manual label review; this is not solved by simple reranking. |

## Failures By Evaluation Focus

| Focus | Count |
|---|---:|
| route_selection | 19 |
| full_text_evidence | 17 |
| cross_topic_comparison | 12 |
| reading_path | 8 |

## Highest-Value Next Step

Do not make the golden set easier. The best next engineering fix is route-aware internal candidate promotion: retrieve a wider candidate set for paper/chunk routes, preserve the current top-k behavior for metadata paths, and promote candidates using stable signals that do not depend on batch-relative min-max scoring. The top-20 probe shows this has room to improve Hit@10 without changing the corpus.

## Sample Remaining Failures

| Type | Focus | Query | Hit@10 | Hit@20 |
|---|---|---|---:|---:|
| partial_recall_top10 | route_selection | What are the main approaches to retrieval-augmented generation? | 1/3 | 1/3 |
| partial_recall_top10 | route_selection | Give an overview of parameter-efficient fine-tuning methods. | 1/3 | 3/3 |
| cutoff_recoverable_top20 | route_selection | What are the key ideas behind the transformer architecture? | 0/3 | 3/3 |
| partial_recall_top10 | route_selection | How is hallucination in large language models defined and categorized? | 2/3 | 3/3 |
| partial_recall_top10 | route_selection | What methods exist for grounding LLM outputs in retrieved evidence? | 1/3 | 2/3 |
| partial_recall_top10 | route_selection | Overview of attention mechanism variants for efficiency. | 2/3 | 3/3 |
| partial_recall_top10 | route_selection | What are common tool-use frameworks for LLM agents? | 1/3 | 1/3 |
| partial_recall_top10 | route_selection | What is retrieval-augmented generation used for in knowledge-intensive tasks? | 1/3 | 1/3 |
| partial_recall_top10 | route_selection | How do LLMs perform planning for embodied or multi-step tasks? | 2/3 | 2/3 |
| partial_recall_top10 | route_selection | What benchmarks measure factual accuracy of language models? | 1/3 | 1/3 |
| partial_recall_top10 | route_selection | What are positional encoding methods in transformers? | 2/3 | 2/3 |
| partial_recall_top10 | route_selection | Overview of quantization for efficient LLM fine-tuning. | 2/3 | 2/3 |
| deep_retrieval_miss | full_text_evidence | Which datasets are used to evaluate hallucination detection methods? | 0/3 | 0/3 |
| partial_recall_top10 | full_text_evidence | What evaluation metrics are reported for RAG systems? | 1/3 | 2/3 |
| cutoff_recoverable_top20 | full_text_evidence | What benchmarks are used for tool-use agents? | 0/3 | 1/3 |
| partial_recall_top10 | full_text_evidence | How much does LoRA reduce GPU memory during fine-tuning? | 1/3 | 2/3 |
| cutoff_recoverable_top20 | full_text_evidence | What datasets are used in parameter-efficient fine-tuning experiments? | 0/3 | 1/3 |
| deep_retrieval_miss | full_text_evidence | How is retrieval quality measured in RAG evaluation papers? | 0/3 | 0/3 |
| partial_recall_top10 | full_text_evidence | What computational complexity do attention mechanisms have? | 1/3 | 1/3 |
| cutoff_recoverable_top20 | full_text_evidence | What ablation studies are reported for transformer components? | 0/3 | 1/3 |
