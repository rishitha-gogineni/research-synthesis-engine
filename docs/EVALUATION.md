# Evaluation Strategy

The evaluation suite is designed to check whether the system retrieves the right evidence, routes questions to the right retrieval layer, rewrites contextual follow-ups, and refuses weak or out-of-corpus questions when evidence is insufficient.

## Fixture Summary

The main fixture is `tests/fixtures/eval_queries.json`.

```text
queries: 35
queries_with_relevant_ids: 22
single_turn_queries: 26
multi_turn_queries: 4
out_of_corpus_queries: 3
weak_evidence_queries: 2
```

## Coverage By Focus

| Evaluation Focus | Query Count | Purpose |
| --- | ---: | --- |
| `full_text_evidence` | 12 | Checks dataset, metric, method, result, and limitation questions that should retrieve chunks. |
| `cross_topic_comparison` | 6 | Checks questions that need evidence across topics or retrieval granularities. |
| `confidence_gate` | 5 | Checks out-of-corpus or under-specified queries that should not produce unsupported answers. |
| `metadata_filter` | 4 | Checks top-cited and year-filtered questions. |
| `contextual_rewrite` | 4 | Checks follow-up questions that require chat history to become standalone queries. |
| `route_selection` | 3 | Checks broad overview questions. |
| `reading_path` | 1 | Checks reading recommendation behavior. |

## Metrics Reported

`retrieval.evaluate` reports two classes of metrics.

Rigorous labeled-subset metrics:

- Recall@5
- Recall@10
- MRR

These are computed only for queries where `expected_relevant_ids` is non-empty.

Sanity and behavior metrics:

- route accuracy
- topic hit rate
- keyword hit rate
- rewrite keyword hit rate
- confidence decision accuracy
- CRAG fallback success rate

Topic and keyword checks are useful for broad coverage, but they are intentionally treated as looser checks than exact-ID Recall/MRR.

## Running Evaluation

Start Qdrant first:

```bash
docker compose up -d qdrant
```

Then run:

```bash
python -m retrieval.evaluate --queries tests/fixtures/eval_queries.json
```

Machine-readable output:

```bash
python -m retrieval.evaluate --queries tests/fixtures/eval_queries.json --json
```

## Latest Local Run

Run date: 2026-07-24, with local Qdrant collections available.

```text
queries: 35
queries_with_relevant_ids: 22
queries_topic_keyword_only: 13
multi_turn_queries: 4
out_of_corpus_queries: 3
evaluation_focus_counts: confidence_gate=5, contextual_rewrite=4, cross_topic_comparison=6, full_text_evidence=12, metadata_filter=4, reading_path=1, route_selection=3
route_accuracy: 0.71
rewrite_keyword_hit_rate: 1.00 (contextual subset, n=4)
confidence_decision_accuracy: 0.80 (labeled confidence subset, n=5)
crag_fallback_success_rate: 0.80 (expected fallback subset, n=5)
topic_hit_rate@5: 1.00 (sanity check, n=30)
keyword_hit_rate@5: 0.91 (sanity check, n=33)
recall@5 (labeled subset, n=22): 0.64
topic_hit_rate@10: 1.00 (sanity check, n=30)
keyword_hit_rate@10: 0.94 (sanity check, n=33)
recall@10 (labeled subset, n=22): 0.73
mrr (labeled subset, n=22): 0.57
```

The confidence-gate metric improved after adding query-support scoring to the CRAG guardrail. Recall/MRR also improved after counting full-text chunks as relevant when either their exact chunk ID or parent paper ID matches a labeled expected ID.

## Labeling Policy

Exact relevant IDs are added only when a paper or chunk can be identified from local artifacts. Unlabeled queries are kept for route, topic, keyword, rewrite, and confidence behavior checks; they are not counted as Recall/MRR failures.

The labeled set should grow over time as more demo questions are manually inspected.
