# Evaluation Strategy

The evaluation suite is designed to check whether the system retrieves the right evidence, routes questions to the right retrieval layer, rewrites contextual follow-ups, and refuses weak or out-of-corpus questions when evidence is insufficient.

## Fixture Summary

The main fixture is `tests/fixtures/eval_queries.json`.

```text
queries: 50
queries_with_relevant_ids: 36
single_turn_queries: 39
multi_turn_queries: 5
out_of_corpus_queries: 4
weak_evidence_queries: 2
```

## Coverage By Focus

| Evaluation Focus | Query Count | Purpose |
| --- | ---: | --- |
| `full_text_evidence` | 19 | Checks dataset, metric, method, result, and limitation questions that should retrieve chunks. |
| `cross_topic_comparison` | 7 | Checks questions that need evidence across topics or retrieval granularities. |
| `confidence_gate` | 6 | Checks out-of-corpus or under-specified queries that should not produce unsupported answers. |
| `metadata_filter` | 6 | Checks top-cited and year-filtered questions. |
| `contextual_rewrite` | 5 | Checks follow-up questions that require chat history to become standalone queries. |
| `route_selection` | 6 | Checks broad overview questions. |
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
queries: 50
queries_with_relevant_ids: 36
queries_topic_keyword_only: 14
multi_turn_queries: 5
out_of_corpus_queries: 4
evaluation_focus_counts: confidence_gate=6, contextual_rewrite=5, cross_topic_comparison=7, full_text_evidence=19, metadata_filter=6, reading_path=1, route_selection=6
route_accuracy: 1.00
rewrite_keyword_hit_rate: 1.00 (contextual subset, n=5)
confidence_decision_accuracy: 1.00 (labeled confidence subset, n=6)
crag_fallback_success_rate: 1.00 (expected fallback subset, n=6)
topic_hit_rate@5: 1.00 (sanity check, n=44)
keyword_hit_rate@5: 0.92 (sanity check, n=48)
recall@5 (labeled subset, n=36): 0.94
topic_hit_rate@10: 1.00 (sanity check, n=44)
keyword_hit_rate@10: 0.94 (sanity check, n=48)
recall@10 (labeled subset, n=36): 1.00
mrr (labeled subset, n=36): 0.86
```

The latest run uses the 50-question golden fixture. It reflects three evaluation hardening changes: route labels support explicit acceptable alternatives, exact-ID labels include multiple manually inspected relevant papers/chunks when appropriate, and the confidence gate filters generic off-topic matches more aggressively.

## Labeling Policy

Exact relevant IDs are added only when a paper or chunk can be identified from local artifacts. Some queries also declare `acceptable_routes` when a broader route is valid, such as `hybrid_both` for a detailed chunk question that benefits from paper context. Unlabeled queries are kept for route, topic, keyword, rewrite, and confidence behavior checks; they are not counted as Recall/MRR failures.

The labeled set should grow over time as more demo questions are manually inspected.
