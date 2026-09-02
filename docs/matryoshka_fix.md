# Matryoshka Embedding and Router Fix Evaluation

> Historical record: `tests/fixtures/eval_queries_v2.json` has since been superseded and no longer exists. Numbers below are preserved for audit trail, not reproducible as-is.

## Commands

Baseline command saved from Task 3:

```bash
python -m retrieval.evaluate --queries tests/fixtures/eval_queries_v2.json --json
```

After-run output:

```bash
python -m retrieval.evaluate --queries tests/fixtures/eval_queries_v2.json --json > eval_results/matryoshka_after.json
```

## Metric Comparison

The baseline file was produced before the two router regression queries were added, so it contains 80 queries. The after-run contains 82 queries. Retrieval metrics are computed over the labeled subset reported by the evaluator.

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Queries | 80 | 82 | +2 |
| Paper Recall@10 | 0.450 | 0.463 | +0.013 |
| Chunk Recall@10 | 0.182 | 0.182 | +0.000 |
| Aggregate Recall@10 | 0.364 | 0.374 | +0.010 |
| MRR | 0.366 | 0.379 | +0.013 |
| Router accuracy | 0.575 | 0.976 | +0.401 |
| P50 end-to-end latency | Not captured by current runner | Not captured by current runner | N/A |
| P95 end-to-end latency | Not captured by current runner | Not captured by current runner | N/A |
| Cost per query | Not captured by current runner | Not captured by current runner | N/A |

## Router Regression Queries

The two regression queries were added after the baseline, so there is no before value for this exact subset.

| Query | Expected route | Actual route after | Result |
|---|---|---|---|
| What is the difference between AI agents and RAG? | hybrid_both | hybrid_both | pass |
| How does RAG differ from fine-tuning? | hybrid_both | hybrid_both | pass |

Regression-query router accuracy after: **1.000** (2/2).

## Interpretation

The server-side Matryoshka change helped slightly and did not show an aggregate retrieval regression. It also fixes the correctness issue: stored 1024-dimensional vectors now come directly from the embedding API and have L2 norms near 1.0, instead of being locally sliced 3072-dimensional vectors with lower norms.

The router phrase additions and precedence fix also helped. Direct calls to `route_query()` classify both regression examples as `hybrid_both`, and the full retrieval pipeline now preserves that route instead of overriding "difference between" as a metadata/range query.

Latency and cost-per-query were requested for this report, but the current evaluation runner does not capture timing or token/cost telemetry. Those fields should be added to the evaluator before publishing latency or cost numbers from this experiment.

## Recommendation

Keep the Matryoshka fix. It is technically correct and produced a small measured gain.

Keep the router phrase additions and comparison-before-metadata precedence fix. Both regression queries now pass through the full retrieval pipeline, and normal ranked-list queries such as highly cited papers after a year still use the metadata path.

## Story For Interviews

The first hypothesis was that the embedding pipeline was throwing away useful geometry. We had been requesting the default OpenAI embedding size and then slicing the returned vector locally from 3072 dimensions down to 1024. That made storage cheaper, but it also meant the shortened vector was no longer the normalized Matryoshka representation intended by the embedding model. The fix was to request 1024 dimensions from the API directly, for both paper abstracts and full-text chunks, then rebuild the Qdrant collections from those vectors.

The second hypothesis was that the router was missing common comparison phrasing and that a downstream metadata shortcut was too aggressive. Queries such as "difference between X and Y" are not bibliography requests; they need a synthesized comparison across papers and chunks. I added explicit comparison signals for "difference between," "versus," "vs," and "how does X differ," then changed the query-pattern classifier so comparison intent is checked before ranked-list metadata language.

I measured the system before any code changes on the v2 fixture, then reran the same evaluation command after re-embedding, re-indexing, and fixing the routing precedence. The embedding change produced a small but real retrieval improvement: aggregate Recall@10 moved from 0.364 to 0.374, paper Recall@10 moved from 0.450 to 0.463, and MRR moved from 0.366 to 0.379. Chunk Recall@10 stayed flat at 0.182. The routing fix produced the larger operational improvement: route accuracy moved from 0.575 to 0.976, and both comparison regression queries now route to `hybrid_both`.

The decision is to keep both fixes. The Matryoshka change is technically cleaner and modestly improves retrieval. The routing change prevents comparison questions from being accidentally treated as ranked paper lists. What I learned is that retrieval quality depends on several layers working together: embedding geometry, route classification, and downstream orchestration all affect the final answer. Measuring each layer before and after made the improvement defensible instead of just intuitive.

## Follow-up Retrieval Fallback Ablation

After the Matryoshka and router fixes, I tested two retrieval-fusion follow-ups against the same 82-query v2 fixture. RRF was not kept: it preserved Hit Rate@10 and slightly improved aggregate Recall@10, but reduced Hit Rate@5, Recall@5, and MRR.

A better fix was to preserve the existing retrieval order when the default local cross-encoder cannot load. That matches the deployed free-tier configuration, where cross-encoder reranking is disabled for memory reasons. The preserve-order fallback improved Hit Rate@5 from 0.529 to 0.559, Recall@5 from 0.276 to 0.286, and MRR from 0.379 to 0.403, while Hit Rate@10 and Recall@10 stayed unchanged.

Recommendation: keep preserve-order fallback behavior and leave RRF as an opt-in ablation only.
