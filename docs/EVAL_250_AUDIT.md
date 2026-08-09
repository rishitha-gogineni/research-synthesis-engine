# Audited 250-query evaluation

Use `tests/fixtures/eval_queries_250_audited.json` for the next v1/v2/v3
comparison. The original `eval_queries_250.json` remains unchanged so earlier
results stay reproducible.

## Why the fixture changed

The legacy fixture mixed retrieval quality with ground-truth defects. Its 250
queries contained 989 paper-label assignments, including duplicate OpenAlex
records, topic-mismatched labels, and chunk-level labels that the chunk corpus
could not retrieve. Several generated questions and their keywords were also
mechanically truncated.

The audited fixture applies deterministic, corpus-backed repairs:

| Check | Legacy | Audited |
|---|---:|---:|
| Queries | 250 | 250 |
| Labeled queries | 233 | 233 |
| Paper-label assignments | 989 | 597 |
| Unique labeled papers | 133 | 126 |
| Chunk-level queries | 97 | 74 |
| Chunk queries with no reachable label | 11 | 0 |
| Topic-mismatched queries | 9 | 0 |
| Duplicate-title alias groups | untreated | 11 canonicalized |

The 23 reclassified queries target papers that have metadata but no reachable
canonical v2 full-text chunks. They now measure paper-level retrieval instead
of imposing an impossible chunk-level target. `data/paper_id_aliases.json`
allows evaluation results containing either member of a duplicate OpenAlex
pair to match the same canonical paper.

This is a corpus-alignment audit, not a claim that every relevance judgment was
human-adjudicated. Before publishing benchmark results, manually review the
broad synthesis and reading-list questions because relevance for those queries
is inherently subjective.

## Rebuild and verify

```bash
python scripts/build_eval_250_audited.py
python scripts/build_eval_250_audited.py --check
pytest tests/test_eval_250_audited.py tests/test_evaluate.py tests/test_breakdown_250_eval.py -q
```

The build also writes
`tests/fixtures/eval_queries_250_audit_manifest.json`, which records each
changed query, its before/after route and labels, and the repair reasons.

## Run all three retrieval strategies

Run every strategy against the same audited fixture and keep the output files
separate from the legacy results:

```bash
python -m retrieval.evaluate \
  --queries tests/fixtures/eval_queries_250_audited.json \
  --chunk-collection research_paper_chunks \
  --json > eval_250_audited_v1.json

python -m retrieval.evaluate \
  --queries tests/fixtures/eval_queries_250_audited.json \
  --chunk-collection research_paper_chunks_v2 \
  --json > eval_250_audited_v2.json

python -m retrieval.evaluate \
  --queries tests/fixtures/eval_queries_250_audited.json \
  --chunk-collection research_paper_chunks_v3 \
  --json > eval_250_audited_v3.json

python scripts/compare_250_eval.py \
  --v1 eval_250_audited_v1.json \
  --v2 eval_250_audited_v2.json \
  --v3 eval_250_audited_v3.json
```

Do not compare an audited score directly with a legacy score: the relevance
labels and route expectations differ. Compare v1, v2, and v3 only when they
were run on the same fixture and code revision.

One source-data issue remains separate from retrieval: the local metadata lists
“Attention Is All You Need” as 2025. The benchmark builder treats it as 2017
for year filters without mutating the source corpus. Correct and re-index that
record separately if production filtering also needs the fix.
