# Good morning — Day 1 status (2026-08-09)

## Summary

I completed Day 1 of the PyMuPDF migration overnight. **All 152 papers extracted, chunked, and ready to embed.** Zero failures, zero data loss.

You're now ready for Day 2 (embed + eval), which is **one command**.

---

## What I did while you slept

| Task | Result |
|------|--------|
| 1. Inventory | 152/152 PDFs OK, 0 corrupt, 0 scanned. Total 2,913 pages. |
| 2. Map PDFs → RSE paper IDs | 152/152 matched via OpenAlex ID |
| 4. PyMuPDF extraction | 152/152 extracted successfully in 155s. **11.4x more paragraph structure preserved vs pypdf.** |
| 5. Paragraph + section-aware chunking | 4,187 chunks (vs v1's 4,909) — respects section boundaries, skips references |

## Key metric so far

**PyMuPDF preserved 49,998 paragraph breaks vs pypdf's 4,399.** That's the whole reason for this migration — pypdf was destroying paragraph structure, PyMuPDF preserves it.

## Files created

```
scripts/
├── inventory_pdfs.py          — PDF quality check
├── map_pdfs_to_paper_ids.py   — filename → paper_id mapping
├── extract_pymupdf.py         — PyMuPDF blocks extraction with section detection
├── chunk_paragraph_aware.py   — greedy paragraph packer
├── run_day2_pipeline.py       — YOUR DAY 2 ENTRY POINT
├── compare_v1_v2_metrics.py   — v1 vs v2 side-by-side
├── compare_pdf_extractors.py  — pypdf vs pdfplumber vs PyMuPDF (diagnostic)
└── compare_chunking.py        — 6-strategy chunking diagnostic

data/
├── pdf_to_paper_id.json           — 152 mappings (gitignored)
├── full_text_papers_v2.json       — 12.1MB extracted text + section tags
└── full_text_chunks_v2.json       — 4,187 chunks ready to embed
```

## Git status

Committed to branch **`pymupdf-migration`** (NOT pushed yet — I didn't want to touch remote without your OK). Commit message summarizes everything.

To see what changed:
```bash
git log --oneline -1
git diff main pymupdf-migration --stat
```

---

## What YOU do today (Day 2) — ~1 hour, ~$3-5

### Step 1: Check Docker is running (30 sec)

Open Docker Desktop if not already open.

### Step 2: Activate venv (5 sec)

```bash
cd "/Users/tpothune/Documents/Rishitha/Research Synthesis Engine"
source .venv/bin/activate
```

### Step 3: Install any missing deps (2 min)

The venv is fresh. May need:
```bash
pip install -e ".[dev]"
```

If that's slow or errors, at minimum you need:
```bash
pip install openai qdrant-client python-dotenv pypdf pymupdf pydantic fastapi
```

### Step 4: Run the Day 2 pipeline (45-60 min, mostly automated)

```bash
python scripts/run_day2_pipeline.py
```

This does:
1. Embeds 4,187 v2 chunks with `text-embedding-3-large` (~$1-2, ~15 min)
2. Uploads them to Qdrant Cloud into a **new collection** `research_paper_chunks_v2`
   → **v1 collection stays untouched. Deployed app not affected.**
3. Runs baseline eval on v1 collection → `eval_v1_baseline.json`
4. Runs eval on v2 collection → `eval_v2_pymupdf.json`

### Step 5: See the results

```bash
python scripts/compare_v1_v2_metrics.py
```

Prints a table like:
```
Metric                       v1         v2          delta
------------------------------------------------------------
route_accuracy            1.000      1.000         +0.000
hit_rate@10               0.721      0.XXX      +X.XXX (+X%)
recall@10                 0.393      0.XXX      +X.XXX (+X%)
mrr                       0.412      0.XXX      +X.XXX (+X%)
```

---

## Expected outcome

Honest prediction based on what changed:

| Metric | Expected direction | Rationale |
|--------|-------------------|-----------|
| **hit@10** | +2-5% | Better chunks = tighter semantic matches |
| **recall@10** | +3-8% | Paragraph boundaries preserve context |
| **MRR** | +2-5% | Cleaner chunks rank higher |
| **route_accuracy** | Unchanged (still 1.000) | Router doesn't care about chunks |
| **confidence_gate** | Unchanged | Independent of chunk quality |

**If numbers are flat or drop**, likely causes:
1. `text-embedding-3-large` finds v1's noisy chunks equally well (semantic robustness > structure)
2. Eval fixture chunk IDs don't match new IDs → fallback to paper_id match works but sub-optimal
3. Some queries were previously hitting duplicated/boilerplate chunks

Either way, we'll know within an hour of you kicking off the pipeline.

---

## If something breaks

**Common failure modes and fixes:**

1. **"Module not found"** → `pip install -e ".[dev]"` in venv
2. **"OPENAI_API_KEY not set"** → check `.env` is in project root
3. **"Qdrant collection already exists"** → the indexer handles this, but if it errors, delete the collection first via Qdrant Cloud dashboard
4. **"Rate limit"** → the embed script has retries; if it stops, just re-run the pipeline (embeddings are cached in `data/embedded_full_text_chunks_v2.json`, so it won't re-embed what's already done)
5. **Eval hangs** → OpenAI slowness. Wait or Ctrl+C and re-run.

---

## Day 3 (tomorrow) — depends on today's results

### If v2 wins on recall:
- Update README with before/after comparison
- Merge `pymupdf-migration` → `main`
- Redeploy so prod uses new collection
- **Interview story: "Identified pypdf as extraction bottleneck via multi-extractor comparison. Migrated to PyMuPDF, improved recall@10 by X%."**

### If v2 is flat/worse:
- Still write the "interesting finding" README section — "Migration preserved 11x more paragraph structure but retrieval was already robust to structure loss"
- **Interview story: "Ran controlled A/B on extraction quality. Discovered that text-embedding-3-large is robust enough that chunking structure matters less than I expected — retrieval quality is dominated by embedding model, not chunk boundaries."**
- **Both are legitimate engineering wins.** You measured, you learned.

---

## Cost tracking

- Day 1 (me overnight): **$0** (all local)
- Day 2 (you today): **~$3-5** (embedding + eval LLM calls)
- Day 3 (whichever direction): **~$0** (docs/deploy)

Total: **under $10 for the entire migration.**

---

## Questions I couldn't answer

If any of these matter, let me know when you're back:

1. **Should I push the branch to GitHub?** I didn't — safer to let you do it after eval results.
2. **Deploy immediately if v2 wins?** I'd recommend waiting to double-check the deployed app still works with the new collection name before flipping.
3. **Keep v1 collection forever or delete after migration?** I'd keep it for a week as insurance.

---

Sleep well is over — coffee up, hit the pipeline, and we'll know the outcome in an hour.

**One command:** `python scripts/run_day2_pipeline.py`

—Claude
