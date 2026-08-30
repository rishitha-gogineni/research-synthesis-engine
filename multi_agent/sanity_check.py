"""End-to-end sanity check: sample chunks, generate questions, verify pipeline.

For each sampled chunk:
1. Show the chunk content (ground truth)
2. Generate a natural question about it
3. Run the multi-agent pipeline
4. Check if the answer cites the source and reflects the content

Usage:
    python3 -m multi_agent.sanity_check --count 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from openai import OpenAI

from ingestion.embed import load_env_file


DEFAULT_CHUNKS = Path("data/full_text_chunks.json")

# Reject chunks that look like reference lists or metadata
CITATION_RE = re.compile(r"\[\d+\]|arXiv:\d{4}\.\d+")


def is_distinctive_chunk(chunk: dict) -> bool:
    """Filter for chunks with core scientific content, not references or boilerplate."""
    text = str(chunk.get("text", ""))
    if len(text) < 500:
        return False

    section = chunk.get("section_hint", "").lower()
    if section in {"references", "bibliography", "acknowledgments", "unknown"}:
        return False
    if section not in {"introduction", "methodology", "methods", "results", "experiments", "conclusion", "abstract"}:
        return False

    # Reject chunks with too many citation markers (likely a reference list)
    citation_count = len(CITATION_RE.findall(text))
    if citation_count > 5:
        return False

    # Prefer chunks with mid-length sentences (real prose, not tables/formulas)
    sentences = re.split(r"[.!?]+", text)
    long_sentences = [s for s in sentences if 40 < len(s.strip()) < 250]
    if len(long_sentences) < 3:
        return False

    return True


def generate_question(chunk: dict, client: OpenAI) -> str:
    text = str(chunk.get("text", ""))[:1500]
    title = chunk.get("title", "")
    topic = chunk.get("topic", "")
    prompt = f"""You are creating a retrieval evaluation query.

Given this chunk from the paper "{title}" (topic: {topic}), write ONE natural question that a researcher would ask if they wanted to find this SPECIFIC paper's contribution.

Chunk:
{text}

Rules:
- Question must be about the paper's ACTUAL contribution, method, or finding — not tangential examples
- 10-20 words
- Reference the paper's method/concept by name where possible
- Do NOT copy phrases directly, but stay grounded in what THIS paper is about
- The question should be answerable ONLY by finding this paper (or very similar ones)

Return ONLY the question, no preamble."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def run_sanity_check(count: int = 3, seed: int = 7) -> None:
    load_env_file(Path(".env"))
    from multi_agent.orchestrator import run_research

    chunks = json.loads(DEFAULT_CHUNKS.read_text(encoding="utf-8"))
    good = [c for c in chunks if is_distinctive_chunk(c)]
    print(f"Filtered {len(good)}/{len(chunks)} distinctive chunks")

    random.seed(seed)
    random.shuffle(good)

    # Diversify by paper_id
    seen_papers = set()
    sampled = []
    for c in good:
        pid = c.get("paper_id")
        if pid in seen_papers:
            continue
        seen_papers.add(pid)
        sampled.append(c)
        if len(sampled) >= count:
            break

    client = OpenAI()

    hits_in_findings = 0
    hits_in_citations = 0

    for i, chunk in enumerate(sampled, 1):
        print("\n" + "=" * 80)
        print(f"SANITY CHECK {i}/{count}")
        print("=" * 80)
        title = chunk.get("title", "")
        paper_id = chunk.get("paper_id", "")
        print(f"Paper:   {title[:80]}")
        print(f"Section: {chunk.get('section_hint', 'unknown')}")
        print(f"Chunk ID: {chunk.get('chunk_id', '')}")
        print(f"\nGROUND TRUTH (chunk content, first 400 chars):")
        print("-" * 80)
        print(str(chunk.get("text", ""))[:400])
        print("-" * 80)

        print("\nGenerating question...")
        question = generate_question(chunk, client)
        print(f"QUESTION: {question}")

        print("\nRunning multi-agent pipeline...")
        result = run_research(question, openai_client=client)

        synthesis = result.get("synthesis", {}).get("synthesis", "")
        store_summary = result.get("store_summary", {})
        judge = result.get("judge_scores", {})
        cited_report = result.get("cited_report", {})

        print(f"\nPIPELINE RESULT:")
        print(f"  Agents used: {store_summary.get('total_agents', 0)}")
        print(f"  Findings: {store_summary.get('total_findings', 0)}")
        print(f"  Judge overall: {judge.get('overall', 'N/A')}")

        print(f"\nANSWER (first 500 chars):")
        print("-" * 80)
        print(synthesis[:500] if synthesis else "(empty)")
        print("-" * 80)

        # Check retrieval fidelity - did we find the source paper?
        title_words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", title)}

        # Check findings for paper_id or title overlap
        found_in_findings = False
        findings_text = json.dumps(result.get("synthesis", {})).lower()
        for f in result.get("cited_report", {}).get("references", []):
            ref_str = json.dumps(f).lower() if isinstance(f, dict) else str(f).lower()
            if paper_id and paper_id.lower() in ref_str:
                found_in_findings = True
                break
            ref_words = set(re.findall(r"[a-z]{4,}", ref_str))
            overlap = title_words & ref_words
            if len(overlap) >= 3:
                found_in_findings = True
                break

        # Also check the raw synthesis text for the paper title
        if title[:30].lower() in findings_text:
            found_in_findings = True

        if found_in_findings:
            hits_in_findings += 1
        print(f"\nSOURCE PAPER FOUND: {found_in_findings}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {hits_in_findings}/{count} queries retrieved the source paper")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run_sanity_check(count=args.count, seed=args.seed)
