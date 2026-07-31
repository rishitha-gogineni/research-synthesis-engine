from retrieval.corpus_index import CorpusIndex, classify_question_pattern, normalize_text


def sample_index():
    papers = [
        {
            "paper_id": "p-bitfit",
            "title": "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models",
            "topic": "Fine-tuning (LoRA / PEFT)",
            "year": 2022,
            "citation_count": 677,
            "abstract": "Introduction of BitFit, a sparse fine-tuning method.",
            "main_contribution": "Introduces BitFit.",
        },
        {
            "paper_id": "p-agent-survey",
            "title": "A survey on large language model based autonomous agents",
            "topic": "AI Agents & Tool Use",
            "year": 2024,
            "citation_count": 1205,
            "abstract": "A comprehensive survey of LLM-based autonomous agents.",
            "main_contribution": "Survey of autonomous agents.",
        },
        {
            "paper_id": "p-agent-low",
            "title": "A review of large language models and autonomous agents in chemistry",
            "topic": "AI Agents & Tool Use",
            "year": 2024,
            "citation_count": 204,
            "abstract": "Review of LLM agents in chemistry.",
            "main_contribution": "Review of agents in chemistry.",
        },
        {
            "paper_id": "p-old-agent",
            "title": "Older survey on agents",
            "topic": "AI Agents & Tool Use",
            "year": 2023,
            "citation_count": 9999,
            "abstract": "Survey before the cutoff.",
            "main_contribution": "Older survey.",
        },
    ]
    chunks = [
        {
            "chunk_id": "c-bitfit-1",
            "paper_id": "p-bitfit",
            "title": "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models",
            "topic": "Fine-tuning (LoRA / PEFT)",
            "chunk_index": 0,
            "text": "BitFit updates only bias terms during fine-tuning.",
        },
        {
            "chunk_id": "c-bitfit-2",
            "paper_id": "p-bitfit",
            "title": "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models",
            "topic": "Fine-tuning (LoRA / PEFT)",
            "chunk_index": 1,
            "text": "BitFit can be competitive with full fine-tuning.",
        },
    ]
    return CorpusIndex.from_records(papers, chunks)


def test_normalize_text_supports_title_keys():
    assert normalize_text("BitFit: Simple PEFT!") == "bitfit simple peft"


def test_question_pattern_classifier_matches_common_human_questions():
    assert classify_question_pattern("Show me highly cited AI agent survey papers published after 2023.") == "ranked_list"
    assert classify_question_pattern('Explain the paper "BitFit: Simple Parameter-efficient Fine-tuning"') == "paper_lookup"
    assert classify_question_pattern("Compare LoRA and BitFit.") == "comparison"
    assert classify_question_pattern("What datasets evaluate LoRA?") == "dataset_method"
    assert classify_question_pattern("What are its limitations?", has_chat_history=True) == "follow_up"


def test_title_lookup_resolves_exact_and_partial_paper_mentions():
    index = sample_index()

    exact = index.resolve_paper('Explain the paper "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models"')
    partial = index.resolve_paper("Explain the BitFit paper.")

    assert exact is not None
    assert exact.paper_id == "p-bitfit"
    assert partial is not None
    assert partial.paper_id == "p-bitfit"


def test_ranked_papers_use_topic_year_and_citation_indexes():
    index = sample_index()

    rows = index.ranked_papers("Show me highly cited AI agent survey papers published after 2023.", top_k=5)

    assert [row["paper_id"] for row in rows] == ["p-agent-survey", "p-agent-low"]
    assert all(row["year"] > 2023 for row in rows)
    assert rows[0]["citation_count"] > rows[1]["citation_count"]
    assert rows[0]["matched_by"] == ["metadata_filter", "corpus_index"]


def test_paper_lookup_returns_ordered_chunks_for_resolved_paper():
    index = sample_index()

    chunks = index.chunk_candidates_for_paper("p-bitfit", top_k=2)

    assert [chunk["chunk_id"] for chunk in chunks] == ["c-bitfit-1", "c-bitfit-2"]
    assert chunks[0]["matched_by"] == ["paper_lookup", "corpus_index"]
