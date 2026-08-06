from ingestion.embed import build_embedding_record, build_embedding_text, embed_texts
from shared.schemas import EnrichedPaper


def make_enriched_paper() -> EnrichedPaper:
    return EnrichedPaper(
        paper_id="W123",
        title="A test paper",
        abstract="This paper studies retrieval augmented generation.",
        authors=["Ada Lovelace"],
        citation_count=42,
        arxiv_id="2401.12345",
        url="https://example.com",
        year=2024,
        topic="Retrieval-Augmented Generation (RAG)",
        main_contribution="Introduces a retrieval method.",
        methodology="Evaluates retrieval over question answering.",
        dataset_used="QA benchmark",
        key_result="Improves answer grounding.",
        limitations="not stated in abstract",
    )


def test_build_embedding_text_contains_core_fields():
    text = build_embedding_text(make_enriched_paper())

    assert "Title: A test paper" in text
    assert "Topic: Retrieval-Augmented Generation (RAG)" in text
    assert "Main contribution: Introduces a retrieval method." in text
    assert "Limitations: not stated in abstract" in text


class FakeEmbeddingResponse:
    def __init__(self, embedding: list[float]) -> None:
        self.data = [type("EmbeddingItem", (), {"embedding": embedding})()]


class FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeEmbeddingResponse([0.1, 0.2, 0.3])


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsClient()


def test_embed_texts_requests_server_side_dimensions():
    client = FakeOpenAIClient()

    embeddings = embed_texts(client, "text-embedding-3-large", ["hello"], dimensions=1024)

    assert embeddings == [[0.1, 0.2, 0.3]]
    assert client.embeddings.calls == [
        {"model": "text-embedding-3-large", "input": ["hello"], "dimensions": 1024}
    ]


def test_build_embedding_record_stores_metadata_and_server_sized_embedding():
    paper = make_enriched_paper()
    full_embedding = [float(index) for index in range(1024)]

    record = build_embedding_record(
        paper=paper,
        full_embedding=full_embedding,
        embedding_text=build_embedding_text(paper),
        model="text-embedding-3-large",
        dimensions=1024,
    )

    assert record["paper_id"] == "W123"
    assert record["embedding_model"] == "text-embedding-3-large"
    assert record["full_embedding_dimensions"] == 1024
    assert record["embedding_dimensions"] == 1024
    assert len(record["embedding"]) == 1024
    assert record["metadata"]["main_contribution"] == "Introduces a retrieval method."

