from ingestion.fetch_papers import fetch_topic, normalize_paper, reconstruct_abstract


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def get(self, url, params, headers, timeout):
        self.requests.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeResponse(self.payload)


def test_reconstruct_abstract_from_openalex_inverted_index():
    abstract = reconstruct_abstract({"RAG": [3], "Grounded": [0], "generation": [2], "improves": [1]})

    assert abstract == "Grounded improves generation RAG"


def test_normalize_paper_skips_missing_abstract():
    paper = normalize_paper(
        {
            "id": "https://openalex.org/W123",
            "title": "A useful paper",
            "abstract_inverted_index": None,
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "cited_by_count": 10,
        },
        topic="RAG",
    )

    assert paper is None


def test_fetch_topic_normalizes_openalex_payload():
    session = FakeSession(
        {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "doi": "https://doi.org/10.1000/example",
                    "title": "Retrieval augmented generation",
                    "abstract_inverted_index": {
                        "A": [0],
                        "paper": [1],
                        "about": [2],
                        "grounded": [3],
                        "generation.": [4],
                    },
                    "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                    "cited_by_count": 42,
                    "publication_year": 2024,
                    "ids": {
                        "openalex": "https://openalex.org/W123",
                        "doi": "https://doi.org/10.1000/example",
                        "arxiv": "https://arxiv.org/abs/2401.12345",
                    },
                    "primary_location": {"landing_page_url": "https://example.com/paper"},
                }
            ]
        }
    )

    papers, skipped = fetch_topic(
        topic="Retrieval-Augmented Generation (RAG)",
        per_topic=1,
        session=session,
        api_key="test-key",
        mailto="rishitha@example.com",
        delay_seconds=0,
    )

    assert skipped == 0
    assert len(papers) == 1
    assert papers[0].paper_id == "https://openalex.org/W123"
    assert papers[0].abstract == "A paper about grounded generation."
    assert papers[0].authors == ["Ada Lovelace"]
    assert papers[0].citation_count == 42
    assert papers[0].arxiv_id == "2401.12345"
    assert papers[0].topic == "Retrieval-Augmented Generation (RAG)"
    assert session.requests[0]["params"]["api_key"] == "test-key"
    assert session.requests[0]["params"]["mailto"] == "rishitha@example.com"
    assert session.requests[0]["params"]["sort"] == "cited_by_count:desc"
    assert session.requests[0]["params"]["filter"] == "title.search:retrieval augmented generation"
