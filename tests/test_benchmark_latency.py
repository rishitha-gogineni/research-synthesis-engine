from tools.benchmark_latency import build_payload, format_table


def test_build_payload_enables_debug_metrics():
    payload = build_payload("Compare RAG and agents.", top_k=7)

    assert payload == {"question": "Compare RAG and agents.", "top_k": 7, "include_debug": True}


def test_build_payload_supports_fast_first_guidance():
    payload = build_payload("Compare RAG and agents.", top_k=7, fast_first=True)

    assert payload["include_evidence_matrix"] is True
    assert payload["include_reading_path"] is False
    assert payload["include_open_problems"] is False


def test_format_table_renders_latency_columns():
    table = format_table(
        [
            {
                "endpoint": "/guidance",
                "status": 200,
                "wall_ms": 123.4,
                "api_total_ms": 120.0,
                "retrieval_ms": 10.0,
                "confidence_ms": 2.0,
                "brief_ms": 50.0,
                "evidence_matrix_ms": 15.0,
                "reading_path_ms": None,
                "open_problems_ms": None,
                "error": None,
            }
        ]
    )

    assert "endpoint | status | wall_ms" in table
    assert "/guidance | 200 | 123.4" in table
    assert "15.0 | - | - | -" in table
