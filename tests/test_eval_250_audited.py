import json

from scripts.build_eval_250_audited import (
    ALIAS_OUTPUT,
    AUDIT_MANIFEST,
    OUTPUT_FIXTURE,
    build,
    serialize,
    validate,
)
from scripts.generate_eval_250 import has_field


def test_has_field_rejects_corpus_placeholder_values():
    for value in (
        "",
        "not specified",
        "Not specified.",
        "not stated in abstract",
        "unknown",
        "N/A",
    ):
        assert has_field({"limitations": value}, "limitations") is False

    assert has_field({"limitations": "The study covers only English data."}, "limitations") is True


def test_audited_fixture_is_reproducible_and_corpus_aligned():
    queries, aliases, manifest = build()
    stats = validate(queries, aliases)
    manifest["validation"] = stats

    assert OUTPUT_FIXTURE.read_text() == serialize(queries)
    assert ALIAS_OUTPUT.read_text() == serialize(aliases)
    assert AUDIT_MANIFEST.read_text() == serialize(manifest)
    assert stats == {
        "queries": 250,
        "labeled_queries": 233,
        "label_assignments": 597,
        "unique_labeled_papers": 126,
        "chunk_level_queries": 74,
        "unreachable_chunk_labels": 0,
        "topic_mismatched_queries": 0,
        "duplicate_alias_groups": 11,
    }

    serialized = serialize(queries).lower()
    for fragment in (
        "retrieval-augmented-related",
        "evaluating-related",
        "interactive-related",
        '"criter"',
        '"photo-r"',
    ):
        assert fragment not in serialized


def test_alias_map_has_no_cycles_or_alias_targets():
    payload = json.loads(ALIAS_OUTPUT.read_text())
    aliases = payload["aliases"]

    assert aliases
    assert not (set(aliases) & set(aliases.values()))
