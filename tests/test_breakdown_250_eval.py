import json

from scripts.breakdown_250_eval import is_hit, semantic_category


def test_breakdown_does_not_count_legacy_stringified_empty_set_as_hit():
    evaluation = {
        "id_hit_sets": {"10": "set()"},
        "id_hit_fractions": {"10": 0.0},
    }

    assert is_hit(evaluation, 10) is False


def test_breakdown_counts_positive_recall_fraction_as_hit():
    evaluation = {
        "id_hit_sets": {"10": "{'paper-1'}"},
        "id_hit_fractions": {"10": 0.25},
    }

    assert is_hit(evaluation, 10) is True


def test_section_rationales_roll_up_to_one_semantic_category():
    assert semantic_category("[section:results] result lookup") == "section_specific"
    assert semantic_category("[section:methodology] method lookup") == "section_specific"


def test_evaluation_hit_sets_are_json_native():
    payload = {"id_hit_sets": {5: [], 10: ["paper-1"]}}

    assert json.loads(json.dumps(payload))["id_hit_sets"] == {
        "5": [],
        "10": ["paper-1"],
    }
