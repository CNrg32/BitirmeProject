"""Unit tests for mvp_rules: load_rules, infer_category, infer_triage, extract_slots, predict_mvp."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mvp_rules import (
    load_rules,
    infer_category,
    infer_triage,
    extract_slots,
    predict_mvp,
)


class TestLoadRules:
    def test_load_rules_returns_dict(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert isinstance(rules, dict)
        assert "categories" in rules
        assert "triage" in rules
        assert "slot_extraction_regex" in rules

    def test_load_rules_categories_have_expected_keys(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        cats = rules["categories"]
        assert "medical" in cats
        assert "fire" in cats
        assert "crime" in cats


class TestInferCategory:
    def test_medical_keywords(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_category("Someone is not breathing and unconscious", rules) == "medical"
        assert infer_category("heart attack chest pain", rules) == "medical"

    def test_fire_keywords(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_category("There is a fire in the house", rules) == "fire"
        assert infer_category("smoke and flames everywhere", rules) == "fire"

    def test_crime_keywords(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_category("Someone was shot with a gun", rules) == "crime"

    def test_other_fallback(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_category("something random happened", rules) == "other"

    def test_empty_string_returns_other(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_category("", rules) == "other"


class TestInferTriage:
    def test_critical_red_flags(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_triage("He is not breathing", 0, 0, 0, rules) == "CRITICAL"
        assert infer_triage("unconscious", 0, 0, 0, rules) == "CRITICAL"

    def test_deaths_meta_critical(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_triage("some text", 1, 0, 0, rules) == "CRITICAL"

    def test_potential_death_meta_critical(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_triage("some text", 0, 1, 0, rules) == "CRITICAL"

    def test_false_alarm_non_urgent(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_triage("false alarm", 0, 0, 1, rules) == "NON_URGENT"

    def test_urgent_keywords(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_triage("chest pain and difficulty breathing", 0, 0, 0, rules) == "URGENT"

    def test_default_urgent(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        assert infer_triage("something happened", 0, 0, 0, rules) == "URGENT"


class TestExtractSlots:
    def test_extract_age(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        slots = extract_slots("He is 63 years old", rules)
        assert slots.get("age") == 63

    def test_extract_severity(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        slots = extract_slots("pain is 8/10", rules)
        assert slots.get("severity_1_10") == 8

    def test_extract_duration_minutes(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        slots = extract_slots("for 15 minutes", rules)
        assert slots.get("duration_minutes") == 15

    def test_extract_red_flags(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        slots = extract_slots("not breathing and unconscious", rules)
        assert "red_flags" in slots
        assert len(slots["red_flags"]) >= 1

    def test_empty_text_returns_empty_slots(self):
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        slots = extract_slots("", rules)
        assert slots == {}


class TestPredictMvp:
    def test_predict_mvp_structure(self):
        out = predict_mvp("My dad is not breathing, he is 63 years old", 0, 0, 0)
        assert "category" in out
        assert "triage_level" in out
        assert "red_flags" in out
        assert "slots" in out
        assert out["triage_level"] in ("CRITICAL", "URGENT", "NON_URGENT")
        assert out["category"] in ("medical", "fire", "crime", "other")

    def test_predict_mvp_critical_medical(self):
        out = predict_mvp("Person is not breathing and unconscious", 0, 0, 0)
        assert out["triage_level"] == "CRITICAL"
        assert out["category"] == "medical"

    def test_predict_mvp_meta_deaths(self):
        out = predict_mvp("Something happened", deaths=1, potential_death=0, false_alarm=0)
        assert out["triage_level"] == "CRITICAL"
