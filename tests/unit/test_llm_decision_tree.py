"""Unit tests for LLM decision-tree JSON parsing (llm_service._parse_decision_tree_json)."""
from __future__ import annotations

import pytest

from services.llm_service import _parse_decision_tree_json, _EMPTY_LLM_RESPONSE


class TestParseDecisionTreeJson:
    """Test _parse_decision_tree_json extracts steps and flattens to slots/triage/category."""

    def test_valid_full_json(self):
        raw = """{
            "decision_steps": [
                {"step_id": "extract_slots", "extracted_slots": {"chief_complaint": "heart attack", "age": "60"}},
                {"step_id": "category", "category": "medical"},
                {"step_id": "triage", "triage_level": "CRITICAL", "red_flags": ["chest pain"]},
                {"step_id": "completion", "is_complete": true, "next_question_key": null}
            ],
            "extracted_slots": {"chief_complaint": "heart attack", "age": "60"},
            "category": "medical",
            "triage_level": "CRITICAL",
            "is_complete": true,
            "red_flags": ["chest pain"]
        }"""
        result = _parse_decision_tree_json(raw)
        assert result["response_text"] == ""
        assert result["extracted_slots"].get("chief_complaint") == "heart attack"
        assert result["extracted_slots"].get("age") == "60"
        assert result["category"] == "medical"
        assert result["triage_level"] == "CRITICAL"
        assert result["is_complete"] is True
        assert result["red_flags"] == ["chest pain"]
        assert result.get("next_question_key") is None
        assert "decision_steps" in result

    def test_steps_override_top_level(self):
        raw = """{
            "decision_steps": [
                {"step_id": "category", "category": "fire"},
                {"step_id": "triage", "triage_level": "URGENT", "red_flags": []},
                {"step_id": "completion", "is_complete": false, "next_question_key": "trapped"}
            ],
            "category": "other",
            "triage_level": "NON_URGENT",
            "is_complete": false
        }"""
        result = _parse_decision_tree_json(raw)
        assert result["category"] == "fire"
        assert result["triage_level"] == "URGENT"
        assert result["next_question_key"] == "trapped"
        assert result["is_complete"] is False

    def test_invalid_json_returns_empty_shape(self):
        result = _parse_decision_tree_json("not json at all")
        assert result["response_text"] == ""
        assert result["extracted_slots"] == _EMPTY_LLM_RESPONSE["extracted_slots"]
        assert result["triage_level"] == "URGENT"
        assert result["category"] == "other"
        assert result["is_complete"] is False
        assert result["red_flags"] == []

    def test_empty_object_returns_defaults(self):
        result = _parse_decision_tree_json("{}")
        assert result["response_text"] == ""
        assert result["triage_level"] == "URGENT"
        assert result["category"] == "other"
        assert result["is_complete"] is False
        assert result["red_flags"] == []

    def test_json_in_code_fence_stripped(self):
        raw = """```json
        {"category": "crime", "triage_level": "CRITICAL", "is_complete": false, "red_flags": []}
        ```"""
        result = _parse_decision_tree_json(raw)
        assert result["category"] == "crime"
        assert result["triage_level"] == "CRITICAL"
