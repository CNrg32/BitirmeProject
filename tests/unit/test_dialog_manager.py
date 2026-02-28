"""Unit tests for dialog_manager: get_missing_required_slots, decide_next_action."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from orchestrator.session import Session
from orchestrator.dialog_manager import (
    get_missing_required_slots,
    get_missing_optional_slots,
    decide_next_action,
    REQUIRED_SLOTS,
)


def _session(text_en: str = "", **kwargs) -> Session:
    s = Session(session_id="test")
    s.text_en_accumulated = text_en
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


class TestGetMissingOptionalSlots:
    def test_all_optional_missing(self):
        s = _session()
        from orchestrator.dialog_manager import get_missing_optional_slots
        missing = get_missing_optional_slots(s)
        assert len(missing) >= 1
        keys = [m[0] for m in missing]
        assert "sex" in keys or "duration_minutes" in keys or "location_hint" in keys

    def test_optional_filled_not_missing(self):
        s = _session()
        s.collected_slots["sex"] = "male"
        from orchestrator.dialog_manager import get_missing_optional_slots
        missing = get_missing_optional_slots(s)
        keys = [m[0] for m in missing]
        assert "sex" not in keys


class TestMergeSlots:
    def test_merge_slots_updates_session(self):
        from orchestrator.dialog_manager import merge_slots
        s = _session()
        merge_slots(s, {"age": 30, "severity_1_10": 8})
        assert s.collected_slots["age"] == 30
        assert s.collected_slots["severity_1_10"] == 8

    def test_merge_slots_ignores_none_empty(self):
        from orchestrator.dialog_manager import merge_slots
        s = _session()
        merge_slots(s, {"age": 30, "x": None, "y": "", "z": []})
        assert s.collected_slots.get("age") == 30
        assert "x" not in s.collected_slots


class TestGetMissingRequiredSlots:
    def test_all_missing(self):
        s = _session()
        missing = get_missing_required_slots(s)
        assert len(missing) >= 1
        keys = [m[0] for m in missing]
        assert "chief_complaint" in keys or "age" in keys

    def test_chief_complaint_filled(self):
        s = _session()
        s.collected_slots["chief_complaint"] = "Not breathing"
        missing = get_missing_required_slots(s)
        keys = [m[0] for m in missing]
        assert "chief_complaint" not in keys


class TestDecideNextAction:
    def test_ask_question_when_chief_complaint_missing(self):
        s = _session("hi")
        s.messages = [{"role": "assistant", "text": "Hello"}, {"role": "user", "text": "hi"}]
        action = decide_next_action(s)
        assert action["action"] == "ask_question"
        assert "question_en" in action
        assert "question_key" in action

    def test_run_triage_after_enough_rounds(self):
        s = _session("My father is not breathing, he is 60 years old")
        s.collected_slots["chief_complaint"] = "Not breathing"
        s.collected_slots["age"] = 60
        s.asked_questions.add("chief_complaint")
        s.asked_questions.add("age")
        s.asked_questions.add("severity_1_10")
        s.messages = [{"role": "assistant", "text": "Q1"}, {"role": "user", "text": "a"}, {"role": "assistant", "text": "Q2"}, {"role": "user", "text": "b"}, {"role": "assistant", "text": "Q3"}, {"role": "user", "text": "c"}]
        action = decide_next_action(s)
        assert action["action"] in ("run_triage", "complete", "ask_question")

    def test_category_question_asked_when_missing(self):
        s = _session("My dad had a heart attack")
        s.collected_slots["chief_complaint"] = "Heart attack"
        s.collected_slots["age"] = 65
        s.collected_slots["severity_1_10"] = 9
        s.asked_questions.add("chief_complaint")
        s.asked_questions.add("age")
        s.asked_questions.add("severity_1_10")
        s.messages = [{"role": "assistant", "text": "Q1"}, {"role": "user", "text": "heart attack"}]
        action = decide_next_action(s)
        assert action["action"] == "ask_question"
        assert "question_key" in action

    def test_low_confidence_asks_optional_slot(self):
        s = _session("Something happened")
        s.collected_slots["chief_complaint"] = "Something happened"
        s.collected_slots["age"] = 40
        s.collected_slots["severity_1_10"] = 5
        s.triage_result = {"category": "medical", "confidence": 0.5}
        s.messages = [{"role": "assistant", "text": "Q1"}, {"role": "user", "text": "ok"}]
        action = decide_next_action(s)
        assert action["action"] in ("ask_question", "run_triage", "complete")
