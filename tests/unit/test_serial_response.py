"""Unit tests for orchestrator.serial_response (decision-tree serial reply)."""
from __future__ import annotations

import pytest

from orchestrator.serial_response import build_serial_response


class TestBuildSerialResponse:
    """Test build_serial_response chooses correct template by triage/slots/next_question."""

    def test_critical_red_flags_returns_dispatch_msg(self):
        text = build_serial_response(
            slots={},
            category="medical",
            triage_level="CRITICAL",
            red_flags=["cardiac arrest"],
            is_complete=False,
            next_question_key="age",
            lang="en",
        )
        assert "dispatched" in text or "stay on the line" in text.lower()

    def test_critical_red_flags_turkish(self):
        text = build_serial_response(
            slots={},
            category="medical",
            triage_level="CRITICAL",
            red_flags=["kanama"],
            is_complete=False,
            next_question_key=None,
            lang="tr",
        )
        assert "Yardım" in text or "hatta" in text

    def test_is_complete_returns_complete_msg(self):
        text = build_serial_response(
            slots={"chief_complaint": "fall", "caller_name": "Ali", "age": "50"},
            category="medical",
            triage_level="URGENT",
            red_flags=[],
            is_complete=True,
            next_question_key=None,
            lang="en",
        )
        assert "enough information" in text or "on the way" in text.lower()

    def test_next_question_key_returns_question_en(self):
        text = build_serial_response(
            slots={"chief_complaint": "heart pain"},
            category="medical",
            triage_level="URGENT",
            red_flags=[],
            is_complete=False,
            next_question_key="age",
            lang="en",
        )
        assert "old" in text or "age" in text.lower()

    def test_next_question_key_chief_complaint_en(self):
        text = build_serial_response(
            slots={},
            category="other",
            triage_level="URGENT",
            red_flags=[],
            is_complete=False,
            next_question_key="chief_complaint",
            lang="en",
        )
        assert "describe" in text or "happened" in text or "wrong" in text

    def test_missing_chief_complaint_fallback_asks_chief_complaint(self):
        text = build_serial_response(
            slots={},
            category="other",
            triage_level="URGENT",
            red_flags=[],
            is_complete=False,
            next_question_key=None,
            lang="en",
        )
        assert "describe" in text or "happened" in text or "wrong" in text

    def test_unknown_next_question_key_uses_chief_complaint_question(self):
        text = build_serial_response(
            slots={"chief_complaint": "accident"},
            category="medical",
            triage_level="URGENT",
            red_flags=[],
            is_complete=False,
            next_question_key="unknown_key_xyz",
            lang="en",
        )
        assert len(text) > 0
        # Should still return some question or fallback
        assert isinstance(text, str)
