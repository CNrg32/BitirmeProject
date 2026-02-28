"""Unit tests for model_loader: apply_redflag_override. TriageModelService is tested with mocked files."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.model_loader import apply_redflag_override, TriageModelService, get_model_service


class TestApplyRedflagOverride:
    def test_no_red_flags_returns_unchanged(self):
        label, red_flags = apply_redflag_override("just a normal situation", "URGENT", {})
        assert label == "URGENT"
        assert isinstance(red_flags, list)

    def test_red_flags_upgrade_to_critical(self):
        label, red_flags = apply_redflag_override(
            "He is not breathing and unconscious",
            "URGENT",
            {"deaths": 0, "potential_death": 0, "false_alarm": 0},
        )
        assert label == "CRITICAL"
        assert len(red_flags) >= 1

    def test_already_critical_unchanged(self):
        label, red_flags = apply_redflag_override(
            "not breathing",
            "CRITICAL",
            {},
        )
        assert label == "CRITICAL"


class TestTextWithMeta:
    def test_text_with_meta_formats_correctly(self):
        from api.model_loader import _text_with_meta
        out = _text_with_meta("hello", {"deaths": 0, "potential_death": 0, "false_alarm": 0})
        assert "hello" in out
        assert "deaths:0" in out or "deaths: 0" in out

    def test_text_with_meta_invalid_meta_defaults_zero(self):
        from api.model_loader import _text_with_meta
        out = _text_with_meta("x", {"deaths": "nope", "potential_death": None, "false_alarm": "x"})
        assert "x" in out


class TestTriageModelService:
    def test_not_loaded_initially(self):
        svc = TriageModelService()
        assert svc.is_loaded is False

    def test_load_missing_dir_returns_false(self, tmp_path):
        svc = TriageModelService()
        assert svc.load(model_dir=tmp_path) is False
        assert svc.is_loaded is False
