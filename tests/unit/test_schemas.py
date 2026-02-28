"""Unit tests for Pydantic schemas (validation, defaults)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.schemas import (
    TriageLevel,
    Category,
    MetaInput,
    SlotsInput,
    PredictRequest,
    PredictResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionMessageRequest,
    SessionMessageResponse,
    SessionTranscribeRequest,
    ImageAnalysisResult,
)


class TestPredictRequest:
    def test_valid_request(self):
        r = PredictRequest(
            text_en="Someone is not breathing",
            meta=MetaInput(),
            slots=SlotsInput(),
        )
        assert r.text_en == "Someone is not breathing"
        assert r.meta.deaths == 0

    def test_empty_text_en_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(text_en="", meta=MetaInput(), slots=SlotsInput())

    def test_min_length_enforced(self):
        with pytest.raises(ValidationError):
            PredictRequest(text_en="", meta=MetaInput(), slots=SlotsInput())

    def test_meta_defaults(self):
        r = PredictRequest(text_en="x")
        assert r.meta.deaths == 0
        assert r.meta.potential_death == 0
        assert r.meta.false_alarm == 0


class TestSessionMessageRequest:
    def test_text_only(self):
        r = SessionMessageRequest(session_id="abc123", text="Help")
        assert r.session_id == "abc123"
        assert r.text == "Help"
        assert r.audio_base64 is None
        assert r.image_base64 is None

    def test_optional_fields_none(self):
        r = SessionMessageRequest(session_id="sid")
        assert r.text is None
        assert r.latitude is None
        assert r.longitude is None


class TestSessionTranscribeRequest:
    def test_requires_audio_base64(self):
        r = SessionTranscribeRequest(session_id="sid", audio_base64="YQ==")
        assert r.session_id == "sid"
        assert r.audio_base64 == "YQ=="


class TestPredictResponse:
    def test_minimal_response(self):
        r = PredictResponse(
            category="medical",
            triage_level="CRITICAL",
            red_flags=[],
            slots={},
        )
        assert r.category == "medical"
        assert r.triage_level == "CRITICAL"
        assert r.needs_more_info is False
        assert r.recommended_questions == []


class TestImageAnalysisResult:
    def test_available_false(self):
        r = ImageAnalysisResult(summary="Not loaded", available=False)
        assert r.available is False
        assert r.classification is None


class TestEnums:
    def test_triage_levels(self):
        assert TriageLevel.CRITICAL.value == "CRITICAL"
        assert TriageLevel.URGENT.value == "URGENT"
        assert TriageLevel.NON_URGENT.value == "NON_URGENT"

    def test_categories(self):
        assert Category.MEDICAL.value == "medical"
        assert Category.OTHER.value == "other"
