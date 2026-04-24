from __future__ import annotations

import os
import time
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from orchestrator.session import Session
from services.case_store import (
    CaseRecord,
    build_case_record_from_session,
    persist_case_if_configured,
)


def _make_session(**overrides: Any) -> Session:
    session = Session(
        session_id="sess_abc123",
        language="tr",
        created_at=time.time() - 120,
    )
    session.is_complete = True
    session.dispatch_status = "DISPATCHED"
    session.dispatch_target = "medical"
    session.dispatch_timestamp = time.time() - 30
    session.collected_slots = {
        "caller_name": "Ayşe",
        "age": 63,
        "location_hint": "Kadıköy",
        "latitude": 40.99,
        "longitude": 29.02,
    }
    session.messages = [
        {"role": "user", "text": "Babam nefes almıyor"},
        {"role": "assistant", "text": "Ambulans yönlendiriyorum"},
        {"role": "user", "text": "Kadıköy'deyiz"},
        {"role": "assistant", "text": "Ekip yolda"},
    ]
    session.triage_result = {
        "triage_level": "CRITICAL",
        "category": "medical",
        "confidence": 0.94,
        "red_flags": ["not_breathing", "unconscious"],
    }
    session.image_analysis = {
        "available": True,
        "summary": "Indoor scene, visible injury",
        "classification": {"detected_class": "medical_emergency", "confidence": 0.88},
        "visual_triage": {
            "triage_level": "CRITICAL",
            "action": "EARLY_DISPATCH",
            "visual_flags": ["blood_visible"],
        },
    }
    for key, value in overrides.items():
        setattr(session, key, value)
    return session


def test_build_case_record_from_session_captures_core_fields():
    session = _make_session()

    record = build_case_record_from_session(session, final_report="FINAL REPORT TEXT")

    assert isinstance(record, CaseRecord)
    assert record.session_id == "sess_abc123"
    assert record.language == "tr"
    assert record.is_complete is True
    assert record.user_turn_count == 2
    assert record.dispatch_status == "DISPATCHED"
    assert record.dispatch_target == "medical"
    assert record.triage_level == "CRITICAL"
    assert record.category == "medical"
    assert record.confidence == pytest.approx(0.94)
    assert record.red_flags == ["not_breathing", "unconscious"]
    assert record.final_report == "FINAL REPORT TEXT"
    assert record.collected_slots["age"] == 63
    assert len(record.messages) == 4
    assert record.created_at and record.created_at.endswith("+00:00")
    assert record.dispatch_timestamp and record.dispatch_timestamp.endswith("+00:00")


def test_to_item_coerces_floats_to_decimal_for_dynamodb():
    session = _make_session()
    record = build_case_record_from_session(session, final_report="r")

    item = record.to_item()

    assert item["session_id"] == "sess_abc123"
    assert isinstance(item["confidence"], Decimal)
    assert item["confidence"] == Decimal("0.94")
    triage = item["triage_result"]
    assert isinstance(triage["confidence"], Decimal)
    lat = item["collected_slots"]["latitude"]
    assert isinstance(lat, Decimal)
    # Image analysis gets compacted (no raw bytes, capped summary)
    assert set(item["image_analysis"].keys()) <= {
        "available",
        "summary",
        "scene",
        "confidence",
        "visual_triage",
        "consistency",
    }


def test_to_item_omits_none_optional_fields():
    session = Session(session_id="empty_sess")
    session.is_complete = True

    item = build_case_record_from_session(session).to_item()

    assert item["session_id"] == "empty_sess"
    assert item["is_complete"] is True
    assert item["user_turn_count"] == 0
    assert "triage_level" not in item
    assert "dispatch_target" not in item
    assert "final_report" not in item
    assert "image_analysis" not in item


def test_persist_case_if_configured_noop_when_env_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CASES_DYNAMODB_TABLE", raising=False)
    session = _make_session()

    # Should not raise and not attempt any DynamoDB call.
    persist_case_if_configured(session, final_report="r")


def test_persist_case_if_configured_calls_store(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CASES_DYNAMODB_TABLE", "TestCases")

    session = _make_session()
    captured: Dict[str, Any] = {}

    class _FakeStore:
        def put(self, record: CaseRecord) -> Dict[str, Any]:
            captured["record"] = record
            return record.to_item()

    with patch("services.case_store.get_case_store", return_value=_FakeStore()):
        persist_case_if_configured(session, final_report="FINAL")

    assert "record" in captured
    assert captured["record"].session_id == "sess_abc123"
    assert captured["record"].final_report == "FINAL"


def test_persist_case_if_configured_swallows_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CASES_DYNAMODB_TABLE", "TestCases")
    session = _make_session()

    class _BrokenStore:
        def put(self, record: CaseRecord) -> Dict[str, Any]:
            raise RuntimeError("boom")

    with patch("services.case_store.get_case_store", return_value=_BrokenStore()):
        # Must not raise; orchestrator should never break from storage failures.
        persist_case_if_configured(session, final_report="r")
