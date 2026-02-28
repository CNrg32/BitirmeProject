"""Unit tests for report_composer.compose_report."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from orchestrator.report_composer import compose_report


def test_compose_report_contains_triage_and_category():
    triage_result = {"triage_level": "CRITICAL", "category": "medical", "red_flags": [], "confidence": 0.9}
    slots = {"age": 63, "chief_complaint": "Not breathing"}
    report = compose_report(triage_result, slots)
    assert "CRITICAL" in report
    assert "medical" in report or "Medical" in report
    assert "Ambulance" in report or "EMS" in report


def test_compose_report_contains_instructions():
    triage_result = {"triage_level": "URGENT", "category": "medical", "red_flags": [], "confidence": None}
    slots = {}
    report = compose_report(triage_result, slots)
    assert "Stay calm" in report or "calm" in report.lower()
    assert "•" in report or "instructions" in report.lower()


def test_compose_report_includes_slots():
    triage_result = {"triage_level": "NON_URGENT", "category": "other", "red_flags": []}
    slots = {"age": 25, "severity_1_10": 4, "location_hint": "home"}
    report = compose_report(triage_result, slots)
    assert "25" in report
    assert "4" in report
    assert "home" in report


def test_compose_report_red_flags_section():
    triage_result = {
        "triage_level": "CRITICAL",
        "category": "medical",
        "red_flags": ["not breathing", "unconscious"],
        "confidence": None,
    }
    slots = {}
    report = compose_report(triage_result, slots)
    assert "RED FLAGS" in report
    assert "not breathing" in report
    assert "unconscious" in report


def test_compose_report_includes_gps_in_slots():
    triage_result = {"triage_level": "URGENT", "category": "other", "red_flags": [], "confidence": None}
    slots = {"latitude": 41.0082, "longitude": 28.9784, "age": 30}
    report = compose_report(triage_result, slots)
    assert "41.008200" in report or "41.01" in report
    assert "28.978400" in report or "28.98" in report
    assert "maps.google.com" in report


def test_compose_report_image_analysis_section_when_available():
    triage_result = {"triage_level": "URGENT", "category": "fire", "red_flags": [], "confidence": None}
    slots = {}
    image_analysis = {
        "available": True,
        "summary": "Fire scene detected",
        "classification": {"detected_class": "Fire", "confidence": 0.95, "dispatch_units": ["Fire"]},
        "consistency": {"consistency_detail": "CONSISTENT", "consistency_score": 0.9},
    }
    report = compose_report(triage_result, slots, image_analysis=image_analysis)
    assert "Image Analysis" in report
    assert "Fire" in report


def test_compose_report_image_with_top3_and_possible_fake():
    triage_result = {"triage_level": "CRITICAL", "category": "medical", "red_flags": [], "confidence": None}
    slots = {}
    image_analysis = {
        "available": True,
        "summary": "Scene with alternatives",
        "classification": {
            "detected_class": "Accident",
            "confidence": 0.7,
            "dispatch_units": ["EMS"],
            "top3": [
                {"class": "Accident", "confidence": 0.7},
                {"class": "Fire", "confidence": 0.2},
                {"class": "Other", "confidence": 0.1},
            ],
        },
        "consistency": {
            "consistency_detail": "INCONSISTENT",
            "consistency_score": 0.3,
            "possible_fake": True,
            "risk_notes": ["Image may not match description"],
        },
    }
    report = compose_report(triage_result, slots, image_analysis=image_analysis)
    assert "Alternatives" in report
    assert "WARNING" in report
    assert "Image may not match" in report
    assert "Image may not match description" in report
