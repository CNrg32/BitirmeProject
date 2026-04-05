from __future__ import annotations

from typing import Any, Dict


VALID_CATEGORIES = {"medical", "fire", "crime", "other"}
VALID_LEVELS = {"CRITICAL", "URGENT", "NON_URGENT"}


def analyze_text_mock(text: str, scenario: str = "valid") -> Dict[str, Any]:
    """
    Development-only TextAnalyze mock used for fallback strategy tests.

    Scenarios:
    - fail: Simulates model failure
    - uncertain: Returns uncertain result
    - valid: Returns deterministic triage result
    """
    scenario = (scenario or "valid").strip().lower()
    text = (text or "").strip()

    if scenario == "fail":
        return {
            "status": "fail",
            "error": "mock_failure",
            "category": None,
            "triage_level": None,
            "confidence": 0.0,
        }

    if scenario == "uncertain":
        return {
            "status": "uncertain",
            "error": None,
            "category": "other",
            "triage_level": "URGENT",
            "confidence": 0.35,
        }

    # valid
    lowered = text.lower()
    if any(k in lowered for k in ["kan", "nefes", "kalp", "bayıl", "bleed", "breath", "heart"]):
        category = "medical"
        triage = "CRITICAL"
        conf = 0.9
    elif any(k in lowered for k in ["yangın", "duman", "fire", "smoke", "alev"]):
        category = "fire"
        triage = "URGENT"
        conf = 0.82
    elif any(k in lowered for k in ["bıçak", "silah", "saldırı", "crime", "weapon", "assault"]):
        category = "crime"
        triage = "URGENT"
        conf = 0.8
    else:
        category = "other"
        triage = "NON_URGENT"
        conf = 0.7

    if category not in VALID_CATEGORIES or triage not in VALID_LEVELS:
        return {
            "status": "fail",
            "error": "invalid_mock_output",
            "category": None,
            "triage_level": None,
            "confidence": 0.0,
        }

    return {
        "status": "valid",
        "error": None,
        "category": category,
        "triage_level": triage,
        "confidence": conf,
    }
