from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.text_analyze_mock import analyze_text_mock


def test_text_analyze_mock_fail():
    out = analyze_text_mock(text="test", scenario="fail")
    assert out["status"] == "fail"
    assert out["category"] is None


def test_text_analyze_mock_uncertain():
    out = analyze_text_mock(text="belirsiz", scenario="uncertain")
    assert out["status"] == "uncertain"
    assert out["category"] == "other"
    assert out["triage_level"] == "URGENT"


def test_text_analyze_mock_valid_medical():
    out = analyze_text_mock(text="Nefes alamiyor ve kanamasi var", scenario="valid")
    assert out["status"] == "valid"
    assert out["category"] == "medical"
    assert out["triage_level"] in ("CRITICAL", "URGENT", "NON_URGENT")
