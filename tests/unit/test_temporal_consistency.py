"""orchestrator._apply_temporal_consistency birim testleri."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from orchestrator.orchestrator import _apply_temporal_consistency  # noqa: E402
from orchestrator.session import Session  # noqa: E402


def _mk_session() -> Session:
    return Session(session_id="t1")


def test_first_critical_low_conf_smoothed_to_urgent():
    s = _mk_session()
    out = _apply_temporal_consistency(s, {
        "triage_level": "CRITICAL",
        "confidence": 0.70,
        "red_flags": [],
    })
    # Tek basina gelen CRITICAL + conf<0.80 + red_flag yok -> URGENT
    # (Gecmis yokken recent<2 oldugu icin smoothing dev dey dusmez! Kural len>=2)
    # Bu testte history uzunlugu 1 oldugundan smoothing uygulanmamali.
    assert out["triage_level"] == "CRITICAL"


def test_first_critical_after_two_urgents_smoothed():
    s = _mk_session()
    # Iki URGENT gecmis
    _apply_temporal_consistency(s, {"triage_level": "URGENT", "confidence": 0.6, "red_flags": []})
    _apply_temporal_consistency(s, {"triage_level": "URGENT", "confidence": 0.6, "red_flags": []})
    # Simdi ilk CRITICAL, conf orta duzey, red_flag yok -> smoothing URGENT'a dusurmeli
    out = _apply_temporal_consistency(s, {
        "triage_level": "CRITICAL", "confidence": 0.72, "red_flags": [],
    })
    assert out["triage_level"] == "URGENT"
    assert out.get("temporal_smoothed") is True


def test_critical_with_redflag_not_smoothed():
    s = _mk_session()
    _apply_temporal_consistency(s, {"triage_level": "URGENT", "confidence": 0.6, "red_flags": []})
    _apply_temporal_consistency(s, {"triage_level": "URGENT", "confidence": 0.6, "red_flags": []})
    out = _apply_temporal_consistency(s, {
        "triage_level": "CRITICAL", "confidence": 0.6, "red_flags": ["critical_signal"],
    })
    assert out["triage_level"] == "CRITICAL"
    assert out.get("temporal_smoothed") is not True


def test_critical_with_high_confidence_not_smoothed():
    s = _mk_session()
    _apply_temporal_consistency(s, {"triage_level": "URGENT", "confidence": 0.6, "red_flags": []})
    _apply_temporal_consistency(s, {"triage_level": "URGENT", "confidence": 0.6, "red_flags": []})
    out = _apply_temporal_consistency(s, {
        "triage_level": "CRITICAL", "confidence": 0.88, "red_flags": [],
    })
    assert out["triage_level"] == "CRITICAL"


def test_second_critical_confirms_after_smoothing():
    s = _mk_session()
    _apply_temporal_consistency(s, {"triage_level": "URGENT", "confidence": 0.6, "red_flags": []})
    first = _apply_temporal_consistency(s, {
        "triage_level": "CRITICAL", "confidence": 0.72, "red_flags": [],
    })
    assert first["triage_level"] == "URGENT"
    # Ikinci CRITICAL tahmini - recent icinde CRITICAL var (prev_critical>=1)
    second = _apply_temporal_consistency(s, {
        "triage_level": "CRITICAL", "confidence": 0.72, "red_flags": [],
    })
    assert second["triage_level"] == "CRITICAL"


def test_temporal_history_capped():
    s = _mk_session()
    for _ in range(20):
        _apply_temporal_consistency(s, {
            "triage_level": "URGENT", "confidence": 0.6, "red_flags": [],
        })
    assert len(s.triage_history) <= 8


def test_sentiment_override_not_smoothed():
    s = _mk_session()
    _apply_temporal_consistency(s, {"triage_level": "URGENT", "confidence": 0.6, "red_flags": []})
    _apply_temporal_consistency(s, {"triage_level": "URGENT", "confidence": 0.6, "red_flags": []})
    out = _apply_temporal_consistency(s, {
        "triage_level": "CRITICAL", "confidence": 0.72, "red_flags": [],
        "sentiment_override": True,
    })
    assert out["triage_level"] == "CRITICAL"
