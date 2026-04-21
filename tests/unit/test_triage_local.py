"""Yerel XLM-R triage servisinin conservative karar kurallarinin birim testleri."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.triage_local_service import (  # noqa: E402
    LocalTriageService,
    _count_user_turns,
    _flatten_history,
    _last_user_text,
)


def test_flatten_history_prepends_role_tags():
    history = [
        {"role": "user", "text": "basim agriyor"},
        {"role": "assistant", "text": "ates var mi?"},
        {"role": "user", "text": "37.5"},
    ]
    out = _flatten_history(history, max_turns=10)
    assert out == "[USER] basim agriyor [ASSISTANT] ates var mi? [USER] 37.5"


def test_flatten_history_skips_empty_and_trims():
    history = [{"role": "user", "text": ""}, {"role": "user", "text": "   "}]
    out = _flatten_history(history, max_turns=10)
    assert out == "[USER]"


def test_count_user_turns():
    hist = [
        {"role": "user", "text": "a"},
        {"role": "assistant", "text": "b"},
        {"role": "user", "text": "c"},
    ]
    assert _count_user_turns(hist) == 2


def test_last_user_text():
    hist = [
        {"role": "user", "text": "first"},
        {"role": "assistant", "text": "hello"},
        {"role": "user", "text": "second"},
    ]
    assert _last_user_text(hist) == "second"


@pytest.fixture
def svc() -> LocalTriageService:
    s = LocalTriageService()
    # Model yuklenmesin; yalnizca karar fonksiyonunu test ediyoruz.
    s._triage_labels = ["CRITICAL", "URGENT", "NON_URGENT"]
    s._category_labels = ["medical", "fire", "crime", "other"]
    s._quality_labels = ["meaningful", "gibberish", "out_of_scope"]
    return s


# === conservative rules ===

def test_conservative_rules_downgrades_critical_without_redflag_on_first_turn(svc):
    probs = [0.72, 0.2, 0.08]
    label, conf, _ = svc._apply_conservative_rules(probs, redflag_prob=0.1, user_turns=1)
    assert label == "URGENT"
    assert 0.0 < conf <= 1.0


def test_conservative_rules_allows_critical_when_redflag_fires_early(svc):
    probs = [0.72, 0.2, 0.08]
    label, _, _ = svc._apply_conservative_rules(probs, redflag_prob=0.85, user_turns=1)
    assert label == "CRITICAL"


def test_conservative_rules_allows_critical_after_min_turns(svc):
    probs = [0.72, 0.2, 0.08]
    label, _, _ = svc._apply_conservative_rules(probs, redflag_prob=0.1, user_turns=3)
    assert label == "CRITICAL"


def test_conservative_rules_downgrade_to_non_urgent_when_low_confidence(svc):
    probs = [0.2, 0.35, 0.45]
    label, _, _ = svc._apply_conservative_rules(probs, redflag_prob=0.1, user_turns=3)
    assert label == "NON_URGENT"


def test_conservative_rules_urgent_when_critical_confidence_below_threshold(svc):
    probs = [0.58, 0.3, 0.12]
    label, _, _ = svc._apply_conservative_rules(probs, redflag_prob=0.1, user_turns=3)
    assert label == "URGENT"


# === input_quality floor ===

def test_conservative_rules_gibberish_forces_non_urgent(svc):
    # Model CRITICAL diyor ama input gibberish -> NON_URGENT
    probs = [0.9, 0.05, 0.05]
    label, _, dbg = svc._apply_conservative_rules(
        probs, redflag_prob=0.1, user_turns=3, input_quality="gibberish",
    )
    assert label == "NON_URGENT"
    assert any("gibberish" in r for r in dbg["reasons"])


def test_conservative_rules_out_of_scope_forces_non_urgent(svc):
    probs = [0.85, 0.1, 0.05]
    label, _, _ = svc._apply_conservative_rules(
        probs, redflag_prob=0.1, user_turns=3, input_quality="out_of_scope",
    )
    assert label == "NON_URGENT"


def test_conservative_rules_gibberish_but_redflag_high_keeps_critical(svc):
    # input gibberish ama red_flag yuksek -> floor bypassed
    probs = [0.9, 0.05, 0.05]
    label, _, _ = svc._apply_conservative_rules(
        probs, redflag_prob=0.9, user_turns=3, input_quality="gibberish",
    )
    assert label == "CRITICAL"


# === NON_URGENT regex floor ===

def test_conservative_rules_nonurgent_regex_floor(svc):
    probs = [0.8, 0.15, 0.05]
    label, _, dbg = svc._apply_conservative_rules(
        probs, redflag_prob=0.1, user_turns=3, nonurgent_floor=True,
    )
    assert label == "NON_URGENT"
    assert any("nonurgent_regex_floor" in r for r in dbg["reasons"])


# === panic fusion ===

def test_conservative_rules_panic_high_relaxes_threshold(svc):
    # conf 0.62 < 0.70 normalde URGENT olur ama panic_high -> threshold 0.60'a iner
    probs = [0.62, 0.3, 0.08]
    label, _, dbg = svc._apply_conservative_rules(
        probs, redflag_prob=0.1, user_turns=3, panic_score=0.85,
    )
    assert label == "CRITICAL"
    assert any("panic_high" in r for r in dbg["reasons"])


def test_conservative_rules_panic_low_downgrades_critical(svc):
    # panic yok, red_flag dusuk -> CRITICAL yuksek confidence olsa bile URGENT'a dusur
    probs = [0.78, 0.15, 0.07]
    label, _, dbg = svc._apply_conservative_rules(
        probs, redflag_prob=0.1, user_turns=3, panic_score=0.1,
    )
    assert label == "URGENT"
    assert any("panic_low" in r for r in dbg["reasons"])


def test_conservative_rules_panic_low_with_redflag_keeps_critical(svc):
    probs = [0.78, 0.15, 0.07]
    label, _, _ = svc._apply_conservative_rules(
        probs, redflag_prob=0.9, user_turns=3, panic_score=0.1,
    )
    assert label == "CRITICAL"


# === predict_from_history ===

def test_predict_from_history_returns_empty_when_model_missing(svc, tmp_path):
    svc.model_dir = tmp_path / "does-not-exist"
    out = svc.predict_from_history([{"role": "user", "text": "merhaba"}])
    assert out["triage_level"] == "URGENT"
    assert out["red_flags"] == []
    assert out["response_text"] == ""


def test_predict_from_history_uses_guardrails_with_mocked_model(svc):
    svc.model_dir = Path("/virtual")
    # Ilk kullanici turu, red flag dusuk -> CRITICAL downgrade olmali
    raw1 = ([0.78, 0.18, 0.04], [0.9, 0.05, 0.03, 0.02], 0.1, [1.0, 0.0, 0.0])
    # Red flag yuksek -> CRITICAL izinli
    raw2 = ([0.82, 0.15, 0.03], [0.9, 0.05, 0.03, 0.02], 0.9, [1.0, 0.0, 0.0])
    with patch.object(svc.__class__, "is_available", property(lambda self: True)), \
         patch.object(svc, "_predict_raw", return_value=raw1):
        out = svc.predict_from_history([{"role": "user", "text": "kolum agriyor"}])
        assert out["triage_level"] == "URGENT"
        assert out["category"] == "medical"
        assert out["red_flags"] == []
        assert out["input_quality"] == "meaningful"

    with patch.object(svc.__class__, "is_available", property(lambda self: True)), \
         patch.object(svc, "_predict_raw", return_value=raw2):
        out = svc.predict_from_history([{"role": "user", "text": "nefes alamiyor"}])
        assert out["triage_level"] == "CRITICAL"
        assert "critical_signal" in out["red_flags"]


def test_predict_from_history_gibberish_quality_forces_non_urgent(svc):
    svc.model_dir = Path("/virtual")
    raw = ([0.95, 0.03, 0.02], [0.2, 0.1, 0.2, 0.5], 0.1, [0.05, 0.9, 0.05])
    with patch.object(svc.__class__, "is_available", property(lambda self: True)), \
         patch.object(svc, "_predict_raw", return_value=raw):
        out = svc.predict_from_history([
            {"role": "user", "text": "asdasd asdasd pwpw"},
            {"role": "assistant", "text": "acil durum nedir?"},
            {"role": "user", "text": "asdasd sdfsdf"},
        ])
        assert out["triage_level"] == "NON_URGENT"
        assert out["input_quality"] == "gibberish"
