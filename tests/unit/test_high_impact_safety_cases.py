"""High-impact safety tests for final-report critical scenarios.

These tests directly cover the high-score scenarios documented in the final report:
- Fallback safety flow
- Intent confirmation policy presence
- Sentiment escalation boundaries
- Dispatch lock behavior
- Timeout / silent dispatch behavior
- Multilingual robustness checks
- Image-text inconsistency handling
- Adversarial noise handling
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from main import simulate_fallback_for_session
from orchestrator.session import Session, SessionStore, can_redispatch, get_session_store
from orchestrator.orchestrator import _is_gibberish, _merge_sentiment_into_triage, handle_message
from services.image_service import analyze_consistency
from services.llm_prompt_config import _get_dialog_system_prompt
from services.text_analyze_mock import analyze_text_mock


def test_high_impact_fallback_uncertain_enables_manual_buttons():
    store = get_session_store()
    s = store.create(language="tr")

    out = simulate_fallback_for_session(
        session_id=s.session_id,
        text="belirsiz durum",
        scenario="uncertain",
    )

    assert out["chatbot_mode"] == "fallback"
    assert out["dispatch_status"] == "FALLBACK_PENDING"
    assert out["fallback_options"] == ["medical", "crime", "fire"]


def test_high_impact_fallback_fail_enables_manual_buttons():
    store = get_session_store()
    s = store.create(language="en")

    out = simulate_fallback_for_session(
        session_id=s.session_id,
        text="service failed",
        scenario="fail",
    )

    assert out["chatbot_mode"] == "fallback"
    assert out["dispatch_status"] == "FALLBACK_PENDING"
    assert out["fallback_options"] == ["medical", "crime", "fire"]


def test_high_impact_intent_confirmation_policy_present_in_prompt():
    prompt = _get_dialog_system_prompt().lower()
    assert "intent confirmation" in prompt
    assert "fallback" in prompt
    assert "strict emergency scope" in prompt


def test_high_impact_sentiment_boundary_upgrades_when_confident():
    session = Session(session_id="s1")
    session.sentiment_result = {
        "triage_level": "URGENT",
        "confidence": 0.72,
        "panic_level": "high",
    }
    triage = {"triage_level": "NON_URGENT", "confidence": 0.51}

    merged = _merge_sentiment_into_triage(session, triage)

    assert merged["triage_level"] == "URGENT"
    assert merged.get("sentiment_override") is True


def test_high_impact_sentiment_boundary_does_not_upgrade_when_low_confidence():
    session = Session(session_id="s2")
    session.sentiment_result = {
        "triage_level": "CRITICAL",
        "confidence": 0.49,
        "panic_level": "high",
    }
    triage = {"triage_level": "URGENT", "confidence": 0.6}

    merged = _merge_sentiment_into_triage(session, triage)

    assert merged["triage_level"] == "URGENT"
    assert merged.get("sentiment_override") is not True


def test_high_impact_dispatch_lock_blocks_redispatch_before_ttl():
    session = Session(session_id="lock-1")
    session.dispatch_status = "DISPATCHED"
    session.dispatch_timestamp = time.time() - 60  # 1 minute ago

    assert can_redispatch(session, redispatch_ttl_seconds=48 * 3600) is False


def test_high_impact_dispatch_lock_allows_redispatch_after_ttl():
    session = Session(session_id="lock-2")
    session.dispatch_status = "DISPATCHED"
    session.dispatch_timestamp = time.time() - (49 * 3600)  # 49 hours ago

    assert can_redispatch(session, redispatch_ttl_seconds=48 * 3600) is True


def test_high_impact_timeout_urgent_triggers_silent_dispatch():
    store = SessionStore(ttl_seconds=3600)
    s = store.create(language="en")
    s.messages.append({"role": "assistant", "text": "What is your emergency?"})
    s.triage_result = {"triage_level": "URGENT", "category": "medical"}
    s.dispatch_status = "PENDING"
    s.timeout_deadline = time.time() - 1  # already expired
    s.last_user_activity_at = time.time() - 181

    with patch("orchestrator.orchestrator.get_session_store", return_value=store):
        with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff"):
            out = handle_message(s.session_id, user_text="hello")

    assert out.get("is_complete") is True
    assert out.get("dispatch_status") == "SILENT_DISPATCHED"


def test_high_impact_multilingual_keywords_map_to_stable_category():
    tr_out = analyze_text_mock("Nefes alamiyor, yardim edin", scenario="valid")
    en_out = analyze_text_mock("He cannot breathe and needs help", scenario="valid")

    assert tr_out["status"] == "valid"
    assert en_out["status"] == "valid"
    assert tr_out["category"] == "medical"
    assert en_out["category"] == "medical"


def test_high_impact_image_text_inconsistency_is_flagged():
    image_result = {
        "detected_class": "NormalVideos",
        "mapped_category": "other",
        "confidence": 0.93,
    }

    consistency = analyze_consistency(
        image_result=image_result,
        text_category="medical",
        text_triage_level="CRITICAL",
    )

    assert consistency["is_consistent"] is False
    assert consistency["possible_fake"] is True


def test_high_impact_adversarial_noise_is_rejected_by_filter():
    assert _is_gibberish("qwerqwer!!111") is True
    assert _is_gibberish("My father is not breathing") is False
