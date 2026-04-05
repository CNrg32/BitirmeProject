from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from main import simulate_fallback_for_session
from orchestrator.session import get_session_store


def test_simulate_fallback_for_session_uncertain_sets_fallback_pending():
    store = get_session_store()
    s = store.create(language="tr")

    out = simulate_fallback_for_session(
        session_id=s.session_id,
        text="belirsiz bir durum",
        scenario="uncertain",
    )

    assert out["chatbot_mode"] == "fallback"
    assert out["dispatch_status"] == "FALLBACK_PENDING"
    assert out["fallback_options"] == ["medical", "crime", "fire"]


def test_simulate_fallback_for_session_valid_stays_normal():
    store = get_session_store()
    s = store.create(language="tr")

    out = simulate_fallback_for_session(
        session_id=s.session_id,
        text="yangin var",
        scenario="valid",
    )

    assert out["chatbot_mode"] == "normal"
    assert isinstance(out["fallback_options"], list)
