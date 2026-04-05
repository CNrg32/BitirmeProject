"""Unit tests for orchestrator: start_session, handle_message branches (mocked deps)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from orchestrator.session import Session, SessionStore, get_session_store


class TestStartSession:
    def test_start_session_returns_id_and_greeting(self):
        from orchestrator.orchestrator import start_session
        with patch("orchestrator.orchestrator.get_session_store") as mock_store:
            store = SessionStore(ttl_seconds=3600)
            mock_store.return_value = store
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff\xfb"):
                out = start_session(language=None)
        assert "session_id" in out
        assert "greeting" in out
        assert out["greeting_audio_b64"] is not None

    def test_start_session_with_language_locks(self):
        from orchestrator.orchestrator import start_session
        with patch("orchestrator.orchestrator.get_session_store") as mock_store:
            store = SessionStore(ttl_seconds=3600)
            mock_store.return_value = store
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff\xfb"):
                out = start_session(language="tr")
        assert "Acil" in out["greeting"] or "acil" in out["greeting"].lower()
        sess = store.get(out["session_id"])
        assert sess is not None
        assert sess.language_locked is True


class TestHandleMessage:
    def test_handle_message_unknown_session_returns_error(self):
        from orchestrator.orchestrator import handle_message
        with patch("orchestrator.orchestrator.get_session_store") as mock_store:
            mock_store.return_value = SessionStore(ttl_seconds=3600)
            out = handle_message("nonexistent-sid", user_text="Help")
        assert "error" in out
        assert "not found" in out["error"].lower() or "expired" in out["error"].lower()

    def test_handle_message_completed_session_returns_done_msg(self):
        from orchestrator.orchestrator import handle_message
        store = SessionStore(ttl_seconds=3600)
        s = store.create(language="en")
        s.is_complete = True
        s.triage_result = {"triage_level": "URGENT", "category": "medical"}
        with patch("orchestrator.orchestrator.get_session_store", return_value=store):
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff"):
                out = handle_message(s.session_id, user_text="Another message")
        assert out.get("is_complete") is True
        assert "complete" in out.get("assistant_text", "").lower()

    def test_handle_message_no_text_no_image_returns_prompt(self):
        from orchestrator.orchestrator import handle_message
        store = SessionStore(ttl_seconds=3600)
        s = store.create(language="en")
        s.messages.append({"role": "assistant", "text": "Hi"})
        with patch("orchestrator.orchestrator.get_session_store", return_value=store):
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff"):
                out = handle_message(s.session_id, user_text=None, audio_bytes=None, image_bytes=None)
        assert "assistant_text" in out
        assert "describe" in out["assistant_text"].lower() or "emergency" in out["assistant_text"].lower()

    def test_handle_message_with_image_runs_image_analysis(self):
        from orchestrator.orchestrator import handle_message
        store = SessionStore(ttl_seconds=3600)
        s = store.create(language="en")
        s.messages.append({"role": "assistant", "text": "Hi"})
        with patch("orchestrator.orchestrator.get_session_store", return_value=store):
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff"):
                with patch("services.image_service.analyze_image", return_value={"available": True, "summary": "Image received"}):
                        out = handle_message(
                            s.session_id,
                            user_text=None,
                            image_bytes=b"\xff\xd8\xff",
                        )
        assert "assistant_text" in out
        assert store.get(s.session_id).image_bytes is not None

    def test_handle_message_gibberish_first_time_requests_clarification(self):
        from orchestrator.orchestrator import handle_message

        store = SessionStore(ttl_seconds=3600)
        s = store.create(language="tr")
        s.messages.append({"role": "assistant", "text": "Merhaba"})

        with patch("orchestrator.orchestrator.get_session_store", return_value=store):
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff"):
                with patch("orchestrator.orchestrator._is_gibberish_with_llm", return_value=True):
                    out = handle_message(s.session_id, user_text="akska")

        assert out.get("is_complete") is False
        msg = out.get("assistant_text", "").lower()
        assert ("anlayamad" in msg or "net" in msg or "describe" in msg or "understand" in msg)
        sess = store.get(s.session_id)
        assert sess is not None
        assert sess.troll_count == 1

    def test_handle_message_gibberish_second_time_closes_session(self):
        from orchestrator.orchestrator import handle_message

        store = SessionStore(ttl_seconds=3600)
        s = store.create(language="tr")
        s.messages.append({"role": "assistant", "text": "Merhaba"})
        s.troll_count = 1

        with patch("orchestrator.orchestrator.get_session_store", return_value=store):
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff"):
                out = handle_message(s.session_id, user_text="qwerqwer")

        assert out.get("is_complete") is True
        assert "kapat" in out.get("assistant_text", "").lower() or "session" in out.get("assistant_text", "").lower()

    def test_handle_message_alphanumeric_noise_triggers_clarification(self):
        from orchestrator.orchestrator import handle_message

        store = SessionStore(ttl_seconds=3600)
        s = store.create(language="tr")
        s.messages.append({"role": "assistant", "text": "Merhaba"})

        with patch("orchestrator.orchestrator.get_session_store", return_value=store):
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff"):
                with patch("orchestrator.orchestrator._is_gibberish_with_llm", return_value=True):
                    out = handle_message(s.session_id, user_text="qwe12e")

        assert out.get("is_complete") is False
        sess = store.get(s.session_id)
        assert sess is not None
        assert sess.troll_count == 1

    def test_hayir_is_not_gibberish_token(self):
        from orchestrator.orchestrator import _is_gibberish

        assert _is_gibberish("hayır") is False
        assert _is_gibberish("hayir") is False

    def test_handle_message_hayir_does_not_close_session(self):
        from orchestrator.orchestrator import handle_message

        store = SessionStore(ttl_seconds=3600)
        s = store.create(language="tr")
        s.messages.append({"role": "assistant", "text": "Bacağınızı hareket ettirdiğinizde ağrı artıyor mu?"})

        with patch("orchestrator.orchestrator.get_session_store", return_value=store):
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff"):
                out = handle_message(s.session_id, user_text="hayır")

        assert out.get("is_complete") is False
        sess = store.get(s.session_id)
        assert sess is not None
        assert sess.troll_count == 0

    def test_handle_message_returns_nearby_places_when_location_exists(self):
        from orchestrator.orchestrator import handle_message

        store = SessionStore(ttl_seconds=3600)
        s = store.create(language="en")
        s.messages.append({"role": "assistant", "text": "Hi"})

        mock_llm = MagicMock()
        mock_llm.is_available = True
        mock_llm.chat.side_effect = [
            {
                "category": "medical",
                "triage_level": "URGENT",
                "confidence": 0.9,
                "red_flags": [],
            },
            {
                "response_text": "Help is on the way.",
                "extracted_slots": {},
                "triage_level": "URGENT",
                "category": "medical",
                "is_complete": False,
                "red_flags": [],
            },
        ]

        with patch("orchestrator.orchestrator.get_session_store", return_value=store):
            with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff"):
                with patch("orchestrator.orchestrator._is_gibberish_with_llm", return_value=False):
                    with patch("services.llm_service.get_llm_service", return_value=mock_llm):
                        with patch(
                            "services.nearby_places_service.get_nearby_places",
                            return_value=[{"id": "h1", "type": "hospital", "name": "A Hospital"}],
                        ):
                            out = handle_message(
                                s.session_id,
                                user_text="My father has chest pain",
                                latitude=41.0,
                                longitude=29.0,
                            )

        assert out.get("nearby_places") == [{"id": "h1", "type": "hospital", "name": "A Hospital"}]
