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
