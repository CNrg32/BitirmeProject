"""Unit tests for Session and SessionStore."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from orchestrator.session import Session, SessionStore, get_session_store


class TestSession:
    def test_session_creation(self):
        s = Session(session_id="test-123")
        assert s.session_id == "test-123"
        assert s.language is None
        assert s.collected_slots == {}
        assert s.messages == []
        assert s.is_complete is False
        assert s.text_en_accumulated == ""
        assert s.meta["deaths"] == 0

    def test_session_with_language(self):
        s = Session(session_id="x", language="tr")
        assert s.language == "tr"


class TestSessionStore:
    def test_create_returns_session(self):
        store = SessionStore(ttl_seconds=3600)
        s = store.create()
        assert isinstance(s, Session)
        assert len(s.session_id) == 12
        assert store.get(s.session_id) is not None

    def test_create_with_language(self):
        store = SessionStore(ttl_seconds=3600)
        s = store.create(language="en")
        assert s.language == "en"

    def test_get_unknown_returns_none(self):
        store = SessionStore(ttl_seconds=3600)
        assert store.get("nonexistent") is None

    def test_delete_removes_session(self):
        store = SessionStore(ttl_seconds=3600)
        s = store.create()
        sid = s.session_id
        store.delete(sid)
        assert store.get(sid) is None

    def test_ttl_expiry(self):
        store = SessionStore(ttl_seconds=1)
        s = store.create()
        sid = s.session_id
        assert store.get(sid) is not None
        time.sleep(1.1)
        assert store.get(sid) is None

    def test_cleanup_expired(self):
        store = SessionStore(ttl_seconds=0)
        s = store.create()
        sid = s.session_id
        # Force expiry by patching created_at
        s.created_at = time.time() - 10
        n = store.cleanup_expired()
        assert n >= 1
        assert store.get(sid) is None
