from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Session:
    session_id: str
    language: Optional[str] = None
    collected_slots: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, str]] = field(default_factory=list)
    triage_result: Optional[Dict[str, Any]] = None
    sentiment_result: Optional[Dict[str, Any]] = None
    is_complete: bool = False
    asked_questions: Set[str] = field(default_factory=set)
    text_en_accumulated: str = ""
    meta: Dict[str, Any] = field(default_factory=lambda: {
        "deaths": 0, "potential_death": 0, "false_alarm": 0
    })
    image_analysis: Optional[Dict[str, Any]] = None
    language_locked: bool = False
    image_bytes: Optional[bytes] = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._sessions: Dict[str, Session] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def create(self, language: Optional[str] = None) -> Session:
        with self._lock:
            sid = uuid.uuid4().hex[:12]
            session = Session(session_id=sid, language=language)
            self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.created_at > self._ttl:
                del self._sessions[session_id]
                return None
            session.updated_at = time.time()
            return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if now - s.created_at > self._ttl
            ]
            for sid in expired:
                del self._sessions[sid]
        return len(expired)


_store = SessionStore()


def get_session_store() -> SessionStore:
    return _store
