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
    
    # === FAZ 1-2: Groq Triage & State Management ===
    # Initial triage result from Groq (turn 1)
    initial_triage: Optional[Dict[str, Any]] = None
    
    # Troll/noise detection counter
    troll_count: int = 0
    
    # Slot filling tracking: {slot_name: attempt_count}
    slot_attempt_counts: Dict[str, int] = field(default_factory=dict)
    
    # Whether there's a pending contradiction to clarify
    contradiction_pending: bool = False
    
    # Dispatch state machine: PENDING, DISPATCHED, SILENT_DISPATCHED, FALLBACK_PENDING, CANCELLED
    dispatch_status: str = "PENDING"
    
    # Which service was dispatched to: 'police', 'medical', 'fire' or None
    dispatch_target: Optional[str] = None
    
    # Timestamp when dispatch occurred (for 48-hour redispatch prevention)
    dispatch_timestamp: Optional[float] = None
    
    # Absolute timestamp when 3-minute timeout will trigger
    timeout_deadline: Optional[float] = None
    
    # Last user message timestamp (for 3-min timeout calculation)
    last_user_activity_at: float = field(default_factory=time.time)
    
    # Whether session was resumed after timeout
    resumed_after_timeout: bool = False
    
    # Whether in witness mode (caller is witness, not victim)
    witness_mode: bool = False
    
    # Whether waiting for post-dispatch update from user
    pending_update_after_dispatch: bool = False
    
    # Last N messages for context (managed: keep only last 8-10 turns)
    message_history: List[Dict[str, str]] = field(default_factory=list)
    
    # FAZ 5: Track which slot is currently being asked for (2-attempt rule)
    pending_question_key: Optional[str] = None

    # Image-layer state: invalid/unclear image retry count and post-dispatch updates.
    image_attempt_count: int = 0
    image_updates: List[Dict[str, Any]] = field(default_factory=list)


def can_redispatch(session: Session, redispatch_ttl_seconds: int = 48 * 3600) -> bool:
    """
    Check if a session can be re-dispatched (dispatch lock).
    
    Args:
        session: The session to check
        redispatch_ttl_seconds: How long to prevent re-dispatch (default 48 hours)
    
    Returns:
        True if enough time has passed since last dispatch, False if still recent
    """
    if session.dispatch_status != "DISPATCHED" and session.dispatch_status != "SILENT_DISPATCHED":
        # No prior dispatch, can proceed
        return True
    
    if session.dispatch_timestamp is None:
        # Dispatch status set but no timestamp? Shouldn't happen, allow redispatch
        return True
    
    time_since_dispatch = time.time() - session.dispatch_timestamp
    return time_since_dispatch > redispatch_ttl_seconds


def truncate_message_history(session: Session, max_turns: int = 8) -> None:
    """
    Keep only the last N turns of message history (for Groq context limit ~8000 tokens).
    
    Args:
        session: The session to truncate
        max_turns: Maximum number of turns to keep (default 8, ~2k tokens)
    
    Modifies session.message_history in-place.
    """
    if len(session.message_history) > max_turns:
        # Keep only last N turns
        session.message_history = session.message_history[-max_turns:]


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
