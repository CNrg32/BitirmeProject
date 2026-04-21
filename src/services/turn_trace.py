"""
Step-by-step conversation tracing to stdout (console).

Enable with either environment variable:
  TURN_TRACE=1
  CONVERSATION_TRACE=1

Safe for local dev; default off so production logs stay quiet.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def is_turn_trace_enabled() -> bool:
    for key in ("TURN_TRACE", "CONVERSATION_TRACE"):
        v = (os.environ.get(key) or "").strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
    return False


def _trunc(text: str, max_len: int = 320) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _safe_json(obj: Any, max_len: int = 600) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        s = str(obj)
    return _trunc(s, max_len)


def trace_print(lines: str) -> None:
    if not is_turn_trace_enabled():
        return
    print(lines, flush=True)


def trace_banner(title: str, session_id: Optional[str] = None) -> None:
    if not is_turn_trace_enabled():
        return
    sid = f" session={session_id}" if session_id else ""
    trace_print(f"\n{'=' * 72}\n  TURN TRACE | {title}{sid}\n{'=' * 72}")


def trace_step(step: str, detail: Optional[str] = None) -> None:
    if not is_turn_trace_enabled():
        return
    msg = f"  ▸ [{step}]"
    if detail:
        msg += f" {detail}"
    trace_print(msg)


def trace_kv(label: str, data: Dict[str, Any]) -> None:
    if not is_turn_trace_enabled():
        return
    trace_print(f"  ▸ [{label}]")
    for k, v in data.items():
        trace_print(f"      · {k}: {_safe_json(v) if not isinstance(v, str) else _trunc(v)}")


def trace_llm_response(
    *,
    provider: str,
    model: str,
    task: str,
    elapsed_s: float,
    parsed: Dict[str, Any],
) -> None:
    if not is_turn_trace_enabled():
        return
    trace_print(
        f"  ▸ [LLM yanıt] provider={provider} | model={model} | task={task} | "
        f"süre={elapsed_s:.2f}s"
    )
    slots = parsed.get("extracted_slots") or {}
    slot_keys = list(slots.keys()) if isinstance(slots, dict) else []
    trace_print(
        "      · parse: "
        f"triage_level={parsed.get('triage_level')} | category={parsed.get('category')} | "
        f"is_complete={parsed.get('is_complete')} | "
        f"dispatch_action={parsed.get('dispatch_action')} | "
        f"legal_close={parsed.get('legal_close')} | "
        f"confidence={parsed.get('confidence')} | "
        f"red_flags={parsed.get('red_flags')}"
    )
    if slot_keys:
        trace_print(f"      · extracted_slots keys: {slot_keys}")
    rt = parsed.get("response_text") or ""
    if rt:
        trace_print(f"      · response_text: {_trunc(rt)}")


def trace_orchestrator_outcome(
    *,
    triage_level: str,
    category: str,
    dispatch_status: str,
    dispatch_target: Optional[str],
    is_complete: bool,
    user_turn_count: int,
) -> None:
    if not is_turn_trace_enabled():
        return
    trace_print(
        "  ▸ [Orchestrator son durum] "
        f"triage_level={triage_level} | category={category} | "
        f"dispatch_status={dispatch_status} | dispatch_target={dispatch_target} | "
        f"is_complete={is_complete} | user_turns={user_turn_count}"
    )


def monotonic_start() -> float:
    return time.monotonic()
