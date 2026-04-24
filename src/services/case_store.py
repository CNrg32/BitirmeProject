"""DynamoDB-backed store for completed emergency sessions (cases).

Each record represents a single completed (or closed) session and captures
the conversation history, collected slots, triage result, dispatch outcome,
and the final report text. Records are keyed by ``session_id`` and written
once per session at finalize-time from ``orchestrator._reply``.

Enable by setting ``CASES_DYNAMODB_TABLE`` (and standard AWS creds / region).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseRecord:
    session_id: str
    created_at: str
    completed_at: str
    language: Optional[str]
    is_complete: bool
    user_turn_count: int
    dispatch_status: str
    dispatch_target: Optional[str]
    dispatch_timestamp: Optional[str]
    triage_level: Optional[str]
    category: Optional[str]
    confidence: Optional[float]
    red_flags: List[str]
    triage_result: Optional[Dict[str, Any]]
    collected_slots: Dict[str, Any]
    messages: List[Dict[str, str]]
    final_report: Optional[str]
    image_analysis: Optional[Dict[str, Any]] = None
    witness_mode: bool = False
    resumed_after_timeout: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_item(self) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "is_complete": self.is_complete,
            "user_turn_count": self.user_turn_count,
            "dispatch_status": self.dispatch_status,
            "red_flags": list(self.red_flags or []),
            "collected_slots": _to_dynamodb_value(self.collected_slots or {}),
            "messages": _to_dynamodb_value(self.messages or []),
            "witness_mode": bool(self.witness_mode),
            "resumed_after_timeout": bool(self.resumed_after_timeout),
            "meta": _to_dynamodb_value(self.meta or {}),
        }
        if self.language:
            item["language"] = self.language
        if self.dispatch_target:
            item["dispatch_target"] = self.dispatch_target
        if self.dispatch_timestamp:
            item["dispatch_timestamp"] = self.dispatch_timestamp
        if self.triage_level:
            item["triage_level"] = self.triage_level
        if self.category:
            item["category"] = self.category
        if self.confidence is not None:
            item["confidence"] = _to_dynamodb_value(self.confidence)
        if self.triage_result is not None:
            item["triage_result"] = _to_dynamodb_value(self.triage_result)
        if self.final_report:
            item["final_report"] = self.final_report
        if self.image_analysis is not None:
            item["image_analysis"] = _to_dynamodb_value(_compact_image(self.image_analysis))
        return item


def build_case_record_from_session(
    session: Any,
    *,
    final_report: Optional[str] = None,
) -> CaseRecord:
    """Snapshot a ``Session`` dataclass into a ``CaseRecord``.

    Only conversation text + structured state is captured; no raw audio or
    image bytes are persisted here (image bytes live in the dedicated image
    analysis table).
    """
    triage = dict(session.triage_result or {})
    triage_level = triage.get("triage_level")
    category = triage.get("category")
    confidence = triage.get("confidence")
    red_flags = list(triage.get("red_flags") or [])

    created_at_iso = _epoch_to_iso(getattr(session, "created_at", None))
    dispatch_ts_iso = _epoch_to_iso(getattr(session, "dispatch_timestamp", None))

    user_turn_count = sum(
        1 for m in (session.messages or []) if m.get("role") == "user"
    )

    return CaseRecord(
        session_id=session.session_id,
        created_at=created_at_iso,
        completed_at=datetime.now(timezone.utc).isoformat(),
        language=getattr(session, "language", None),
        is_complete=bool(getattr(session, "is_complete", False)),
        user_turn_count=user_turn_count,
        dispatch_status=getattr(session, "dispatch_status", "PENDING") or "PENDING",
        dispatch_target=getattr(session, "dispatch_target", None),
        dispatch_timestamp=dispatch_ts_iso,
        triage_level=triage_level,
        category=category,
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        red_flags=red_flags,
        triage_result=triage or None,
        collected_slots=dict(getattr(session, "collected_slots", {}) or {}),
        messages=list(getattr(session, "messages", []) or []),
        final_report=final_report,
        image_analysis=getattr(session, "image_analysis", None),
        witness_mode=bool(getattr(session, "witness_mode", False)),
        resumed_after_timeout=bool(getattr(session, "resumed_after_timeout", False)),
        meta=dict(getattr(session, "meta", {}) or {}),
    )


class CaseStore:
    def put(self, record: CaseRecord) -> Dict[str, Any]:
        raise NotImplementedError

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def list(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DynamoDBCaseStore(CaseStore):
    def __init__(
        self,
        *,
        table_name: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.table_name = table_name or os.environ.get("CASES_DYNAMODB_TABLE", "")
        if not self.table_name:
            raise RuntimeError(
                "CASES_DYNAMODB_TABLE must be set for case (session) storage."
            )

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for DynamoDB case storage. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        region = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        endpoint = endpoint_url or os.environ.get("DYNAMODB_ENDPOINT_URL") or None
        resource_kwargs: Dict[str, Any] = {}
        if region:
            resource_kwargs["region_name"] = region
        if endpoint:
            resource_kwargs["endpoint_url"] = endpoint

        self._dynamodb = boto3.resource("dynamodb", **resource_kwargs)
        self._table = self._dynamodb.Table(self.table_name)

    def put(self, record: CaseRecord) -> Dict[str, Any]:
        item = record.to_item()
        self._table.put_item(Item=item)
        return _normalize_item(item)

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        response = self._table.get_item(Key={"session_id": session_id})
        item = response.get("Item")
        return _normalize_item(item) if item else None

    def list(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        response = self._table.scan(Limit=limit)
        items = response.get("Items", [])
        items.sort(key=lambda i: i.get("completed_at", ""), reverse=True)
        return [_normalize_item(item) for item in items]


def _compact_image(image_analysis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not image_analysis:
        return None
    out: Dict[str, Any] = {}
    if "available" in image_analysis:
        out["available"] = bool(image_analysis.get("available"))
    summary = image_analysis.get("summary")
    if summary:
        out["summary"] = str(summary)[:2000]
    cls = image_analysis.get("classification")
    if isinstance(cls, dict):
        out["scene"] = cls.get("detected_class")
        out["confidence"] = cls.get("confidence")
    visual = image_analysis.get("visual_triage")
    if isinstance(visual, dict):
        out["visual_triage"] = {
            "triage_level": visual.get("triage_level"),
            "action": visual.get("action"),
            "visual_flags": list(visual.get("visual_flags") or []),
        }
    consistency = image_analysis.get("consistency")
    if isinstance(consistency, dict):
        out["consistency"] = {
            "consistency_score": consistency.get("consistency_score"),
            "possible_fake": consistency.get("possible_fake"),
        }
    return out or None


def _epoch_to_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _to_dynamodb_value(value: Any) -> Any:
    # Round-trip through JSON to coerce floats to Decimal (DynamoDB requirement)
    # and drop unsupported types (bytes, datetime, etc.).
    return json.loads(json.dumps(value, default=str), parse_float=Decimal)


def _normalize_item(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, list):
        return [_normalize_item(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_item(item) for key, item in value.items()}
    return value


def get_case_store() -> CaseStore:
    return DynamoDBCaseStore()


def persist_case_if_configured(
    session: Any,
    *,
    final_report: Optional[str] = None,
) -> None:
    """Fire-and-forget: write the session as a case record.

    Silently no-ops (with a warning log) when ``CASES_DYNAMODB_TABLE`` is
    unset or boto3 is unavailable, so the live orchestrator path is never
    blocked by storage errors.
    """
    if not os.environ.get("CASES_DYNAMODB_TABLE"):
        return
    try:
        record = build_case_record_from_session(session, final_report=final_report)
        get_case_store().put(record)
        logger.info(
            "Case persisted to DynamoDB (session_id=%s, triage=%s, dispatch=%s).",
            record.session_id,
            record.triage_level,
            record.dispatch_status,
        )
    except RuntimeError as exc:
        logger.warning("Case storage unavailable: %s", exc)
    except Exception as exc:
        logger.error("Failed to persist case to DynamoDB: %s", exc, exc_info=True)
