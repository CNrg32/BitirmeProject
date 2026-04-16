from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_AUTO_TRANSCRIPTS_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "labels"
    / "auto_transcripts.csv"
)


@dataclass(frozen=True)
class TranscriptRecord:
    transcript_id: str
    path: str
    text: str
    avg_confidence: Optional[Decimal] = None
    chunks: Optional[str] = None
    source: str = "auto_transcripts.csv"
    imported_at: Optional[str] = None

    @classmethod
    def from_csv_row(
        cls,
        row: Dict[str, str],
        *,
        source: str = "auto_transcripts.csv",
        imported_at: Optional[str] = None,
    ) -> "TranscriptRecord":
        path = (row.get("path") or "").strip()
        text = (row.get("text") or "").replace("\r\n", "\n")
        chunks = row.get("chunks") or None
        raw_confidence = (row.get("avg_confidence") or "").strip()
        avg_confidence = Decimal(raw_confidence) if raw_confidence else None
        transcript_id = stable_transcript_id(path or text)

        return cls(
            transcript_id=transcript_id,
            path=path,
            text=text,
            avg_confidence=avg_confidence,
            chunks=chunks,
            source=source,
            imported_at=imported_at,
        )

    def to_item(self) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "transcript_id": self.transcript_id,
            "path": self.path,
            "text": self.text,
            "source": self.source,
            "imported_at": self.imported_at
            or datetime.now(timezone.utc).isoformat(),
        }
        if self.avg_confidence is not None:
            item["avg_confidence"] = self.avg_confidence
        if self.chunks is not None:
            item["chunks"] = self.chunks
        return item


def stable_transcript_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def load_auto_transcripts_csv(
    csv_path: str | Path = DEFAULT_AUTO_TRANSCRIPTS_CSV,
) -> List[TranscriptRecord]:
    path = Path(csv_path)
    imported_at = datetime.now(timezone.utc).isoformat()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            TranscriptRecord.from_csv_row(
                row,
                source=path.name,
                imported_at=imported_at,
            )
            for row in reader
        ]


class TranscriptStore:
    def put_many(self, records: Iterable[TranscriptRecord]) -> int:
        raise NotImplementedError

    def list(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class DynamoDBTranscriptStore(TranscriptStore):
    def __init__(
        self,
        *,
        table_name: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.table_name = table_name or os.environ.get("TRANSCRIPTS_DYNAMODB_TABLE", "")
        if not self.table_name:
            raise RuntimeError(
                "TRANSCRIPTS_DYNAMODB_TABLE must be set for DynamoDB transcript storage."
            )

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for DynamoDB transcript storage. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        region = region_name or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        endpoint = endpoint_url or os.environ.get("DYNAMODB_ENDPOINT_URL") or None
        resource_kwargs: Dict[str, Any] = {}
        if region:
            resource_kwargs["region_name"] = region
        if endpoint:
            resource_kwargs["endpoint_url"] = endpoint

        self._dynamodb = boto3.resource("dynamodb", **resource_kwargs)
        self._table = self._dynamodb.Table(self.table_name)

    def put_many(self, records: Iterable[TranscriptRecord]) -> int:
        count = 0
        with self._table.batch_writer(overwrite_by_pkeys=["transcript_id"]) as batch:
            for record in records:
                batch.put_item(Item=record.to_item())
                count += 1
        return count

    def list(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        response = self._table.scan(Limit=limit)
        return [_normalize_item(item) for item in response.get("Items", [])]

    def get(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        response = self._table.get_item(Key={"transcript_id": transcript_id})
        item = response.get("Item")
        return _normalize_item(item) if item else None


def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    confidence = normalized.get("avg_confidence")
    if isinstance(confidence, Decimal):
        normalized["avg_confidence"] = float(confidence)
    return normalized


def get_transcript_store() -> TranscriptStore:
    return DynamoDBTranscriptStore()
