from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ImageAnalysisRecord:
    image_event_id: str
    image_sha256: str
    image_size_bytes: int
    analysis: Dict[str, Any]
    source: str = "analyze-image"
    session_id: Optional[str] = None
    filename: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def create(
        cls,
        *,
        image_bytes: bytes,
        analysis: Dict[str, Any],
        source: str,
        session_id: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> "ImageAnalysisRecord":
        return cls(
            image_event_id=uuid.uuid4().hex,
            image_sha256=hashlib.sha256(image_bytes).hexdigest(),
            image_size_bytes=len(image_bytes),
            analysis=analysis,
            source=source,
            session_id=session_id,
            filename=filename,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_item(self) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "image_event_id": self.image_event_id,
            "image_sha256": self.image_sha256,
            "image_size_bytes": self.image_size_bytes,
            "analysis": _to_dynamodb_value(self.analysis),
            "source": self.source,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }
        if self.session_id:
            item["session_id"] = self.session_id
        if self.filename:
            item["filename"] = self.filename
        return item


class ImageAnalysisStore:
    def put(self, record: ImageAnalysisRecord) -> Dict[str, Any]:
        raise NotImplementedError

    def list(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DynamoDBImageAnalysisStore(ImageAnalysisStore):
    def __init__(
        self,
        *,
        table_name: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.table_name = table_name or os.environ.get("IMAGE_ANALYSES_DYNAMODB_TABLE", "")
        if not self.table_name:
            raise RuntimeError(
                "IMAGE_ANALYSES_DYNAMODB_TABLE must be set for image analysis storage."
            )

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for DynamoDB image analysis storage. "
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

    def put(self, record: ImageAnalysisRecord) -> Dict[str, Any]:
        item = record.to_item()
        self._table.put_item(Item=item)
        return _normalize_item(item)

    def list(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        response = self._table.scan(Limit=limit)
        items = response.get("Items", [])
        items.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return [_normalize_item(item) for item in items]


def _to_dynamodb_value(value: Any) -> Any:
    return json.loads(json.dumps(value), parse_float=Decimal)


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


def get_image_analysis_store() -> ImageAnalysisStore:
    return DynamoDBImageAnalysisStore()
