from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services import image_service


def test_parse_s3_uri():
    bucket, key = image_service._parse_s3_uri(
        "s3://emergency-models/image/best_emergency_model.pth"
    )

    assert bucket == "emergency-models"
    assert key == "image/best_emergency_model.pth"


def test_parse_s3_uri_rejects_invalid_uri():
    with pytest.raises(ValueError):
        image_service._parse_s3_uri("https://example.com/model.pth")


def test_ensure_cloud_artifacts_downloads_model_and_derived_class_names(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "best_emergency_model.pth"
    class_names_path = tmp_path / "image_class_names.json"
    monkeypatch.setenv(
        "IMAGE_MODEL_S3_URI",
        "s3://emergency-models/image/best_emergency_model.pth",
    )
    monkeypatch.delenv("IMAGE_CLASS_NAMES_S3_URI", raising=False)

    with patch.object(image_service, "_download_s3_file", return_value=True) as download:
        image_service._ensure_cloud_artifacts(model_path, class_names_path)

    download.assert_any_call(
        "s3://emergency-models/image/best_emergency_model.pth",
        model_path,
    )
    download.assert_any_call(
        "s3://emergency-models/image/image_class_names.json",
        class_names_path,
    )


def test_analyze_image_attempts_lazy_load_when_model_is_not_loaded():
    svc = MagicMock()
    svc.is_loaded = False
    svc.load.side_effect = lambda: setattr(svc, "is_loaded", False)

    with patch.object(image_service, "analyze_image_quality", return_value={"usable": True}):
        with patch.object(image_service, "get_image_model_service", return_value=svc):
            result = image_service.analyze_image(b"not-an-image")

    assert result["available"] is False
    svc.load.assert_called_once()


def test_derive_visual_triage_critical_road_accident():
    result = image_service.derive_visual_triage(
        {
            "detected_class": "RoadAccidents",
            "confidence": 0.91,
            "mapped_category": "medical",
        }
    )

    assert result["triage_level"] == "CRITICAL"
    assert result["action"] == "EARLY_DISPATCH"
    assert "vehicle_crash" in result["visual_flags"]


def test_unusable_image_returns_recapture_decision():
    with patch.object(
        image_service,
        "analyze_image_quality",
        return_value={"usable": False, "reasons": ["too_dark"]},
    ):
        result = image_service.analyze_image(b"not-an-image")

    assert result["available"] is False
    assert result["visual_triage"]["action"] == "RECAPTURE_IMAGE"
