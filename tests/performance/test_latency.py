"""Performance tests: /health and /predict latency (deterministic with mocks)."""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("api.model_loader.get_model_service") as mock_svc:
        svc = MagicMock()
        svc.is_loaded = False
        svc.load.return_value = False
        mock_svc.return_value = svc
        img_svc = MagicMock()
        img_svc.is_loaded = False
        img_svc.load.return_value = False
        with patch("services.image_service.get_image_model_service", return_value=img_svc):
            with patch("api.model_loader.load_sentiment_model", return_value=False):
                with patch("services.asr_service.preload_model", return_value=False):
                    with patch("services.tts_service.synthesize", return_value=b"\xff\xfb"):
                        with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff\xfb"):
                            from main import app
                            with TestClient(app) as c:
                                yield c


def test_health_latency_under_1s(client: TestClient):
    t0 = time.perf_counter()
    r = client.get("/health")
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert elapsed < 1.0, f"/health took {elapsed:.2f}s (expected < 1s)"


def test_predict_latency_under_2s(client: TestClient):
    t0 = time.perf_counter()
    r = client.post(
        "/predict",
        json={"text_en": "Someone is not breathing", "meta": {"deaths": 0, "potential_death": 0, "false_alarm": 0}, "slots": {}},
    )
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert elapsed < 2.0, f"/predict took {elapsed:.2f}s (expected < 2s)"


def test_session_start_latency_under_1s(client: TestClient):
    with patch("services.tts_service.synthesize", return_value=b"\xff\xfb"):
        with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff\xfb"):
            t0 = time.perf_counter()
            r = client.post("/session/start", json={})
            elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert elapsed < 1.0, f"/session/start took {elapsed:.2f}s"
