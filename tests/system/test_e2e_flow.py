"""System / E2E tests: full session flow (start -> message -> complete)."""
from __future__ import annotations

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
                        with patch("services.translation_service.translate", side_effect=lambda t, source=None, target=None: t or ""):
                            with patch("services.translation_service.translate_to_english", side_effect=lambda t, sl: t or ""):
                                with patch("services.translation_service.translate_from_english", side_effect=lambda t, tl: t or ""):
                                    with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff\xfb"):
                                        with patch("orchestrator.orchestrator.translate_to_english", side_effect=lambda t, sl: t or ""):
                                            with patch("orchestrator.orchestrator.translate_from_english", side_effect=lambda t, tl: t or ""):
                                                with patch("orchestrator.orchestrator.detect_language", return_value="en"):
                                                    with patch("services.llm_service.get_llm_service") as llm_mock:
                                                        llm_mock.return_value.is_available = False
                                                        from main import app
                                                        with TestClient(app) as c:
                                                            yield c


def test_e2e_health_then_session_start_then_message(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    r = client.post("/session/start", json={"language": "en"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    r = client.post("/session/message", json={"session_id": sid, "text": "My father is not breathing. He is 63 years old."})
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == sid
    assert len(data["assistant_text"]) > 0
    assert data.get("triage_result") is not None or data.get("is_complete") is not None


def test_e2e_predict_then_tts(client: TestClient):
    r = client.post("/predict", json={"text_en": "Heart attack", "meta": {}, "slots": {}})
    assert r.status_code == 200
    assert r.json()["triage_level"] in ("CRITICAL", "URGENT", "NON_URGENT")
    r = client.post("/tts", data={"text": "Help is on the way", "language": "en"})
    assert r.status_code == 200
    assert len(r.content) >= 1
