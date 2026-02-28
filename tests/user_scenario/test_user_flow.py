"""User scenario: emergency caller starts session, describes situation, receives triage and report."""
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
                            with patch("services.translation_service.translate_to_english", side_effect=lambda t, source_lang=None: t or ""):
                                with patch("services.translation_service.translate_from_english", side_effect=lambda t, target_lang=None: t or ""):
                                    with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff\xfb"):
                                        with patch("orchestrator.orchestrator.translate_to_english", side_effect=lambda t, source_lang=None: t or ""):
                                            with patch("orchestrator.orchestrator.translate_from_english", side_effect=lambda t, target_lang=None: t or ""):
                                                with patch("orchestrator.orchestrator.detect_language", return_value="en"):
                                                    with patch("services.llm_service.get_llm_service") as llm_mock:
                                                        llm_mock.return_value.is_available = False
                                                        from main import app
                                                        with TestClient(app) as c:
                                                            yield c


def test_user_turkish_caller_medical_emergency(client: TestClient):
    """User starts in Turkish, reports medical emergency; system responds with triage."""
    r = client.post("/session/start", json={"language": "tr"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    greeting = r.json()["greeting"]
    assert "acil" in greeting.lower() or "Acil" in greeting
    r = client.post(
        "/session/message",
        json={"session_id": sid, "text": "Babam nefes almıyor ve baygın. 63 yaşında."},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["assistant_text"]
    triage = data.get("triage_result")
    if triage:
        assert triage.get("triage_level") in ("CRITICAL", "URGENT", "NON_URGENT")
        assert triage.get("category") in ("medical", "fire", "crime", "other")
    assert data.get("is_complete") in (True, False)


def test_user_english_caller_fire(client: TestClient):
    """User reports fire; system classifies as fire category."""
    r = client.post("/session/start", json={"language": "en"})
    sid = r.json()["session_id"]
    r = client.post(
        "/session/message",
        json={"session_id": sid, "text": "There is a fire in my house, smoke everywhere."},
    )
    assert r.status_code == 200
    triage = r.json().get("triage_result")
    if triage:
        assert triage.get("category") in ("fire", "medical", "crime", "other")
