"""Integration tests for API endpoints (with mocked services)."""
from __future__ import annotations

import base64
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from api.schemas import MetaInput, SlotsInput


@pytest.fixture
def client():
    """API client with all external services mocked (lifespan, TTS, translation)."""
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
                                                    from main import app
                                                    with TestClient(app) as c:
                                                        yield c


class TestHealth:
    def test_health_returns_200(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "ml_model_loaded" in data
        assert "image_model_loaded" in data
        assert "sentiment_model_loaded" in data

    def test_health_structure(self, client: TestClient):
        r = client.get("/health")
        assert r.json()["status"] == "ok"


class TestPredict:
    def test_predict_with_ml_model_loaded(self, client: TestClient):
        from main import _run_predict
        svc = MagicMock()
        svc.is_loaded = True
        svc.predict.return_value = ("URGENT", 0.85)
        with patch("main.get_model_service", return_value=svc):
            result = _run_predict("Someone has chest pain", MetaInput(), SlotsInput())
        assert result["triage_level"] in ("URGENT", "CRITICAL")
        assert result["confidence"] == 0.85
        assert result["category"] in ("medical", "fire", "crime", "other")

    def test_predict_merge_client_slots(self, client: TestClient):
        from main import _run_predict
        result = _run_predict(
            "Person is 50 years old",
            MetaInput(),
            SlotsInput(age=50, severity_1_10=7),
        )
        assert result["slots"].get("age") == 50
        assert result["slots"].get("severity_1_10") == 7

    def test_predict_valid_payload(self, client: TestClient):
        payload = {
            "text_en": "My dad is not breathing and unconscious, he is 63 years old",
            "meta": {"deaths": 0, "potential_death": 0, "false_alarm": 0},
            "slots": {},
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["category"] in ("medical", "fire", "crime", "other")
        assert data["triage_level"] in ("CRITICAL", "URGENT", "NON_URGENT")
        assert "red_flags" in data
        assert "slots" in data

    def test_predict_empty_text_422(self, client: TestClient):
        r = client.post("/predict", json={"text_en": "", "meta": {}, "slots": {}})
        assert r.status_code == 422

    def test_predict_missing_text_en_422(self, client: TestClient):
        r = client.post("/predict", json={"meta": {}, "slots": {}})
        assert r.status_code == 422

    def test_predict_negative_meta_accepted(self, client: TestClient):
        payload = {
            "text_en": "Minor cut on finger",
            "meta": {"deaths": 0, "potential_death": 0, "false_alarm": 1},
            "slots": {},
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["triage_level"] in ("CRITICAL", "URGENT", "NON_URGENT")


class TestSessionStart:
    def test_session_start_returns_id_and_greeting(self, client: TestClient):
        r = client.post("/session/start", json={})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert "greeting" in data
        assert len(data["session_id"]) >= 8

    def test_session_start_with_language(self, client: TestClient):
        r = client.post("/session/start", json={"language": "tr"})
        assert r.status_code == 200
        assert "Acil" in r.json()["greeting"] or "acil" in r.json()["greeting"].lower()


class TestSessionMessage:
    def test_session_message_unknown_session_404(self, client: TestClient):
        r = client.post(
            "/session/message",
            json={"session_id": "nonexistent-id-xyz", "text": "Help"},
        )
        assert r.status_code == 404

    def test_session_message_valid_flow(self, client: TestClient):
        start = client.post("/session/start", json={"language": "en"})
        assert start.status_code == 200
        sid = start.json()["session_id"]
        r = client.post(
            "/session/message",
            json={"session_id": sid, "text": "My father is not breathing, he is 63 years old"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == sid
        assert data["assistant_text"]
        assert "triage_result" in data or data.get("is_complete") is not None
        assert data.get("is_complete") in (True, False)

    def test_session_message_returns_nearby_places(self, client: TestClient):
        start = client.post("/session/start", json={"language": "en"})
        sid = start.json()["session_id"]

        with patch(
            "services.nearby_places_service.get_nearby_places",
            return_value=[
                {
                    "id": "p1",
                    "type": "police",
                    "name": "Central Police",
                    "distance_meters": 450,
                    "latitude": 41.0,
                    "longitude": 29.0,
                }
            ],
        ):
            r = client.post(
                "/session/message",
                json={
                    "session_id": sid,
                    "text": "Someone is attacking me",
                    "latitude": 41.0,
                    "longitude": 29.0,
                },
            )

        assert r.status_code == 200
        data = r.json()
        assert isinstance(data.get("nearby_places"), list)
        assert data["nearby_places"][0]["type"] == "police"

    def test_session_message_invalid_base64_audio_400(self, client: TestClient):
        start = client.post("/session/start", json={})
        sid = start.json()["session_id"]
        r = client.post(
            "/session/message",
            json={"session_id": sid, "audio_base64": "not-valid-base64!!"},
        )
        assert r.status_code == 400


class TestNearbyPlaces:
    def test_nearby_places_endpoint_returns_list(self, client: TestClient):
        with patch(
            "services.nearby_places_service.get_nearby_places",
            return_value=[
                {
                    "id": "h1",
                    "type": "hospital",
                    "name": "State Hospital",
                    "distance_meters": 320,
                    "latitude": 41.0,
                    "longitude": 29.0,
                }
            ],
        ):
            r = client.post(
                "/nearby-places",
                json={"latitude": 41.0, "longitude": 29.0, "preferred_type": "hospital"},
            )

        assert r.status_code == 200
        data = r.json()
        assert isinstance(data.get("nearby_places"), list)
        assert data["nearby_places"][0]["type"] == "hospital"


class TestSessionTranscribe:
    def test_transcribe_unknown_session_404(self, client: TestClient):
        r = client.post(
            "/session/transcribe",
            json={"session_id": "nonexistent", "audio_base64": base64.b64encode(b"x").decode()},
        )
        assert r.status_code == 404

    def test_transcribe_invalid_base64_400(self, client: TestClient):
        start = client.post("/session/start", json={})
        sid = start.json()["session_id"]
        r = client.post(
            "/session/transcribe",
            json={"session_id": sid, "audio_base64": "!!!!invalid!!!!"},
        )
        assert r.status_code == 400

    def test_transcribe_success_with_mock_asr(self, client: TestClient):
        start = client.post("/session/start", json={})
        sid = start.json()["session_id"]
        with patch("services.asr_service.transcribe_audio", return_value=("Transcribed text here", "en", 0.95)):
            r = client.post(
                "/session/transcribe",
                json={"session_id": sid, "audio_base64": base64.b64encode(b"fake-wav").decode()},
            )
        assert r.status_code == 200
        assert r.json()["transcript"] == "Transcribed text here"
        assert r.json()["detected_language"] == "en"


class TestAnalyzeImage:
    def test_analyze_image_empty_file_400(self, client: TestClient):
        r = client.post(
            "/analyze-image",
            files={"image": ("x.jpg", b"", "image/jpeg")},
            data={},
        )
        assert r.status_code == 400

    def test_analyze_image_model_unavailable_503(self, client: TestClient):
        with patch("services.image_service.analyze_image", return_value={"available": False, "summary": "Model not loaded."}):
            r = client.post(
                "/analyze-image",
                files={"image": ("x.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg")},
                data={},
            )
        assert r.status_code == 503


class TestTranslate:
    def test_translate_returns_translated(self, client: TestClient):
        r = client.post("/translate", data={"text": "Hello", "source": "en", "target": "tr"})
        assert r.status_code == 200
        # With mock we get same text back
        assert "translated" in r.json()


class TestTTS:
    def test_tts_returns_audio(self, client: TestClient):
        r = client.post("/tts", data={"text": "Hello", "language": "en"})
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("audio/")
        assert len(r.content) >= 1

    def test_tts_empty_text_returns_422(self, client: TestClient):
        r = client.post("/tts", data={"text": "", "language": "en"})
        assert r.status_code == 422
