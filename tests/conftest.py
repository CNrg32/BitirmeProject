"""
Shared fixtures and mocks for deterministic testing.
External services (LLM, TTS, ASR, translation, image model) are mocked.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is on path (when run from project root)
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# App client (lifespan skipped for unit/integration; we patch model load)
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client():
    """FastAPI TestClient with lifespan and heavy services mocked."""
    from fastapi.testclient import TestClient

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
                    from main import app
                    with TestClient(app) as client:
                        yield client


@pytest.fixture
def app_client_no_lifespan():
    """Client without running lifespan (for routes that don't need models)."""
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Session store (fresh store for tests, optional TTL override)
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_session_store():
    """Return a new SessionStore and override get_session_store."""
    from orchestrator.session import SessionStore, get_session_store
    store = SessionStore(ttl_seconds=3600)
    with patch("orchestrator.session.get_session_store", return_value=store):
        with patch("orchestrator.orchestrator.get_session_store", return_value=store):
            with patch("main.get_session_store", return_value=store):
                yield store


# ---------------------------------------------------------------------------
# Mock TTS (synthesize returns minimal bytes to avoid network)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_tts():
    with patch("services.tts_service.synthesize", return_value=b"\xff\xfb\x90\x00"):
        with patch("orchestrator.orchestrator.synthesize", return_value=b"\xff\xfb\x90\x00"):
            yield


# ---------------------------------------------------------------------------
# Mock LLM (unavailable so rule-based flow is used)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_unavailable():
    with patch("services.llm_service.get_llm_service") as m:
        inst = MagicMock()
        inst.is_available = False
        m.return_value = inst
        yield m


# ---------------------------------------------------------------------------
# Mock translation (return input to avoid Google Translate)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_translate_functions():
    """Keep translation as pass-through so tests are deterministic."""
    with patch("services.translation_service.translate", side_effect=lambda t, source=None, target=None: t or ""):
        with patch("services.translation_service.translate_to_english", side_effect=lambda t, sl: t or ""):
            with patch("services.translation_service.translate_from_english", side_effect=lambda t, tl: t or ""):
                with patch("orchestrator.orchestrator.translate_to_english", side_effect=lambda t, sl: t or ""):
                    with patch("orchestrator.orchestrator.translate_from_english", side_effect=lambda t, tl: t or ""):
                        with patch("orchestrator.orchestrator.detect_language", return_value="en"):
                            yield


# ---------------------------------------------------------------------------
# Mock ASR (return fixed transcript)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_asr():
    with patch("services.asr_service.transcribe_audio") as m:
        m.return_value = ("My father is not breathing", "en", 0.95)
        yield m


# ---------------------------------------------------------------------------
# Mock image service (model not loaded or return stub)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_image_service_unavailable():
    with patch("services.image_service.analyze_image") as m:
        m.return_value = {"available": False, "summary": "Image model not loaded."}
        yield m

    with patch("services.image_service.get_image_model_service") as m2:
        inst = MagicMock()
        inst.is_loaded = False
        inst.load.return_value = False
        m2.return_value = inst
        yield m2


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_predict_payload():
    return {
        "text_en": "My dad is not breathing and unconscious, he is 63 years old",
        "meta": {"deaths": 0, "potential_death": 0, "false_alarm": 0},
        "slots": {},
    }


@pytest.fixture
def valid_session_start_payload():
    return {"language": "en"}


@pytest.fixture
def small_audio_b64():
    """Minimal valid base64 (tiny WAV-like bytes)."""
    return base64.b64encode(b"\x00\x01\x02\x03\x04").decode("ascii")
