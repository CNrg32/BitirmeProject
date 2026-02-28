"""Unit tests for asr_service with mocked Whisper model."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestPreloadModel:
    def test_preload_fails_when_whisper_import_error(self):
        from services import asr_service
        with patch.object(asr_service, "_get_model", side_effect=ImportError("no whisper")):
            with patch.object(asr_service, "_model", None):
                result = asr_service.preload_model()
        assert result is False

    def test_preload_succeeds_with_mock_model(self):
        from services import asr_service
        with patch.object(asr_service, "_get_model", return_value=MagicMock()):
            result = asr_service.preload_model()
        assert result is True


class TestTranscribeAudio:
    def test_transcribe_returns_tuple_with_mock_model(self):
        from services import asr_service
        mock_seg = MagicMock()
        mock_seg.text = "Hello"
        mock_seg.avg_logprob = -0.1
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
        with patch.object(asr_service, "_get_model", return_value=mock_model):
            with patch("tempfile.NamedTemporaryFile") as mock_tmp:
                m = MagicMock()
                m.name = "/tmp/fake"
                m.__enter__ = MagicMock(return_value=m)
                m.__exit__ = MagicMock(return_value=None)
                mock_tmp.return_value = m
                text, lang, conf = asr_service.transcribe_audio(audio_bytes=b"RIFF\x00\x00\x00\x00WAVE", language="en")
        assert text == "Hello"
        assert lang == "en"
        assert isinstance(conf, float)
