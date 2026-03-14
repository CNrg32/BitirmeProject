"""Unit tests for tts_service (with mocked gTTS)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import services.tts_service as _tts_mod
_real_synthesize = _tts_mod.synthesize


class TestSynthesize:
    def test_synthesize_empty_returns_empty_bytes(self):
        assert _real_synthesize("") == b""
        assert _real_synthesize("   ") == b""

    def test_synthesize_success_with_mock(self):
        # Edge TTS öncelikli; mock'layınca gTTS yedek yolunu test ederiz
        with patch("services.tts_service._synthesize_edge_async", side_effect=Exception("skip edge")):
            with patch("gtts.gTTS") as mock_gtts:
                mock_inst = MagicMock()
                def write_to_fp(buf):
                    buf.write(b"\xff\xfb")
                mock_inst.write_to_fp.side_effect = write_to_fp
                mock_gtts.return_value = mock_inst
                result = _real_synthesize("Hi", "en")
                assert len(result) >= 1

    def test_synthesize_exception_returns_empty(self):
        # Edge TTS ve gTTS ikisi de hata verirse boş bytes döner
        with patch("services.tts_service._synthesize_edge_async", side_effect=Exception("edge fail")):
            with patch("gtts.gTTS", side_effect=Exception("network error")):
                result = _real_synthesize("Hello", "en")
        assert result == b""


class TestResolveLang:
    def test_resolve_known_lang(self):
        from services.tts_service import _resolve_lang
        assert _resolve_lang("en") == "en"
        assert _resolve_lang("tr") == "tr"

    def test_resolve_unknown_defaults_en(self):
        from services.tts_service import _resolve_lang
        assert _resolve_lang("xx") == "en"
        assert _resolve_lang(None) == "en"


class TestSynthesizeToFile:
    def test_synthesize_to_file_with_mock(self):
        from services.tts_service import synthesize_to_file
        with patch("services.tts_service.synthesize", return_value=b"\x00\x01"):
            path = synthesize_to_file("Hi", "en")
            assert path.exists()
            assert path.read_bytes() == b"\x00\x01"
            path.unlink(missing_ok=True)
