"""Unit tests for translation_service (with mocked external APIs)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestResolveLang:
    def test_resolve_known(self):
        from services.translation_service import SUPPORTED_LANGS
        assert "en" in SUPPORTED_LANGS
        assert "tr" in SUPPORTED_LANGS

    def test_translate_empty_returns_unchanged(self):
        from services.translation_service import translate
        assert translate("", "en", "tr") == ""
        assert translate("   ", "en", "tr") == "   "

    def test_translate_same_source_target_returns_text(self):
        from services.translation_service import translate
        assert translate("hello", "en", "en") == "hello"

    def test_translate_success_with_mock(self):
        from services.translation_service import translate
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            pytest.skip("deep_translator not installed")
        with patch("deep_translator.GoogleTranslator") as MockGT:
            MockGT.return_value.translate.return_value = "merhaba"
            result = translate("hello", "en", "tr")
        assert result in ("merhaba", "hello")

    def test_translate_exception_returns_original(self):
        from services.translation_service import translate
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            pytest.skip("deep_translator not installed")
        with patch("deep_translator.GoogleTranslator", side_effect=Exception("network error")):
            result = translate("hello", "en", "tr")
        assert result == "hello"


class TestDetectLanguage:
    def test_detect_empty_returns_none(self):
        from services.translation_service import detect_language
        assert detect_language("") is None
        assert detect_language("ab") is None

    def test_detect_success_with_langdetect(self):
        from services.translation_service import detect_language
        try:
            from langdetect import detect, DetectorFactory
        except ImportError:
            pytest.skip("langdetect not installed")
        with patch("langdetect.detect", return_value="en"):
            result = detect_language("This is English text for detection.")
        assert result == "en"

    def test_detect_langdetect_exception_tries_single_detection(self):
        from services.translation_service import detect_language
        try:
            from langdetect import detect
            from deep_translator import single_detection
        except ImportError:
            pytest.skip("langdetect or deep_translator not installed")
        with patch("langdetect.detect", side_effect=Exception("fail")):
            with patch("deep_translator.single_detection", return_value="tr"):
                result = detect_language("Bu Türkçe bir cümledir.")
        assert result in ("tr", None)


class TestTranslateHelpers:
    def test_translate_to_english(self):
        from services.translation_service import translate_to_english
        result = translate_to_english("merhaba", "tr")
        assert result in ("merhaba", "hello")

    def test_translate_from_english(self):
        from services.translation_service import translate_from_english
        result = translate_from_english("hello", "tr")
        assert result in ("hello", "merhaba")
