"""Optional backend paths for translation_service (mocked HTTP)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestGoogleRest:
    def test_translate_google_rest_parses_response(self):
        from services import translation_service as ts

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": {"translations": [{"translatedText": "Merhaba"}]},
        }
        with patch.object(ts, "_GOOGLE_TRANSLATE_API_KEY", "fake-key"):
            with patch("httpx.Client") as mock_cls:
                mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
                out = ts._translate_google_rest("Hello", "en", "tr")
        assert out == "Merhaba"


class TestDeepL:
    def test_translate_deepl_parses_response(self):
        from services import translation_service as ts

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"translations": [{"text": "Hallo"}]}
        with patch.object(ts, "_DEEPL_API_KEY", "fake-key"):
            with patch("httpx.Client") as mock_cls:
                mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
                out = ts._translate_deepl("Hello", "en", "de")
        assert out == "Hallo"
