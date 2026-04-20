"""Google Cloud TTS path (mocked)."""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestGoogleCloudTts:
    def test_synthesize_google_cloud_returns_mp3_bytes(self):
        from services import tts_service as ts

        mp3_dummy = b"\xff\xfb\x90\x00"
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"audioContent": base64.b64encode(mp3_dummy).decode("ascii")}
        with patch.object(ts, "_GOOGLE_TTS_API_KEY", "fake-key"):
            with patch("httpx.Client") as mock_cls:
                mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
                out = ts._synthesize_google_cloud("Hello", "en")
        assert out == mp3_dummy
