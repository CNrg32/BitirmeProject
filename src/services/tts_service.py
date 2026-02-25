from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_LANG_MAP = {
    "en": "en",
    "tr": "tr",
    "de": "de",
    "fr": "fr",
    "es": "es",
    "ar": "ar",
    "ru": "ru",
    "zh": "zh-CN",
    "ja": "ja",
    "ko": "ko",
    "pt": "pt",
    "it": "it",
    "nl": "nl",
    "pl": "pl",
    "uk": "uk",
    "hi": "hi",
}


def _resolve_lang(code: str) -> str:
    code = (code or "en").lower().strip()
    return _LANG_MAP.get(code, "en")


def synthesize(text: str, lang: str = "en") -> bytes:
    if not text or not text.strip():
        return b""

    try:
        from gtts import gTTS

        resolved_lang = _resolve_lang(lang)
        tts = gTTS(text=text, lang=resolved_lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        logger.error("TTS synthesis failed (lang=%s): %s", lang, exc)
        return b""


def synthesize_to_file(text: str, lang: str = "en", path: str | Path | None = None) -> Path:
    audio = synthesize(text, lang)
    if path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        path = tmp.name
    path = Path(path)
    path.write_bytes(audio)
    return path
