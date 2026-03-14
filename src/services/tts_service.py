from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_TTS_CACHE_MAX = max(0, int(os.environ.get("TTS_CACHE_MAX", "0").strip() or "0"))  # 0 = disabled
_TTS_CACHE_SIZE = min(2048, max(1, _TTS_CACHE_MAX)) if _TTS_CACHE_MAX else 0

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

# Microsoft Edge TTS – Neural sesler (daha insanı, robotik değil)
# Dil kodu -> Edge TTS voice ShortName (Neural)
_EDGE_VOICES = {
    "en": "en-US-JennyNeural",
    "tr": "tr-TR-EmelNeural",
    "de": "de-DE-KatjaNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "ar": "ar-SA-ZariyahNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "pt": "pt-BR-FranciscaNeural",
    "it": "it-IT-ElsaNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
    "uk": "uk-UA-OstapNeural",
    "hi": "hi-IN-SwaraNeural",
}


def _resolve_lang(code: str) -> str:
    code = (code or "en").lower().strip()
    return _LANG_MAP.get(code, "en")


def _get_edge_voice(lang_code: str) -> str:
    resolved = _resolve_lang(lang_code)
    return _EDGE_VOICES.get(resolved, _EDGE_VOICES["en"])


async def _synthesize_edge_async(text: str, lang: str) -> bytes:
    import edge_tts

    voice = _get_edge_voice(lang)
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


def _synthesize_gtts(text: str, lang: str) -> bytes:
    from gtts import gTTS

    resolved_lang = _resolve_lang(lang)
    tts = gTTS(text=text, lang=resolved_lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _synthesize_impl(text: str, lang: str) -> bytes:
    """Internal: Önce Edge TTS, başarısız olursa gTTS."""
    if not text or not text.strip():
        return b""
    try:
        return asyncio.run(_synthesize_edge_async(text, lang))
    except Exception as exc:
        logger.warning("Edge TTS failed (lang=%s), falling back to gTTS: %s", lang, exc)
    try:
        return _synthesize_gtts(text, lang)
    except Exception as exc:
        logger.error("TTS synthesis failed (lang=%s): %s", lang, exc)
        return b""


def synthesize(text: str, lang: str = "en") -> bytes:
    """Önce Edge TTS (insanı ses), başarısız olursa gTTS. Optional cache via TTS_CACHE_MAX."""
    if not text or not text.strip():
        return b""
    if _TTS_CACHE_SIZE > 0:
        return _synthesize_cached(text, lang)
    return _synthesize_impl(text, lang)


def _synthesize_cached(text: str, lang: str) -> bytes:
    """Cached TTS (used when TTS_CACHE_MAX > 0). LRU keyed by (text, lang)."""
    return _synthesize_impl(text, lang)


# Apply cache at import when TTS_CACHE_MAX is set (maxsize fixed at process start)
if _TTS_CACHE_SIZE > 0:
    _synthesize_cached = lru_cache(maxsize=_TTS_CACHE_SIZE)(_synthesize_cached)


def synthesize_to_file(text: str, lang: str = "en", path: str | Path | None = None) -> Path:
    audio = synthesize(text, lang)
    if path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        path = tmp.name
    path = Path(path)
    path.write_bytes(audio)
    return path
