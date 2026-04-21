from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_TTS_CACHE_MAX = max(0, int(os.environ.get("TTS_CACHE_MAX", "0").strip() or "0"))  # 0 = disabled
_TTS_CACHE_SIZE = min(2048, max(1, _TTS_CACHE_MAX)) if _TTS_CACHE_MAX else 0
_TTS_EDGE_MAX_RETRIES = max(1, int(os.environ.get("TTS_EDGE_MAX_RETRIES", "3").strip() or "3"))
_TTS_EDGE_RETRY_BASE_S = float(os.environ.get("TTS_EDGE_RETRY_BASE_S", "0.35").strip() or "0.35")

# Slightly slower speech reads more naturally for many neural voices (tweak via env).
_TTS_EDGE_RATE = (os.getenv("TTS_EDGE_RATE") or "-6%").strip() or "+0%"
_TTS_EDGE_VOLUME = (os.getenv("TTS_EDGE_VOLUME") or "+0%").strip() or "+0%"
_TTS_EDGE_PITCH = (os.getenv("TTS_EDGE_PITCH") or "+0Hz").strip() or "+0Hz"

# edge | google | auto — auto uses Google Cloud TTS when GOOGLE_TTS_API_KEY is set, else Edge.
_TTS_PROVIDER = (os.getenv("TTS_PROVIDER") or "auto").strip().lower()
_GOOGLE_TTS_API_KEY = (os.getenv("GOOGLE_TTS_API_KEY") or "").strip()
# Optional overrides (Neural2 / Wavenet names from Google Cloud TTS catalog)
_TTS_GOOGLE_VOICE_EN = (os.getenv("TTS_GOOGLE_VOICE_EN") or "en-US-Neural2-F").strip()
_TTS_GOOGLE_VOICE_TR = (os.getenv("TTS_GOOGLE_VOICE_TR") or "tr-TR-Neural2-A").strip()
_TTS_GOOGLE_SPEAKING_RATE = float((os.getenv("TTS_GOOGLE_SPEAKING_RATE") or "1.0").strip() or "1.0")
_TTS_GOOGLE_PITCH = float((os.getenv("TTS_GOOGLE_PITCH") or "0.0").strip() or "0.0")

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

# Microsoft Edge neural — Aria / Emel: generally clearer and less “flat” than older Jenny defaults.
_EDGE_VOICES = {
    "en": "en-US-AriaNeural",
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

# Google Cloud TTS voice names (languageCode must match voice prefix)
_GOOGLE_VOICES = {
    "en": _TTS_GOOGLE_VOICE_EN,
    "tr": _TTS_GOOGLE_VOICE_TR,
    "de": "de-DE-Neural2-F",
    "fr": "fr-FR-Neural2-A",
    "es": "es-ES-Neural2-E",
    "ar": "ar-XA-Wavenet-B",
    "ru": "ru-RU-Wavenet-E",
    "zh-CN": "cmn-CN-Wavenet-A",
    "ja": "ja-JP-Neural2-B",
    "ko": "ko-KR-Neural2-C",
    "pt": "pt-BR-Neural2-A",
    "it": "it-IT-Neural2-A",
    "nl": "nl-NL-Wavenet-A",
    "pl": "pl-PL-Wavenet-F",
    "uk": "uk-UA-Wavenet-A",
    "hi": "hi-IN-Wavenet-C",
}


def _resolve_lang(code: str) -> str:
    code = (code or "en").lower().strip()
    return _LANG_MAP.get(code, "en")


def _voice_override_key(resolved: str) -> str:
    """Env: TTS_EDGE_VOICE_EN, TTS_EDGE_VOICE_TR, TTS_EDGE_VOICE_ZH_CN, …"""
    safe = resolved.replace("-", "_").upper()
    return f"TTS_EDGE_VOICE_{safe}"


def _get_edge_voice(lang_code: str) -> str:
    """One neural voice per language; override via TTS_EDGE_VOICE_<LANG>."""
    resolved = _resolve_lang(lang_code)
    override = (os.environ.get(_voice_override_key(resolved)) or "").strip()
    if override:
        return override
    return _EDGE_VOICES.get(resolved, _EDGE_VOICES["en"])


def _get_google_voice_name(lang_code: str) -> str:
    resolved = _resolve_lang(lang_code)
    return _GOOGLE_VOICES.get(resolved, _GOOGLE_VOICES["en"])


def _google_language_code_from_voice(voice_name: str) -> str:
    # "en-US-Neural2-F" -> "en-US", "tr-TR-Neural2-A" -> "tr-TR"
    parts = voice_name.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return "en-US"


def get_tts_runtime_info() -> dict:
    uses_google = bool(_GOOGLE_TTS_API_KEY) and _TTS_PROVIDER in ("google", "auto")
    return {
        "edge_voice_policy": "single_neural_per_lang",
        "edge_max_retries": _TTS_EDGE_MAX_RETRIES,
        "cache_max_entries": _TTS_CACHE_MAX,
        "provider_requested": _TTS_PROVIDER,
        "google_tts_configured": bool(_GOOGLE_TTS_API_KEY),
        "primary_tts": "google_then_edge" if uses_google else "edge",
        "edge_rate": _TTS_EDGE_RATE,
        "edge_default_voices": {"en": _EDGE_VOICES["en"], "tr": _EDGE_VOICES["tr"]},
    }


def get_tts_runtime_info_str() -> str:
    d = get_tts_runtime_info()
    return (
        f"edge_retries={d['edge_max_retries']} cache_max={d['cache_max_entries']} "
        f"tts={d['primary_tts']} rate={d['edge_rate']}"
    )


async def _synthesize_edge_async(text: str, lang: str) -> bytes:
    import edge_tts

    voice = _get_edge_voice(lang)
    communicate = edge_tts.Communicate(
        text,
        voice,
        rate=_TTS_EDGE_RATE,
        volume=_TTS_EDGE_VOLUME,
        pitch=_TTS_EDGE_PITCH,
    )
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


async def _synthesize_edge_async_with_retry(text: str, lang: str) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(_TTS_EDGE_MAX_RETRIES):
        try:
            return await _synthesize_edge_async(text, lang)
        except Exception as exc:
            last_exc = exc
            if attempt < _TTS_EDGE_MAX_RETRIES - 1:
                delay = _TTS_EDGE_RETRY_BASE_S * (2**attempt)
                logger.warning(
                    "Edge TTS attempt %d/%d failed (lang=%s): %s — retry in %.2fs",
                    attempt + 1,
                    _TTS_EDGE_MAX_RETRIES,
                    lang,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _synthesize_google_cloud(text: str, lang: str) -> bytes:
    """Google Cloud Text-to-Speech REST (MP3). Requires GOOGLE_TTS_API_KEY."""
    import httpx

    voice_name = _get_google_voice_name(lang)
    language_code = _google_language_code_from_voice(voice_name)
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    body = {
        "input": {"text": text},
        "voice": {
            "languageCode": language_code,
            "name": voice_name,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": _TTS_GOOGLE_SPEAKING_RATE,
            "pitch": _TTS_GOOGLE_PITCH,
        },
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, params={"key": _GOOGLE_TTS_API_KEY}, json=body)
        r.raise_for_status()
        data = r.json()
    raw = data.get("audioContent")
    if not raw:
        return b""
    return base64.b64decode(raw)


def _synthesize_gtts(text: str, lang: str) -> bytes:
    from gtts import gTTS

    resolved_lang = _resolve_lang(lang)
    tts = gTTS(text=text, lang=resolved_lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _synthesize_impl(text: str, lang: str) -> bytes:
    """Google Cloud (optional) → Edge neural (+prosody) → gTTS."""
    if not text or not text.strip():
        return b""

    use_google = bool(_GOOGLE_TTS_API_KEY) and _TTS_PROVIDER in ("google", "auto")
    if use_google:
        try:
            return _synthesize_google_cloud(text, lang)
        except Exception as exc:
            logger.warning("Google Cloud TTS failed (lang=%s): %s", lang, exc)

    try:
        return asyncio.run(_synthesize_edge_async_with_retry(text, lang))
    except Exception as exc:
        logger.warning("Edge TTS failed (lang=%s), falling back to gTTS: %s", lang, exc)

    try:
        return _synthesize_gtts(text, lang)
    except Exception as exc:
        logger.error("TTS synthesis failed (lang=%s): %s", lang, exc)
        return b""


def synthesize(text: str, lang: str = "en") -> bytes:
    """Neural TTS (Google or Edge) then gTTS. Optional cache via TTS_CACHE_MAX."""
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
