from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = {
    "en": "english",
    "tr": "turkish",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "ar": "arabic",
    "ru": "russian",
    "zh": "chinese (simplified)",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "it": "italian",
    "nl": "dutch",
    "pl": "polish",
    "uk": "ukrainian",
    "hi": "hindi",
}


def _resolve_lang(code: str) -> str:
    code = (code or "en").lower().strip()
    return SUPPORTED_LANGS.get(code, code)


def translate(text: str, source: str, target: str) -> str:
    if not text or not text.strip():
        return text

    src = source.lower().strip()
    tgt = target.lower().strip()

    if src == tgt:
        return text

    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source=_resolve_lang(src), target=_resolve_lang(tgt))
        result = translator.translate(text)
        return result or text
    except Exception as exc:
        logger.error("Translation failed (%s→%s): %s", src, tgt, exc)
        return text


def detect_language(text: str) -> Optional[str]:
    """Detect the language of *text* and return its ISO-639-1 code.

    Returns ``None`` when detection fails or the detected language is not
    in ``SUPPORTED_LANGS``.  Uses langdetect (offline, fast) as the primary
    detector and falls back to deep-translator only when needed.
    """
    if not text or len(text.strip()) < 3:
        return None

    try:
        from langdetect import detect as _ld_detect, DetectorFactory
        DetectorFactory.seed = 0  # deterministic results
        detected = _ld_detect(text)
        detected = (detected or "").lower().strip()
        if detected in SUPPORTED_LANGS:
            logger.debug("langdetect → %s", detected)
            return detected
    except Exception as exc:
        logger.debug("langdetect failed: %s", exc)

    try:
        from deep_translator import single_detection
        detected = single_detection(text, api_key="", detector="google")
        detected = (detected or "").lower().strip()
        if detected in SUPPORTED_LANGS:
            logger.debug("deep-translator → %s", detected)
            return detected
    except Exception:
        pass

    return None


def translate_to_english(text: str, source_lang: str) -> str:
    return translate(text, source=source_lang, target="en")


def translate_from_english(text: str, target_lang: str) -> str:
    return translate(text, source="en", target=target_lang)
