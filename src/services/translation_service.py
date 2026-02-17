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


def translate_to_english(text: str, source_lang: str) -> str:
    return translate(text, source=source_lang, target="en")


def translate_from_english(text: str, target_lang: str) -> str:
    return translate(text, source="en", target=target_lang)
