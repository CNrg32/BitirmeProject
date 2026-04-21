from __future__ import annotations

import logging
import os
from typing import Any, Optional

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

# deep_translator | deepl | google | local
_TRANSLATION_BACKEND = (os.getenv("TRANSLATION_BACKEND") or "deep_translator").strip().lower()

# Google Cloud Translation API key (REST v2)
_GOOGLE_TRANSLATE_API_KEY = (os.getenv("GOOGLE_TRANSLATE_API_KEY") or os.getenv("TRANSLATE_GOOGLE_API_KEY") or "").strip()

# DeepL API key (https://www.deepl.com/pro-api)
_DEEPL_API_KEY = (os.getenv("DEEPL_API_KEY") or "").strip()
_DEEPL_FREE = (os.getenv("DEEPL_USE_FREE_API") or "0").strip().lower() in ("1", "true", "yes")

# Lazy Marian models for tr <-> en (offline)
_marian_tr_en: Any = None
_marian_en_tr: Any = None


def _resolve_lang(code: str) -> str:
    code = (code or "en").lower().strip()
    return SUPPORTED_LANGS.get(code, code)


def get_translation_backend_name() -> str:
    """Active backend id for logging and API responses."""
    return _TRANSLATION_BACKEND


def _translate_deep_translator(text: str, source: str, target: str) -> str:
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=_resolve_lang(source), target=_resolve_lang(target))
    result = translator.translate(text)
    return result or text


def _deepl_lang(code: str) -> str:
    c = (code or "en").lower().strip()
    # DeepL uses upper-case ISO codes; Chinese is ZH
    special = {"en": "EN", "tr": "TR", "de": "DE", "fr": "FR", "es": "ES", "pt": "PT", "it": "IT", "nl": "NL", "pl": "PL", "ru": "RU", "ja": "JA", "zh": "ZH"}
    return special.get(c, c.upper()[:2] if len(c) >= 2 else "EN")


def _translate_deepl(text: str, source: str, target: str) -> str:
    if not _DEEPL_API_KEY:
        logger.warning("DEEPL_API_KEY not set; falling back to deep_translator")
        return _translate_deep_translator(text, source, target)

    import httpx

    base = "https://api-free.deepl.com" if _DEEPL_FREE else "https://api.deepl.com"
    url = f"{base}/v2/translate"
    headers = {"Authorization": f"DeepL-Auth-Key {_DEEPL_API_KEY}"}
    data: dict[str, str] = {"text": text, "target_lang": _deepl_lang(target)}
    if source and source != "auto":
        data["source_lang"] = _deepl_lang(source)

    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=headers, data=data)
        r.raise_for_status()
        body = r.json()
    translations = body.get("translations") or []
    if not translations:
        return text
    return translations[0].get("text") or text


def _translate_google_rest(text: str, source: str, target: str) -> str:
    if not _GOOGLE_TRANSLATE_API_KEY:
        logger.warning("GOOGLE_TRANSLATE_API_KEY not set; falling back to deep_translator")
        return _translate_deep_translator(text, source, target)

    import httpx

    url = "https://translation.googleapis.com/language/translate/v2"
    payload: dict = {"q": text, "target": target, "format": "text"}
    if source and source != "auto":
        payload["source"] = source
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, params={"key": _GOOGLE_TRANSLATE_API_KEY}, json=payload)
        r.raise_for_status()
        body = r.json()
    data = body.get("data", {}).get("translations") or []
    if not data:
        return text
    return data[0].get("translatedText") or text


def _load_marian_pair(src: str, tgt: str) -> tuple[Any, Any] | None:
    global _marian_tr_en, _marian_en_tr
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        logger.warning("transformers not available for local translation")
        return None

    if src == "tr" and tgt == "en":
        if _marian_tr_en is None:
            name = "Helsinki-NLP/opus-mt-tr-en"
            tok = MarianTokenizer.from_pretrained(name)
            mdl = MarianMTModel.from_pretrained(name)
            _marian_tr_en = (tok, mdl)
        return _marian_tr_en
    if src == "en" and tgt == "tr":
        if _marian_en_tr is None:
            name = "Helsinki-NLP/opus-mt-en-tr"
            tok = MarianTokenizer.from_pretrained(name)
            mdl = MarianMTModel.from_pretrained(name)
            _marian_en_tr = (tok, mdl)
        return _marian_en_tr
    return None


def _translate_local_marian(text: str, source: str, target: str) -> str:
    src = source.lower().strip()
    tgt = target.lower().strip()
    pair = _load_marian_pair(src, tgt)
    if pair is None:
        logger.debug("No local Marian model for %s→%s; using deep_translator", src, tgt)
        return _translate_deep_translator(text, source, target)

    tokenizer, model = pair
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        gen = model.generate(**batch, max_length=512)
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    return out or text


def translate(text: str, source: str, target: str) -> str:
    if not text or not text.strip():
        return text

    src = source.lower().strip()
    tgt = target.lower().strip()

    if src == tgt:
        return text

    backends: dict[str, Callable[[str, str, str], str]] = {
        "deep_translator": _translate_deep_translator,
        "deepl": _translate_deepl,
        "google": _translate_google_rest,
        "local": _translate_local_marian,
    }

    fn = backends.get(_TRANSLATION_BACKEND, _translate_deep_translator)
    try:
        return fn(text, source, target)
    except Exception as exc:
        logger.error(
            "Translation failed (backend=%s %s→%s): %s",
            _TRANSLATION_BACKEND,
            src,
            tgt,
            exc,
        )
        if fn is not _translate_deep_translator:
            try:
                return _translate_deep_translator(text, source, target)
            except Exception as exc2:
                logger.error("deep_translator fallback failed: %s", exc2)
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
