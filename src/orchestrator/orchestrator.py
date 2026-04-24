"""
Orchestrator – main entry point for session management and message handling.

Turn flow (LLM-powered):
  1.  ASR (if audio provided)
  2.  Session language (preference, first ASR, or first text — then fixed)
  3.  Slot extraction + response generation via Gemini (full conversation history)
  4.  Sentiment analysis on audio (enhances triage level if panic detected)
  5.  Image analysis (if image was attached)
  6.  TTS the assistant reply
  7.  Return structured response

LLM mode requires GROQ_API_KEY (dialog) and OPENAI_API_KEY + OPENAI_FINE_TUNED_MODEL
(triage). If either side is missing, rule-based dialog is used.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.asr_service import get_asr_runtime_info_str
from services.translation_service import (
    translate_to_english,
    translate_from_english,
    detect_language,
    translate,
    get_translation_backend_name,
)
from services.tts_service import synthesize, get_tts_runtime_info_str
from orchestrator.session import Session, get_session_store
from orchestrator.report_composer import compose_report
from services.final_report_openai import generate_final_report_openai
from services.case_store import persist_case_if_configured
from services.turn_trace import (
    is_turn_trace_enabled,
    trace_banner,
    trace_kv,
    trace_orchestrator_outcome,
    trace_step,
)

logger = logging.getLogger(__name__)


def _tts_inline_enabled() -> bool:
    """TTS_INLINE controls whether /session/message synthesises audio inline.

    Defaults to true so audio playback keeps working out of the box. Set
    TTS_INLINE=false in the environment to skip synthesis in the hot path and
    let the client fetch audio via /tts (saves 0.8–2 s per turn).
    """
    raw = (os.environ.get("TTS_INLINE") or "").strip().lower()
    if not raw:
        return True
    return raw in ("1", "true", "yes", "on")


def _uses_single_call_first_turn(llm: Any) -> bool:
    """Groq fast few-shot path (used for gibberish_check heuristics only)."""
    model = str(getattr(llm, "MODEL", ""))
    if model.startswith("groq/"):
        return os.environ.get("GROQ_FAST_PATH", "true").strip().lower() not in ("0", "false", "no")
    return False


# ---------------------------------------------------------------------------
# Greetings
# ---------------------------------------------------------------------------

_GREETINGS = {
    "en": "Emergency assistant here. What is your emergency?",
    "tr": "Acil servis, acil durumunuz nedir?",
    "de": "Notfall-Assistent hier. Was ist Ihr Notfall?",
    "fr": "Assistant d'urgence ici. Quelle est votre urgence ?",
    "es": "Asistente de emergencia aquí. ¿Cuál es su emergencia?",
    "ar": "مساعد الطوارئ هنا. ما هي حالتك الطارئة؟",
    "ru": "Ассистент экстренной помощи. Что случилось?",
}


def _audio_to_data_url(audio_bytes: Optional[bytes]) -> Optional[str]:
    if not audio_bytes:
        return None
    return f"data:audio/mpeg;base64,{base64.b64encode(audio_bytes).decode()}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SUPPORTED_LANGS = ("tr", "en")
_DEFAULT_LANG = "tr"

# Characters that only appear in Turkish (plus common Turkish digraphs that
# disambiguate from English). One occurrence is a strong signal for TR.
_TURKISH_ONLY_CHARS = set("çğıöşüÇĞİÖŞÜ")

# High-signal English stopwords / function words that almost never appear in a
# Turkish emergency utterance. Used as a fallback when langdetect fails or is
# overconfident on a short string.
_ENGLISH_SIGNAL_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "our", "their",
    "me", "him", "us", "them",
    "and", "or", "but", "not", "no", "yes",
    "what", "where", "when", "why", "how", "who", "which",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "might",
    "help", "please", "there", "here", "now", "just", "only",
    "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "from", "with", "for", "about", "as", "by",
    "dont", "wont", "cant", "im", "ive", "youre", "hes", "shes", "its",
    "bleeding", "breathing", "hurt", "hurts", "pain", "fire", "help", "unconscious",
    "someone", "somebody", "father", "mother", "brother", "sister", "son", "daughter",
    "husband", "wife", "child", "children",
}

# High-signal Turkish words that will appear in almost any Turkish emergency
# message — used to outvote langdetect noise.
_TURKISH_SIGNAL_WORDS = {
    "ve", "bir", "bu", "şu", "için", "ile", "ama", "ancak", "fakat",
    "çok", "daha", "en", "var", "yok", "değil",
    "ben", "sen", "biz", "siz", "onlar",
    "ne", "nerede", "nasıl", "neden", "niçin", "kim", "hangi", "kaç",
    "lütfen", "yardım", "imdat", "acil",
    "hasta", "ağrı", "ateş", "kanama", "nefes", "baygın", "bayıldı",
    "baba", "anne", "kardeş", "kızım", "oğlum", "kocam", "eşim", "çocuk",
    "evde", "burada", "orada", "şurada",
    "yanıyor", "yangın", "saldırı", "hırsız", "kaza",
}


def _detect_tr_or_en(text: str) -> str:
    """Robust Turkish-vs-English classifier.

    Priority order:
      1. Turkish-only characters (ç, ğ, ı, ş, ö, ü) → ``tr``.
      2. Stopword vote between English and Turkish signal words.
      3. ``detect_language`` (langdetect) if it returns exactly ``tr`` or ``en``.
      4. Fallback: ``_DEFAULT_LANG`` (``tr``).

    Robust against short strings where langdetect is unreliable.
    """
    if not text or not text.strip():
        return _DEFAULT_LANG

    # Turkish-specific characters are a near-perfect signal.
    if any(c in text for c in _TURKISH_ONLY_CHARS):
        return "tr"

    import re as _re
    tokens = set(_re.findall(r"[a-z']+", text.lower()))
    en_hits = len(tokens & _ENGLISH_SIGNAL_WORDS)
    tr_hits = len(tokens & _TURKISH_SIGNAL_WORDS)

    if en_hits > tr_hits and en_hits >= 1:
        return "en"
    if tr_hits > en_hits and tr_hits >= 1:
        return "tr"

    detected = detect_language(text)
    if detected in _SUPPORTED_LANGS:
        return detected

    return _DEFAULT_LANG


def _clamp_supported_lang(lang: Optional[str]) -> str:
    """Collapse any detected/requested language down to the supported set."""
    if not lang:
        return _DEFAULT_LANG
    lang = lang.strip().lower()
    return lang if lang in _SUPPORTED_LANGS else _DEFAULT_LANG


def start_session(language: Optional[str] = None) -> Dict[str, Any]:
    """Create a new session.

    The UI-supplied ``language`` argument is IGNORED for locking purposes —
    the session language is always determined from the user's first message
    (audio or text) and is restricted to Turkish or English. The greeting is
    rendered in Turkish by default; it switches once the first user input
    arrives.
    """
    store = get_session_store()
    session = store.create(language=_DEFAULT_LANG)
    session.language_locked = False
    lang = _DEFAULT_LANG

    greeting = _GREETINGS.get(lang, _GREETINGS["en"])
    session.messages.append({"role": "assistant", "text": greeting})

    if _tts_inline_enabled():
        audio_bytes = synthesize(greeting, lang=lang)
        audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
        audio_url = _audio_to_data_url(audio_bytes)
    else:
        audio_b64 = None
        audio_url = None

    return {
        "session_id": session.session_id,
        "greeting": greeting,
        "greeting_audio_url": audio_url,
        "greeting_audio_b64": audio_b64,
    }


# Language-aware fallback messages when LLM parse fails
_LLM_FALLBACK: Dict[str, str] = {
    "tr": "Anlayamadım, lütfen tekrar anlatır mısınız?",
    "en": "I didn't catch that, could you please describe the situation again?",
    "de": "Ich habe das nicht verstanden, bitte beschreiben Sie die Situation erneut.",
    "fr": "Je n'ai pas compris, pourriez-vous décrire la situation à nouveau ?",
    "es": "No entendí, ¿podría describir la situación nuevamente?",
    "ar": "لم أفهم، هل يمكنك وصف الوضع مرة أخرى؟",
    "ru": "Я не понял, не могли бы вы описать ситуацию снова?",
}

# Language-aware dispatch messages (Guard 2: red flags)
_DISPATCH_MSG: Dict[str, str] = {
    "tr": "Yardım ekipleri yönlendiriliyor, lütfen hatta kalın.",
    "en": "Emergency services are being dispatched. Please stay on the line.",
    "de": "Notfalldienste werden entsandt. Bitte bleiben Sie in der Leitung.",
    "fr": "Les secours sont en route. Veuillez rester en ligne.",
    "es": "Los servicios de emergencia están en camino. Por favor permanezca en línea.",
    "ar": "يتم إرسال الخدمات الطارئة. الرجاء البقاء على الخط.",
    "ru": "Экстренные службы вызваны. Пожалуйста, оставайтесь на линии.",
}

# Language-aware completion messages (Guard 3: max turns)
_COMPLETE_MSG: Dict[str, str] = {
    "tr": "Teşekkürler, yeterli bilgiye sahibim. Yardım yönlendiriliyor.",
    "en": "Thank you. I have enough information. Help is on the way.",
    "de": "Danke. Ich habe genug Informationen. Hilfe ist unterwegs.",
    "fr": "Merci. J'ai suffisamment d'informations. Les secours arrivent.",
    "es": "Gracias. Tengo suficiente información. La ayuda está en camino.",
    "ar": "شكراً. لدي معلومات كافية. المساعدة في الطريق.",
    "ru": "Спасибо. У меня достаточно информации. Помощь уже едет.",
}

# Final-turn closing messages used when the session ends after a dispatch.
# These are strictly informational ("teams are on the way / case recorded") and
# NEVER contain a question — the last response shown to the user must not leave
# them expecting another turn.
_FINAL_DISPATCHED_MSG: Dict[str, str] = {
    "tr": "Acil yardım ekipleri yönlendirildi ve yardım yolda. Tüm bilgileri aldık, aşağıdaki vaka özetini inceleyebilirsiniz. Gerekirse hatta kalın.",
    "en": "Emergency responders have been dispatched and help is on the way. All information has been received; please review the case summary below and stay on the line if needed.",
    "de": "Notfallkräfte wurden entsandt und Hilfe ist unterwegs. Alle Informationen wurden erfasst. Bitte sehen Sie sich die Fallzusammenfassung unten an.",
    "fr": "Les secours ont été envoyés et l'aide est en route. Toutes les informations ont été enregistrées. Veuillez consulter le résumé du cas ci-dessous.",
    "es": "Los servicios de emergencia han sido enviados y la ayuda está en camino. Toda la información ha sido registrada. Consulte el resumen del caso a continuación.",
    "ar": "تم إرسال فرق الطوارئ والمساعدة في طريقها إليك. تم استلام جميع المعلومات؛ يرجى مراجعة ملخص الحالة أدناه.",
    "ru": "Экстренные службы направлены, помощь уже в пути. Вся информация получена. Пожалуйста, ознакомьтесь с кратким описанием случая ниже.",
}

# Final-turn closing when the case did NOT lead to a dispatch (clearly
# non-emergency). Still informational, no questions.
_FINAL_NON_DISPATCHED_MSG: Dict[str, str] = {
    "tr": "Teşekkürler, gerekli bilgiyi aldık ve durumunuz kayıt altına alındı. Aşağıdaki özeti inceleyebilirsiniz; acil bir değişiklik olursa hemen tekrar arayın.",
    "en": "Thank you. The necessary information has been recorded and your case is saved. Please review the summary below; call again immediately if the situation changes.",
    "de": "Danke. Die erforderlichen Informationen wurden erfasst und Ihr Fall gespeichert. Bitte sehen Sie sich die Zusammenfassung unten an.",
    "fr": "Merci. Les informations nécessaires ont été enregistrées et votre cas est sauvegardé. Veuillez consulter le résumé ci-dessous.",
    "es": "Gracias. La información necesaria ha sido registrada y su caso está guardado. Consulte el resumen a continuación.",
    "ar": "شكراً. تم تسجيل المعلومات اللازمة وحفظ حالتك. يرجى مراجعة الملخص أدناه.",
    "ru": "Спасибо. Необходимая информация получена и ваш случай сохранён. Пожалуйста, ознакомьтесь с резюме ниже.",
}


def _final_closing_text(session: "Session", lang: str) -> str:
    """Return the closing sentence to show on the very last turn.

    Always a statement (never a question). If responders were dispatched the
    message confirms that help is on the way; otherwise it acknowledges the
    case has been recorded. Report card is appended by the caller.
    """
    if session.dispatch_status in ("DISPATCHED", "SILENT_DISPATCHED"):
        return _FINAL_DISPATCHED_MSG.get(lang, _FINAL_DISPATCHED_MSG["en"])
    return _FINAL_NON_DISPATCHED_MSG.get(lang, _FINAL_NON_DISPATCHED_MSG["en"])

_URGENT_CRITICAL_SLOT_KEYS: Dict[str, List[str]] = {
    "medical": ["breathing", "consciousness", "bleeding", "duration", "duration_minutes"],
    "fire": ["trapped", "fire_size", "smoke_inhalation", "injuries"],
    "crime": ["assailant_present", "weapon", "number_injured", "victim_count", "injuries"],
    "other": ["current_danger", "symptom_severity", "risk_context"],
}


def _slot_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in ("", "unknown", "n/a"):
        return False
    return True


def _has_urgent_minimum_slots(slots: Dict[str, Any], category: str) -> bool:
    has_chief_complaint = _slot_has_value(slots.get("chief_complaint"))
    slot_keys = _URGENT_CRITICAL_SLOT_KEYS.get(category, _URGENT_CRITICAL_SLOT_KEYS["other"])
    has_critical_slot = any(_slot_has_value(slots.get(key)) for key in slot_keys)
    return has_chief_complaint and has_critical_slot


def _urgent_missing_slot_question(lang: str, slots: Dict[str, Any], category: str) -> str:
    if not _slot_has_value(slots.get("chief_complaint")):
        return {
            "tr": "Ekipleri doğru yönlendirebilmem için ana şikayeti net söyleyin: tam olarak ne oldu?",
            "en": "To route teams correctly, I need the main complaint clearly: what exactly happened?",
        }.get(lang, "Please state the main complaint clearly: what exactly happened?")

    by_category = {
        "medical": {
            "tr": "Hızlıca bir kritik bilgi daha: kişi şu an nefes alıyor mu?",
            "en": "One critical detail quickly: is the person breathing right now?",
        },
        "fire": {
            "tr": "Hızlıca bir kritik bilgi daha: içeride mahsur kalan var mı?",
            "en": "One critical detail quickly: is anyone trapped inside?",
        },
        "crime": {
            "tr": "Hızlıca bir kritik bilgi daha: saldırgan hâlâ olay yerinde mi?",
            "en": "One critical detail quickly: is the assailant still at the scene?",
        },
        "other": {
            "tr": "Hızlıca bir kritik bilgi daha: şu an devam eden bir tehlike var mı?",
            "en": "One critical detail quickly: is there an ongoing danger right now?",
        },
    }
    localized = by_category.get(category, by_category["other"])
    return localized.get(lang, localized["en"])


def _urgent_micro_location_question(lang: str) -> str:
    return {
        "tr": "Ekipleri hassas yönlendirmek için son bir bilgi: bina, kat, daire ve giriş tarifini paylaşır mısınız?",
        "en": "For precise routing, one final detail: can you share building, floor, apartment, and entrance info?",
    }.get(lang, "For precise routing, please share building, floor, apartment, and entrance details.")


# ---------------------------------------------------------------------------
# Slot validation helpers
# ---------------------------------------------------------------------------
# Groq may "extract" slots that the user never actually stated in the current
# turn (age/name/yes-no hallucinations). The helpers below keep the merge side
# honest: they drop values that have no textual evidence in the latest user
# message unless the slot already holds the same value from an earlier turn.

_DIGIT_RE = re.compile(r"\d")
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-zÇĞİıÖŞÜçğıöşü]{2,}")
_NUMERIC_SLOT_KEYS = {
    "age",
    "number_injured",
    "victim_count",
    "duration_minutes",
    "floor",
    "apartment",
}
_NAME_LIKE_SLOT_KEYS = {"caller_name"}


def _latest_user_text(session: "Session") -> str:
    """Return the most recent user message text in the session (or empty string)."""
    for msg in reversed(getattr(session, "messages", []) or []):
        if msg.get("role") == "user":
            return str(msg.get("text") or "")
    return ""


def _validate_extracted_slots(
    extracted: Dict[str, Any],
    user_text: str,
    session: "Session",
) -> Dict[str, Any]:
    """Drop slot values Groq likely hallucinated for the current turn.

    Rules:
      * empty / None values are always dropped;
      * a numeric slot (age, counts, durations) is kept only when the current
        user turn contains digits OR the value matches what we already stored;
      * a name-like slot (caller_name) is kept only when the current user turn
        contains an alphabetic token OR the value matches what we already
        stored.
    Everything else is passed through unchanged.
    """
    if not isinstance(extracted, dict) or not extracted:
        return {}
    text = (user_text or "").strip()
    has_digit = bool(_DIGIT_RE.search(text))
    has_alpha = bool(_ALPHA_TOKEN_RE.search(text))
    cleaned: Dict[str, Any] = {}
    for key, value in extracted.items():
        if key == "_asking_slot":
            continue
        if value is None:
            continue
        v_str = str(value).strip() if not isinstance(value, (list, dict)) else str(value)
        if not v_str:
            continue

        existing = session.collected_slots.get(key) if hasattr(session, "collected_slots") else None
        if existing is not None and str(existing).strip() == v_str:
            cleaned[key] = value
            continue

        if key in _NUMERIC_SLOT_KEYS and not has_digit:
            logger.info(
                "Slot validation: dropping %s=%r — no digits in user turn ('%s').",
                key, v_str, text[:60],
            )
            continue

        if key in _NAME_LIKE_SLOT_KEYS and not has_alpha:
            logger.info(
                "Slot validation: dropping %s=%r — no name-like tokens in user turn ('%s').",
                key, v_str, text[:60],
            )
            continue

        cleaned[key] = value
    return cleaned


# Unicode ranges for diacritics that never appear in Turkish/English/common languages
# but are present in Vietnamese, Thai, Arabic, etc.
_FOREIGN_DIACRITIC_RE = re.compile(
    r"[\u0300-\u036f"       # combining diacritical marks (generic)
    r"\u1e00-\u1eff"        # Latin Extended Additional (Vietnamese heavy use)
    r"\u0e00-\u0e7f"        # Thai
    r"\u0600-\u06ff"        # Arabic
    r"\u4e00-\u9fff"        # CJK
    r"\u3040-\u30ff]"       # Hiragana/Katakana
)

# Turkish has its own diacritics (ş, ğ, ü, ö, ı, ç) — whitelist them
_TURKISH_SAFE_CHARS = set("şğüöıçŞĞÜÖİÇ")


def _has_foreign_characters(text: str) -> bool:
    """Return True if text contains non-Turkish foreign script/diacritic characters."""
    for char in text:
        if _FOREIGN_DIACRITIC_RE.match(char) and char not in _TURKISH_SAFE_CHARS:
            return True
    return False


def _count_foreign_signal_words(text: str, target_lang: str) -> int:
    """Return how many high-signal tokens belong to the *wrong* language.

    Used to detect code-switching (e.g. the LLM answering in Turkish but
    mixing in English loanwords like 'okay', 'bleeding', 'emergency').
    """
    tokens = set(re.findall(r"[a-zçğıöşüA-ZÇĞİÖŞÜ']+", text.lower()))
    if not tokens:
        return 0
    if target_lang == "tr":
        return len(tokens & _ENGLISH_SIGNAL_WORDS)
    if target_lang == "en":
        wrong = tokens & _TURKISH_SIGNAL_WORDS
        # Also treat any token containing Turkish-only characters as wrong.
        wrong |= {t for t in tokens if any(c in t for c in _TURKISH_ONLY_CHARS)}
        return len(wrong)
    return 0


def _normalize_response_language(response_text: str, target_lang: str) -> str:
    """Force assistant text into the fixed session language when the LLM drifts or mixes languages."""
    text = (response_text or "").strip()
    if not text:
        return response_text

    # 1. Token-based: foreign script/diacritics → translate with auto-detected source
    if _has_foreign_characters(text):
        translated = translate(text, source="auto", target=target_lang)
        return translated or text

    # 2. Full-text language mismatch detected by langdetect
    detected = detect_language(text)
    if detected and detected != target_lang and detected in _SUPPORTED_LANGS:
        translated = translate(text, source=detected, target=target_lang)
        return translated or text

    # 3. Code-switching heuristic: response is nominally in target_lang but
    #    contains noticeable foreign-signal words (e.g. Turkish text mixed
    #    with English loanwords). Force a round-trip translation to clean it.
    foreign_hits = _count_foreign_signal_words(text, target_lang)
    if foreign_hits >= 2:
        source_lang = "en" if target_lang == "tr" else "tr"
        logger.warning(
            "Code-switching detected in LLM response (target=%s, foreign_hits=%d): %r",
            target_lang, foreign_hits, text[:120],
        )
        translated = translate(text, source=source_lang, target=target_lang)
        if translated and translated.strip() and translated != text:
            return translated

    return text


# ---------------------------------------------------------------------------
# User bail / session-end intent (fast path, regex-based)
# ---------------------------------------------------------------------------
# Kullanıcı net biçimde "sorun yok / bitir / iptal / false alarm" derse
# orchestrator LLM'e gitmeden oturumu NON_URGENT olarak kapatır. Aksi hâlde
# LLM ısrarla acil durum sorusu sormaya devam ediyor.

_BAIL_PATTERNS: Dict[str, List[str]] = {
    "tr": [
        r"\bsorun\s+yok\b",
        r"\bproblem\s+yok\b",
        r"\bbir\s+şey\s+yok\b",
        r"\bbi\s*şey\s+yok\b",
        r"\bönemli\s+değil\b",
        r"\bacil\s+değil\b",
        r"\bgerek\s+yok\b",
        r"\biyiyim\b",
        r"\biyidir\b",
        r"\biyiyiz\b",
        r"\bboş\s*ver\b",
        r"\byanlış(lık(la)?| oldu| arama| alarm)?\b",
        r"\b(oturumu|görüşmeyi|konuşmayı|aramayı|sohbeti)\s+(bitir|kapat|sonlandır|iptal\s+et)\w*\b",
        r"\b(bitir|kapat|sonlandır|iptal)\s*(elim|alım|sin|abilir\s*misin)?\b",
        r"\bhayır\s*,?\s*(sorun|problem|bir\s*şey)\s+yok\b",
        r"\bdur(duralım)?\b\s*,?\s*(gerek|lazım)?\s*yok\b",
        r"\bsadece\s+(test|deneme)\b",
    ],
    "en": [
        r"\bno\s+problem\b",
        r"\bnothing\s+wrong\b",
        r"\bnothing\s+is\s+wrong\b",
        r"\bnever\s*mind\b",
        r"\bi\s*'?\s*m\s+fine\b",
        r"\bi\s+am\s+fine\b",
        r"\bi\s*'?\s*m\s+ok(ay)?\b",
        r"\bi\s+am\s+ok(ay)?\b",
        r"\bwe\s*'?\s*re\s+fine\b",
        r"\bwe\s+are\s+fine\b",
        r"\bfalse\s+alarm\b",
        r"\b(end|terminate|cancel|close|stop)\s+(the\s+)?(session|call|chat|conversation)\b",
        r"\bjust\s+(a\s+)?(test|trial|kidding|joking)\b",
        r"\bno\s+emergency\b",
        r"\bnot\s+an\s+emergency\b",
        r"\bno\s+need\b",
        r"\bnot\s+needed\b",
    ],
}


def _is_user_bail_intent(text: str, lang: str) -> bool:
    """Return True when the user clearly signals they want to end the session."""
    if not text:
        return False
    t = text.strip().lower()
    if not t:
        return False
    patterns = _BAIL_PATTERNS.get(lang) or _BAIL_PATTERNS["en"]
    for pattern in patterns:
        if re.search(pattern, t, re.IGNORECASE):
            return True
    # Very short single-word closure intents, independent of session language.
    compact = re.sub(r"[^a-zçğıöşü\s]", "", t)
    compact = re.sub(r"\s+", " ", compact).strip()
    closure_single_tokens = {
        "bitir", "sonlandır", "sonlandir", "kapat", "iptal", "dur", "tamam bitti",
        "stop", "end", "cancel", "quit", "exit", "close", "done",
    }
    if compact in closure_single_tokens:
        return True
    return False


def _handle_user_bail(
    session: Session,
    lang: str,
    asr_transcript: Optional[str],
) -> Dict[str, Any]:
    """Close the session politely as NON_URGENT when the user bails out."""
    already_dispatched = session.dispatch_status in ("DISPATCHED", "SILENT_DISPATCHED")

    if already_dispatched:
        triage = dict(session.triage_result or {})
        triage.setdefault("category", session.dispatch_target or "other")
        triage.setdefault("triage_level", "URGENT")
    else:
        base_category = (session.triage_result or {}).get("category") or (
            session.initial_triage or {}
        ).get("category", "other")
        triage = {
            "category": base_category,
            "triage_level": "NON_URGENT",
            "confidence": 1.00,
            "red_flags": [],
            "slots": session.collected_slots,
            "user_bailed": True,
        }
        session.dispatch_status = "CANCELLED"

    triage["slots"] = session.collected_slots
    session.triage_result = triage
    session.is_complete = True
    session.pending_update_after_dispatch = False

    if already_dispatched:
        close_msg = {
            "tr": "Anladım, oturumu burada kapatıyorum. Ekipler yoldadır; durum kötüleşirse 112'yi arayın.",
            "en": "Understood, I'm closing this session. Responders are on the way; call 112 if anything changes.",
        }.get(lang, "Understood, closing the session. Call 112 if the situation changes.")
    else:
        close_msg = {
            "tr": "Anladım, şu anda acil bir durum olmadığını belirttiniz. Oturumu kapatıyorum. Gerçek bir acil durumda 112'yi arayın ya da sohbeti yeniden başlatın. İyi günler.",
            "en": "Understood, you've indicated there is no emergency right now. I'm closing this session. In a real emergency, please call 112 or start a new chat. Take care.",
        }.get(lang, "Understood, closing the session. Call 112 in a real emergency.")

    # Bail durumlarında kısa kapanış yeterli; ek GPT/şablon raporu üretmiyoruz.
    # Ancak halihazırda dispatch gerçekleşmişse (ekipler yoldaysa) özet raporu
    # koruyarak kullanıcıya gönderelim.
    report_local: Optional[str] = None
    if already_dispatched:
        report_local = _compose_session_report(
            triage_result=triage,
            slots=session.collected_slots,
            image_analysis=session.image_analysis,
            lang=lang,
        )
    final_text = f"{close_msg}\n\n{report_local}" if report_local else close_msg

    if is_turn_trace_enabled():
        trace_step(
            "Kullanıcı bail intent",
            "Kullanıcı oturumu kapatmak istedi → NON_URGENT ile kapanış",
        )
        trace_orchestrator_outcome(
            triage_level=str(triage.get("triage_level", "")),
            category=str(triage.get("category", "")),
            dispatch_status=str(session.dispatch_status or ""),
            dispatch_target=session.dispatch_target,
            is_complete=True,
            user_turn_count=sum(1 for m in session.messages if m.get("role") == "user"),
        )

    return _reply(
        session,
        final_text,
        triage_result=triage,
        image_analysis=session.image_analysis,
        report=report_local,
        is_complete=True,
        user_transcript=asr_transcript,
        tts_text=close_msg,
    )


def _is_gibberish(text: str) -> bool:
    """Minimal hard-noise filter. Ambiguous cases are delegated to LLM."""
    t = (text or "").strip().lower()
    if not t:
        return True

    # Very short random chunks are typically noise.
    if len(t) <= 2:
        return True

    # Mostly non-letter input (symbols/digits) is likely gibberish.
    letters = sum(ch.isalpha() for ch in t)
    if letters == 0:
        return True
    if letters / max(len(t), 1) < 0.35:
        return True

    # Repeated single-character spam (aaaa, zzzz, 1111).
    compact = re.sub(r"\s+", "", t)
    if len(compact) >= 4 and len(set(compact)) == 1:
        return True

    # Keyboard mashing patterns.
    mash_markers = ("asdf", "qwer", "zxcv", "qaz", "wsx")
    if any(m in compact for m in mash_markers):
        return True

    return False


def _is_gibberish_with_llm(text: str, lang: str) -> Optional[bool]:
    """Ask LLM whether input is gibberish. Returns None if decision unavailable."""
    try:
        from services.llm_service import get_llm_service

        llm = get_llm_service()
        if not llm.is_available:
            return None

        compact = re.sub(r"\s+", " ", (text or "").strip())
        if _uses_single_call_first_turn(llm) and len(compact) > 12:
            return None

        result = llm.chat(
            history=[{"role": "user", "text": text}],
            language=lang,
            task="gibberish_check",
        )
        marker = str((result.get("extracted_slots") or {}).get("meaningfulness", "")).strip().lower()
        if marker in ("gibberish", "noise", "nonsense"):
            return True
        if marker in ("meaningful", "valid"):
            return False
        return None
    except Exception as exc:
        logger.debug("LLM gibberish check skipped: %s", exc)
        return None


def handle_message(
    session_id: str,
    user_text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    image_bytes: Optional[bytes] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Dict[str, Any]:
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        if is_turn_trace_enabled():
            trace_step("Hata", "Oturum bulunamadı veya süresi doldu")
        return {"error": "Session not found or expired."}
    
    # ------------------------------------------------------------------
    # FAZ 7: Timeout & Silent Dispatch (3-minute inactivity rule)
    # If session is waiting for response and 3+ minutes passed with no activity:
    # CRITICAL/URGENT → auto-dispatch (silent dispatch)
    # User returning after timeout → resume mode
    # ------------------------------------------------------------------
    current_time = time.time()
    TIMEOUT_SECONDS = 3 * 60

    # Sliding inactivity window: deadline is always relative to the last user activity,
    # not to the first message of the session. Recompute every turn so that ongoing
    # conversations don't trip the 3-minute rule just because total session age > 180s.
    session.timeout_deadline = session.last_user_activity_at + TIMEOUT_SECONDS
    inactivity_elapsed = current_time - session.last_user_activity_at
    logger.debug(
        "Timeout check: inactivity=%.0fs, deadline=%.0f, current=%.0f",
        inactivity_elapsed, session.timeout_deadline, current_time,
    )

    # Check if timeout has been triggered
    if session.dispatch_status in ("PENDING", "FALLBACK_PENDING") and not session.resumed_after_timeout:
        if inactivity_elapsed > TIMEOUT_SECONDS:
            # Timeout triggered
            if session.triage_result:
                triage_level = session.triage_result.get("triage_level", "NON_URGENT")
                if triage_level in ("CRITICAL", "URGENT"):
                    # Silent dispatch for critical/urgent cases
                    session.dispatch_status = "SILENT_DISPATCHED"
                    session.dispatch_target = session.triage_result.get("category", "other")
                    session.dispatch_timestamp = current_time
                    logger.warning(
                        "Silent dispatch triggered: inactivity %.0fs (> %ds). vaka_id=%s, category=%s",
                        inactivity_elapsed, TIMEOUT_SECONDS,
                        session.session_id, session.dispatch_target,
                    )
                    
                    # Prepare timeout message
                    timeout_msg = {
                        "tr": "Yanıt alamadığım için yardım ekiplerini gönderdim. Lütfen güvende kalın.",
                        "en": "No response detected. Emergency services have been dispatched. Please stay safe.",
                    }.get(session.language or "en", "Emergency services dispatched.")
                    
                    if is_turn_trace_enabled():
                        trace_step(
                            "Zaman aşımı",
                            "3 dk sessizlik → CRITICAL/URGENT için SILENT_DISPATCHED + oturum kapanışı",
                        )
                    return _reply(session, timeout_msg, triage_result=session.triage_result, 
                                is_complete=True, report=None)
    
    # User returning after timeout → resume mode
    if inactivity_elapsed > TIMEOUT_SECONDS and not session.resumed_after_timeout:
        session.resumed_after_timeout = True
        logger.info(
            "Session resumed after timeout: vaka_id=%s, inactivity=%.0fs",
            session.session_id, inactivity_elapsed,
        )

    # Guard: zaten tamamlanmış session'a yeni mesaj gelirse raporla cevap ver
    if session.is_complete:
        if session.dispatch_status in ("DISPATCHED", "SILENT_DISPATCHED") and (
            image_bytes or (user_text and user_text.strip()) or audio_bytes
        ):
            if image_bytes:
                session.image_bytes = image_bytes
                _run_image_analysis(session)
            return _handle_post_dispatch_update(
                session,
                user_text or "",
                lang=session.language or "en",
            )
        done_msg = {
            "tr": "Bu oturum tamamlandı. Yeni bir acil durum için lütfen yeniden başlatın.",
            "en": "This session is already complete. Please start a new session for a new emergency.",
        }.get(session.language or "en",
              "This session is already complete. Please start a new session.")
        if is_turn_trace_enabled():
            trace_step("Oturum zaten tamamlandı", "Yeni mesaja kısa bilgi yanıtı")
        return _reply(session, done_msg, triage_result=session.triage_result,
                      report=None, is_complete=True)

    lang = session.language or "en"
    asr_transcript: Optional[str] = None

    # ------------------------------------------------------------------
    # 1. Image
    # ------------------------------------------------------------------
    if image_bytes:
        session.image_bytes = image_bytes
        _run_image_analysis(session)

    # ------------------------------------------------------------------
    # 2. ASR
    # ------------------------------------------------------------------
    if audio_bytes and not user_text:
        try:
            from services.asr_service import transcribe_audio

            t0 = time.monotonic()
            asr_lang_hint = lang if session.language_locked else None
            transcript, detected_lang, _conf = transcribe_audio(
                audio_bytes=audio_bytes, language=asr_lang_hint,
            )
            logger.info(
                "  [TIMING] ASR: %.2fs %s translation_backend=%s",
                time.monotonic() - t0,
                get_asr_runtime_info_str(),
                get_translation_backend_name(),
            )
            user_text = transcript
            asr_transcript = transcript
            # Only lock from audio if we actually got a detected language AND
            # a non-empty transcript — otherwise fall through so the text path
            # (or the next turn) can do the detection.
            if detected_lang and transcript and transcript.strip() and not session.language_locked:
                resolved = _clamp_supported_lang(detected_lang)
                session.language = resolved
                session.language_locked = True
                lang = resolved
                logger.info(
                    "Language locked from first audio: raw=%s -> %s",
                    detected_lang, resolved,
                )
        except Exception as exc:
            logger.error("ASR failed: %s", exc)
            asr_err = {
                "tr": "Sesi anlayamadım, lütfen tekrar deneyin veya mesajı yazarak gönderin.",
                "en": "Sorry, I could not understand the audio. Please try again or type your message.",
                "de": "Ich konnte das Audio nicht verstehen. Bitte versuchen Sie es erneut oder tippen Sie Ihre Nachricht.",
                "fr": "Je n'ai pas pu comprendre l'audio. Veuillez réessayer ou taper votre message.",
                "es": "No pude entender el audio. Por favor intente de nuevo o escriba su mensaje.",
            }.get(lang, "Sorry, I could not understand the audio. Please try again.")
            return _reply(session, asr_err, user_transcript=asr_transcript)

    # ------------------------------------------------------------------
    # 3. Guard: nothing received
    # ------------------------------------------------------------------
    if not user_text or not user_text.strip():
        if image_bytes and session.image_analysis:
            return _handle_image_only(session, lang, asr_transcript)
        return _reply(
            session,
            "I didn't receive any input. Could you please describe your emergency?",
            user_transcript=asr_transcript,
        )

    # ------------------------------------------------------------------
    # 4. Noise / gibberish (hard filter, then optional LLM check)
    # ------------------------------------------------------------------
    hard_noise = _is_gibberish(user_text)
    is_noise = hard_noise
    if not is_noise:
        llm_noise = _is_gibberish_with_llm(user_text, lang)
        if llm_noise is True:
            is_noise = True

    if is_noise:
        session.troll_count += 1
        if session.troll_count >= 2:
            session.is_complete = True
            close_msg = {
                "tr": "Anlamlı bir acil durum bilgisi alamadım. Oturumu kapatıyorum. Gerçek acil durumda lütfen yeniden yazın veya 112'yi arayın.",
                "en": "I could not get meaningful emergency details. I am closing this session. In a real emergency, please start again or call 112.",
            }.get(lang, "I could not get meaningful emergency details. Session closed.")
            return _reply(session, close_msg, is_complete=True, user_transcript=asr_transcript)

        clarify_msg = {
            "tr": "Mesajı anlayamadım. Lütfen acil durumu kısa ve net yazın (örnek: 'Babam nefes almıyor').",
            "en": "I could not understand. Please describe the emergency clearly (example: 'My father is not breathing').",
        }.get(lang, "Please describe the emergency clearly.")
        return _reply(session, clarify_msg, user_transcript=asr_transcript)

    # Reset noise counter once meaningful text is received.
    session.troll_count = 0

    # Session language is always derived from the user's first input (audio or
    # text), restricted to Turkish/English, and then fixed for the rest of the
    # conversation. We prefer our hybrid TR/EN classifier here because
    # ``langdetect`` alone is unreliable on short emergency utterances like
    # "my dad is bleeding" or "help fire".
    if not session.language_locked:
        resolved = _detect_tr_or_en(user_text)
        raw_detected = detect_language(user_text)
        session.language = resolved
        session.language_locked = True
        lang = resolved
        logger.info(
            "Language locked from text: raw_langdetect=%s -> resolved=%s (text=%r)",
            raw_detected, resolved, user_text[:80],
        )

    # ------------------------------------------------------------------
    # 5. Accumulate English text for ML models (sentiment etc.)
    # ------------------------------------------------------------------
    text_en = translate_to_english(user_text, source_lang=lang) if lang != "en" else user_text
    session.text_en_accumulated += " " + text_en
    session.text_en_accumulated = session.text_en_accumulated.strip()

    # Store GPS location if provided (4. GPS entegrasyonu)
    if latitude is not None and longitude is not None:
        session.collected_slots["latitude"] = latitude
        session.collected_slots["longitude"] = longitude
        logger.info("GPS location stored: %.6f, %.6f", latitude, longitude)

    # Update last user activity timestamp (for timeout tracking)
    session.last_user_activity_at = time.time()
    logger.debug("Last user activity: %.0f", session.last_user_activity_at)

    # Add user message to history
    session.messages.append({"role": "user", "text": user_text})

    # ------------------------------------------------------------------
    # Bail intent fast-path: "sorun yok / bitir / sonlandır / no problem /
    # false alarm" gibi net vazgeçme sinyalleri geldiğinde LLM'e gitmeden
    # oturumu NON_URGENT olarak kapat. LLM ısrarla acil soru sormasın diye.
    # ------------------------------------------------------------------
    if _is_user_bail_intent(user_text, lang):
        logger.info("User bail intent detected (lang=%s). Closing session as NON_URGENT.", lang)
        if is_turn_trace_enabled():
            trace_step(
                "Bail intent algılandı",
                f"user_text='{user_text}' → LLM atlanıyor, oturum kapanıyor",
            )
        return _handle_user_bail(session, lang, asr_transcript)

    if is_turn_trace_enabled():
        trace_banner("Kullanıcı turu", session.session_id)
        trace_step("Mesaj sayısı", f"session.messages={len(session.messages)}")
        trace_kv(
            "Girdi özeti",
            {
                "lang": lang,
                "user_text": user_text,
                "asr_transcript": asr_transcript or "",
                "text_en_birikim": session.text_en_accumulated,
                "gps": (latitude, longitude),
            },
        )

    # ------------------------------------------------------------------
    # 6. Sentiment analysis (needs audio for best results)
    # ------------------------------------------------------------------
    if audio_bytes:
        _run_sentiment_analysis(session, text_en, audio_bytes)
        if is_turn_trace_enabled():
            trace_kv("Ses / duygu (sentiment)", dict(session.sentiment_result or {}))

    # ------------------------------------------------------------------
    # 7. LLM turn (or fallback to rule-based)
    # ------------------------------------------------------------------
    from services.llm_service import get_llm_service
    llm = get_llm_service()

    if is_turn_trace_enabled():
        trace_step(
            "Karar",
            "LLM yolu (OpenAI triage + Groq diyalog)" if llm.is_available else "Kural tabanlı yol (LLM eksik)",
        )

    if llm.is_available:
        return _handle_with_llm(session, lang, asr_transcript)
    else:
        return _handle_with_rules(session, lang, asr_transcript)


def _handle_image_only(
    session: Session,
    lang: str,
    asr_transcript: Optional[str],
) -> Dict[str, Any]:
    """Fotoğraf tek başına (metinsiz) geldiğinde triyaj tamamen görüntü modeline bırakılır.

    Kullanıcıdan hiçbir durumda metin açıklaması istenmez; görüntü modelinin
    kararı doğrudan uygulanır:
      - CRITICAL / URGENT  → doğrudan dispatch
      - NON_URGENT         → oturumu acil değil olarak kapat
      - Model kararsız (MANUAL_FALLBACK / RECAPTURE_IMAGE) → yalnızca daha net
        yeni bir fotoğraf istenir; 2. denemeden sonra modelin en iyi tahminine
        göre sonuçlandırılır.
    """
    image_analysis = session.image_analysis or {}
    visual = image_analysis.get("visual_triage") or {}
    action = visual.get("action")
    triage_level = visual.get("triage_level", "URGENT")
    category = visual.get("category", "other")

    # ------------------------------------------------------------------
    # Model kararsız / görsel kalitesi yetersiz → YENİ FOTOĞRAF iste (metin DEĞİL).
    # ------------------------------------------------------------------
    if action in ("RECAPTURE_IMAGE", "MANUAL_FALLBACK"):
        session.image_attempt_count += 1
        if session.image_attempt_count < 2:
            msg = {
                "tr": "Görseli net analiz edemedim. Lütfen daha iyi aydınlatılmış, daha net bir fotoğraf çekip tekrar gönderin.",
                "en": "I could not analyze the photo clearly. Please take a better-lit, clearer photo and send it again.",
            }.get(lang, "Please send a clearer photo.")
            return _reply(
                session,
                msg,
                image_analysis=image_analysis,
                user_transcript=asr_transcript,
            )
        # 2+ deneme: modelin elindeki en iyi tahmine göre karar ver, metin İSTEME.
        triage = _triage_from_visual(image_analysis)
        session.triage_result = triage
        session.collected_slots.update({
            "image_category": category,
            "image_triage_level": triage_level,
            "visual_flags": visual.get("visual_flags", []),
        })
        if triage_level in ("CRITICAL", "URGENT"):
            _mark_dispatch(session, category)
            msg = {
                "tr": "Görseli tam net analiz edemesem de olası bir acil durum tespit ettim. Ekipler yönlendiriliyor. Güvenli alanda kalın.",
                "en": "I could not fully analyze the photo, but it suggests a possible emergency. Responders are being dispatched. Stay in a safe area.",
            }.get(lang, "Responders are being dispatched based on the image.")
            report_local = _compose_session_report(
                triage_result=session.triage_result,
                slots=session.collected_slots,
                image_analysis=image_analysis,
                lang=lang,
            )
            final_text = f"{msg}\n\n{report_local}" if report_local else msg
            return _reply(
                session,
                final_text,
                triage_result=session.triage_result,
                image_analysis=image_analysis,
                report=report_local,
                user_transcript=asr_transcript,
            )
        session.is_complete = True
        session.dispatch_status = session.dispatch_status or "CANCELLED"
        msg = {
            "tr": "Görselde net bir acil durum belirtisi tespit edemedim. Oturumu kapatıyorum. Gerçek bir acil durumda lütfen 112'yi arayın.",
            "en": "I could not detect a clear emergency in the image. Closing the session. In a real emergency please call 112.",
        }.get(lang, "No clear emergency detected. Session closed.")
        report_local = _compose_session_report(
            triage_result=session.triage_result,
            slots=session.collected_slots,
            image_analysis=image_analysis,
            lang=lang,
        )
        final_text = f"{msg}\n\n{report_local}" if report_local else msg
        return _reply(
            session,
            final_text,
            triage_result=session.triage_result,
            image_analysis=image_analysis,
            report=report_local,
            is_complete=True,
            user_transcript=asr_transcript,
        )

    # ------------------------------------------------------------------
    # Model net bir karar verdi → triyajı görselden türet.
    # ------------------------------------------------------------------
    triage = _triage_from_visual(image_analysis)
    session.triage_result = triage
    session.collected_slots.update({
        "image_category": category,
        "image_triage_level": triage_level,
        "visual_flags": visual.get("visual_flags", []),
    })

    # CRITICAL ya da URGENT → doğrudan dispatch. Metin doğrulaması İSTENMEZ.
    if action in ("EARLY_DISPATCH", "VERIFY_THEN_DISPATCH") or triage_level in ("CRITICAL", "URGENT"):
        _mark_dispatch(session, category)
        if action == "EARLY_DISPATCH" or triage_level == "CRITICAL":
            msg = {
                "tr": "Görselde kritik risk tespit edildi. Ekipler yönlendiriliyor. Güvenli alana geçin; konum gönderebilirseniz ekiplere ileteceğim.",
                "en": "Critical risk detected in the image. Responders are being dispatched. Move to a safe area; share your location if you can.",
            }.get(lang, "Critical risk detected. Responders are being dispatched.")
        else:
            msg = {
                "tr": "Görselde acil bir durum tespit edildi. Ekipler yönlendiriliyor. Güvenli alana geçin; konum gönderebilirseniz ekiplere ileteceğim.",
                "en": "An emergency was detected in the image. Responders are being dispatched. Move to a safe area; share your location if you can.",
            }.get(lang, "Emergency detected. Responders are being dispatched.")
        report_local = _compose_session_report(
            triage_result=session.triage_result,
            slots=session.collected_slots,
            image_analysis=image_analysis,
            lang=lang,
        )
        final_text = f"{msg}\n\n{report_local}" if report_local else msg
        return _reply(
            session,
            final_text,
            triage_result=session.triage_result,
            image_analysis=image_analysis,
            report=report_local,
            user_transcript=asr_transcript,
        )

    # NON_URGENT (action == "TEXT_REQUIRED" dahil) → metin İSTEMEDEN kapat.
    session.is_complete = True
    session.dispatch_status = session.dispatch_status or "CANCELLED"
    msg = {
        "tr": "Görselde acil bir durum belirtisi tespit etmedim. Oturumu kapatıyorum. Gerçek bir acil durumda lütfen 112'yi arayın.",
        "en": "I did not detect any emergency signals in the image. Closing the session. In a real emergency please call 112.",
    }.get(lang, "No emergency detected in the image. Session closed.")
    report_local = _compose_session_report(
        triage_result=session.triage_result,
        slots=session.collected_slots,
        image_analysis=image_analysis,
        lang=lang,
    )
    final_text = f"{msg}\n\n{report_local}" if report_local else msg
    return _reply(
        session,
        final_text,
        triage_result=session.triage_result,
        image_analysis=image_analysis,
        report=report_local,
        is_complete=True,
        user_transcript=asr_transcript,
    )


def _triage_from_visual(image_analysis: Dict[str, Any]) -> Dict[str, Any]:
    visual = image_analysis.get("visual_triage") or {}
    classification = image_analysis.get("classification") or {}
    return {
        "category": visual.get("category", classification.get("mapped_category", "other")),
        "triage_level": visual.get("triage_level", "URGENT"),
        "confidence": classification.get("confidence"),
        "red_flags": visual.get("visual_flags", []),
        "slots": {
            "image_detected_class": classification.get("detected_class"),
            "image_action": visual.get("action"),
            "image_quality": image_analysis.get("image_quality"),
        },
        "needs_more_info": visual.get("action") != "EARLY_DISPATCH",
        "recommended_questions": _visual_questions(visual),
        "image_analysis": image_analysis,
    }


def _visual_questions(visual: Dict[str, Any]) -> List[str]:
    action = visual.get("action")
    if action == "EARLY_DISPATCH":
        return ["Exact location/building/floor?", "Immediate visible danger?"]
    if action == "VERIFY_THEN_DISPATCH":
        return ["Any injured/trapped person?", "Is fire/smoke/weapon still present?"]
    if action in ("MANUAL_FALLBACK", "RECAPTURE_IMAGE"):
        return ["Choose incident category.", "Add one short description."]
    return ["Briefly describe what happened."]


def _mark_dispatch(session: Session, category: str) -> None:
    session.dispatch_status = "DISPATCHED"
    session.dispatch_target = category or "other"
    session.dispatch_timestamp = time.time()
    session.pending_update_after_dispatch = True


def _compose_session_report(
    triage_result: Optional[Dict[str, Any]],
    slots: Dict[str, Any],
    image_analysis: Optional[Dict[str, Any]],
    lang: str,
) -> str:
    """Template report, or OpenAI (GPT) when USE_OPENAI_FINAL_REPORT=true and key set."""
    tr = triage_result or {}
    gpt = generate_final_report_openai(
        triage_result=tr,
        slots=slots,
        image_analysis=image_analysis,
        language=lang,
    )
    if gpt:
        return gpt
    return compose_report(
        triage_result=tr,
        slots=slots,
        image_analysis=image_analysis,
        language=lang,
    )


def _handle_post_dispatch_update(session: Session, user_text: str, lang: str) -> Dict[str, Any]:
    image_analysis = session.image_analysis
    update = {
        "text": user_text,
        "image_analysis": image_analysis,
        "timestamp": time.time(),
    }
    session.image_updates.append(update)
    visual = (image_analysis or {}).get("visual_triage") or {}
    if visual.get("triage_level") == "CRITICAL" and session.triage_result:
        session.triage_result["triage_level"] = "CRITICAL"
        session.triage_result.setdefault("red_flags", [])
        for flag in visual.get("visual_flags", []):
            if flag not in session.triage_result["red_flags"]:
                session.triage_result["red_flags"].append(flag)
        if not session.critical_locked:
            logger.info(
                "CRITICAL lock activated via visual triage (session=%s).",
                session.session_id,
            )
        session.critical_locked = True

    msg = {
        "tr": "Güncelleme alındı. Bu bilgi yoldaki ekiplere ek bilgi olarak iletilecek. Yeni risk varsa güvenli alanda kalın.",
        "en": "Update received. This will be forwarded as additional information to the responding team. Stay in a safe area if risk remains.",
    }.get(lang, "Update received.")
    return _reply(
        session,
        msg,
        triage_result=session.triage_result,
        image_analysis=image_analysis,
        is_complete=True,
    )


# ---------------------------------------------------------------------------
# LLM-powered turn
# ---------------------------------------------------------------------------

def _handle_with_llm(
    session: Session,
    lang: str,
    asr_transcript: Optional[str],
) -> Dict[str, Any]:
    from services.llm_service import get_llm_service
    from orchestrator.session import truncate_message_history, can_redispatch

    llm = get_llm_service()

    # ------------------------------------------------------------------
    # FAZ 3.1: Message History Truncation (Groq Context Limit ~8000 tokens)
    # Keep only last 8-10 turns for Groq context window
    # ------------------------------------------------------------------
    truncate_message_history(session, max_turns=8)
    logger.debug("Message history truncated. Keeping last 8 turns. Total: %d",
                 len(session.message_history))

    # ------------------------------------------------------------------
    # Every turn: OpenAI fine-tuned triage (full history) → authoritative category/level/red_flags.
    # Then Groq dialog → user-facing response_text and slots only (triage fields from Groq ignored).
    # ------------------------------------------------------------------
    user_turn_count = sum(1 for m in session.messages if m.get("role") == "user")
    exhausted_slots = [k for k, v in session.slot_attempt_counts.items() if v >= 2]

    if is_turn_trace_enabled():
        trace_step(
            "_handle_with_llm",
            f"user_turns={user_turn_count} | "
            "akış: OpenAI triage → Groq dialog → orchestrator güvenlik kuralları",
        )

    # Triage (OpenAI FT) and dialog (Groq) are run in parallel: the dialog
    # call only uses initial_category/level as a soft hint, and the orchestrator
    # re-applies the authoritative triage after both return. Feeding the
    # PREVIOUS turn's triage as the hint keeps behaviour consistent while
    # eliminating ~1–2 s of sequential wait.
    prior_triage = session.initial_triage or session.triage_result or {}
    dialog_hint_category = str(prior_triage.get("category") or "other")
    dialog_hint_level = str(prior_triage.get("triage_level") or "URGENT")

    history_snapshot = list(session.messages)
    dispatch_status_snapshot = session.dispatch_status
    witness_mode_snapshot = session.witness_mode

    def _run_triage() -> Dict[str, Any]:
        return llm.chat(
            history=history_snapshot,
            language=lang,
            task="triage",
        )

    def _run_dialog() -> Dict[str, Any]:
        return llm.chat(
            history=history_snapshot,
            language=lang,
            task="dialog",
            session_context={
                "initial_category": dialog_hint_category,
                "initial_triage_level": dialog_hint_level,
                "dispatch_status": dispatch_status_snapshot,
                "witness_mode": witness_mode_snapshot,
                "exhausted_slots": exhausted_slots,
            },
        )

    logger.info("OpenAI fine-tuned triage + Groq dialog (parallel)...")
    t_parallel = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_triage = pool.submit(_run_triage)
        fut_dialog = pool.submit(_run_dialog)
        triage_result = fut_triage.result()
        llm_result = fut_dialog.result()
    logger.info("  [TIMING] Triage+Dialog (parallel): %.2fs", time.monotonic() - t_parallel)

    session.initial_triage = {
        "category": triage_result.get("category", "other"),
        "triage_level": triage_result.get("triage_level", "URGENT"),
        "confidence": triage_result.get("confidence", 1.00),
        "red_flags": list(triage_result.get("red_flags") or []),
    }
    session.witness_mode = bool(triage_result.get("is_witness", False))

    # ------------------------------------------------------------------
    # Monotonic CRITICAL lock
    # ------------------------------------------------------------------
    # Bir vakaya CRITICAL kararı verildikten (ve/veya sevk başlatıldıktan)
    # sonra kullanıcının nötr onay mesajları ("tamam", "bekliyorum",
    # "anladım" vb.) modeli yeni turda URGENT/NON_URGENT'a çekebiliyor.
    # Bu geriye dönük düşüş operasyonel açıdan tehlikelidir: sevk edilmiş
    # bir ekip için vaka hâlâ CRITICAL seviyededir.
    # Kural:
    #   1) initial_triage CRITICAL ise lock'u aktive et (kalıcı).
    #   2) Lock aktifken model URGENT/NON_URGENT dönse bile seviyeyi
    #      CRITICAL olarak koru. (Sentiment yukarı yönde override hâlâ
    #      çalışabilir; aşağı yönde yumuşatma yapılamaz.)
    raw_level = session.initial_triage.get("triage_level", "URGENT")
    if raw_level == "CRITICAL":
        if not session.critical_locked:
            logger.info(
                "CRITICAL lock activated for session %s (first CRITICAL commit).",
                session.session_id,
            )
        session.critical_locked = True
    elif session.critical_locked:
        logger.info(
            "CRITICAL lock active: overriding model level %s back to CRITICAL "
            "(session=%s). Neutral acknowledgments must not downgrade a committed CRITICAL case.",
            raw_level,
            session.session_id,
        )
        session.initial_triage["triage_level"] = "CRITICAL"
        session.initial_triage["critical_locked"] = True

    logger.info(
        "Triage (OpenAI FT): category=%s, level=%s",
        session.initial_triage["category"],
        session.initial_triage["triage_level"],
    )

    if is_turn_trace_enabled():
        trace_step(
            "Groq diyalog (ham)",
            f"is_complete={llm_result.get('is_complete')} | "
            f"dispatch_action={llm_result.get('dispatch_action')} | "
            f"legal_close={llm_result.get('legal_close')} | "
            "not: Groq triage_level/category yanıtta kullanılmıyor; yetkili sınıf OpenAI triage",
        )

    response_text: str = llm_result.get("response_text", "")
    extracted_slots: Dict[str, Any] = llm_result.get("extracted_slots", {})
    is_complete: bool = llm_result.get("is_complete", False)
    dispatch_action: str = str(llm_result.get("dispatch_action", "none") or "none").strip().lower()
    legal_close: bool = bool(llm_result.get("legal_close", False))

    # Authoritative triage from OpenAI (not Groq)
    triage_level = str((session.initial_triage or {}).get("triage_level", "URGENT"))
    category = str((session.initial_triage or {}).get("category", "other"))
    red_flags: List[str] = list((session.initial_triage or {}).get("red_flags") or [])

    if legal_close and triage_level == "NON_URGENT":
        is_complete = True

    # ------------------------------------------------------------------
    # FAZ 5: Slot Attempt Tracking (2-Attempt Rule) + _asking_slot signal
    # ------------------------------------------------------------------
    # Groq signals which slot it is asking about *this* turn via
    # extracted_slots["_asking_slot"]. We:
    #   1) Pop that marker so it never leaks into session.collected_slots.
    #   2) Use session.pending_question_key (set LAST turn by us) to decide
    #      whether the previous question was actually answered; if not,
    #      increment its attempt counter (backend half of the 2-attempt rule).
    #   3) Drop hallucinated slot values that have no evidence in the user's
    #      current turn (e.g. age without digits, caller_name without letters).
    #   4) Record the new _asking_slot as pending_question_key for next turn.
    # ------------------------------------------------------------------
    next_asking_slot: Optional[str] = None
    if isinstance(extracted_slots, dict):
        raw_asking = extracted_slots.pop("_asking_slot", None)
        if isinstance(raw_asking, str) and raw_asking.strip():
            next_asking_slot = raw_asking.strip()

    if session.pending_question_key and session.pending_question_key not in extracted_slots:
        from orchestrator.dialog_manager import increment_slot_attempt
        increment_slot_attempt(session, session.pending_question_key)
        logger.debug(
            "Slot %s not filled by LLM, incrementing attempt counter (current: %d)",
            session.pending_question_key,
            session.slot_attempt_counts.get(session.pending_question_key, 0),
        )

    extracted_slots = _validate_extracted_slots(
        extracted_slots,
        user_text=_latest_user_text(session),
        session=session,
    )

    session.pending_question_key = next_asking_slot
    if next_asking_slot:
        logger.debug("Pending question slot recorded for next turn: %s", next_asking_slot)

    # ------------------------------------------------------------------
    # Guard 2 — CRITICAL immediate dispatch (no confirmation wait)
    # Dispatch at once on CRITICAL triage — do NOT wait for is_complete.
    # Report card is shown later when LLM exhausts its slot questions.
    # ------------------------------------------------------------------
    # Tracks whether dispatch happened in THIS turn (any triage level). Used
    # by the post-dispatch completion block to avoid closing on the very turn
    # the dispatch message is shown.
    dispatched_this_turn = False

    if triage_level == "CRITICAL" and session.dispatch_status == "PENDING":
        if can_redispatch(session, redispatch_ttl_seconds=48 * 3600):
            session.dispatch_status = "DISPATCHED"
            session.dispatch_target = category
            session.dispatch_timestamp = time.time()
            session.pending_update_after_dispatch = True
            session.post_dispatch_turn_count = 0
            dispatched_this_turn = True
            logger.info("CRITICAL: Immediate dispatch triggered. vaka_id=%s, target=%s",
                        session.session_id, category)
            dispatch_notice = _DISPATCH_MSG.get(lang, _DISPATCH_MSG["en"])
            if response_text and dispatch_notice not in response_text:
                response_text = dispatch_notice + " " + response_text
            elif not response_text:
                response_text = dispatch_notice
        else:
            logger.warning("Dispatch lock active: CRITICAL case already dispatched. Skipping.")

    # ------------------------------------------------------------------
    # Guard 3 — Maximum turn limit (Dynamic based on triage_level)
    # FAZ 6: Operasyonel Mantik — Escalation Control
    # CRITICAL: dispatched immediately (Guard 2 above), LLM controls completion.
    #   Safety limit: 6 turns max post-dispatch for micro-location collection.
    # URGENT: 4-5 turns max (collect essential + dispatch on is_complete)
    # NON-CRITICAL: 8 turns max (normal slot filling, soft close)
    # ------------------------------------------------------------------
    # For CRITICAL, after dispatch the LLM keeps asking micro-location questions.
    # Give it up to 6 turns total before forcing completion.
    critical_dispatched_max_turns = 6
    max_turns_map = {
        "URGENT": 3,        # Urgent: force dispatch by turn 3 if still incomplete
        "NON_URGENT": 8,    # Non-critical: normal dialog flow
    }
    non_critical_min_turns_before_close = 3
    non_critical_min_informative_slots = 3  # chief_complaint + 1-2 extra details
    
    user_turn_count = sum(1 for m in session.messages if m.get("role") == "user")
    # Include current turn's extracted slots when checking NON-CRITICAL readiness.
    effective_slots = dict(session.collected_slots)
    if extracted_slots:
        effective_slots.update(extracted_slots)
    informative_slot_count = len([
        k for k in effective_slots.keys()
        if k not in ("_asking_slot", "category", "triage_level")
    ])
    urgent_minimum_ready = _has_urgent_minimum_slots(effective_slots, category)
    urgent_dispatched_this_turn = False

    if triage_level == "URGENT" and is_complete and not urgent_minimum_ready:
        is_complete = False
        response_text = _urgent_missing_slot_question(lang, effective_slots, category)

    if triage_level == "URGENT" and session.dispatch_status == "PENDING":
        wants_dispatch_now = dispatch_action == "dispatch_now" or is_complete
        force_dispatch_at_turn = user_turn_count >= max_turns_map.get("URGENT", 3) and not urgent_minimum_ready

        if wants_dispatch_now and not urgent_minimum_ready and not force_dispatch_at_turn:
            is_complete = False
            response_text = _urgent_missing_slot_question(lang, effective_slots, category)
        elif wants_dispatch_now or force_dispatch_at_turn:
            if not can_redispatch(session, redispatch_ttl_seconds=48 * 3600):
                logger.warning("Dispatch lock active: URGENT case already dispatched. Preventing redispatch.")
                is_complete = False
                response_text = {
                    "tr": "Bu durum zaten kayda alındı. Daha fazla bilgi var mı?",
                    "en": "This case has already been reported. Is there any additional information?",
                }.get(lang, "This case has already been reported.")
            else:
                session.dispatch_status = "DISPATCHED"
                session.dispatch_target = category
                session.dispatch_timestamp = time.time()
                session.pending_update_after_dispatch = True
                session.post_dispatch_turn_count = 0
                urgent_dispatched_this_turn = True
                dispatched_this_turn = True
                is_complete = False
                dispatch_notice = _DISPATCH_MSG.get(lang, _DISPATCH_MSG["en"])
                followup_q = _urgent_micro_location_question(lang)
                response_text = f"{dispatch_notice} {followup_q}" if response_text else f"{dispatch_notice} {followup_q}"
                logger.info("URGENT dispatch: vaka_id=%s, target=%s, turn=%d",
                            session.session_id, category, user_turn_count)

    # URGENT pending state should never stall with advice-only text.
    # If no question is present yet, append a direct follow-up question.
    if triage_level == "URGENT" and session.dispatch_status == "PENDING" and not is_complete:
        if "?" not in response_text:
            next_q = _urgent_missing_slot_question(lang, effective_slots, category)
            response_text = f"{response_text} {next_q}".strip() if response_text else next_q

    # NON-CRITICAL should not close too early: collect 1-2 additional slots first.
    if triage_level == "NON_URGENT" and is_complete:
        if (
            user_turn_count < non_critical_min_turns_before_close
            or informative_slot_count < non_critical_min_informative_slots
        ):
            is_complete = False
            logger.info(
                "NON_URGENT closure delayed: turns=%d/%d, slots=%d/%d",
                user_turn_count,
                non_critical_min_turns_before_close,
                informative_slot_count,
                non_critical_min_informative_slots,
            )
            if not response_text or "?" not in response_text:
                response_text = {
                    "tr": "Durumunuzu güvenli şekilde kapatmadan önce iki kısa bilgi daha alayım: şikayet ne zaman başladı ve şu an kötüleşme var mı?",
                    "en": "Before safely closing this case, I need two short details: when did the complaint start, and is it getting worse right now?",
                }.get(lang, "I need two short details before closing this case.")
    
    # FAZ 6: CRITICAL — LLM controls is_complete; safety limit at 6 turns
    if triage_level == "CRITICAL" and user_turn_count >= critical_dispatched_max_turns and not is_complete:
        is_complete = True
        logger.info("CRITICAL: safety max turns (%d) reached – forcing card display.", critical_dispatched_max_turns)
        if not response_text:
            response_text = _COMPLETE_MSG.get(lang, _COMPLETE_MSG["en"])
    
    # FAZ 6: NON-CRITICAL soft close — offer graceful exit
    elif triage_level == "NON_URGENT" and user_turn_count >= max_turns_map.get("NON_URGENT", 8) - 1:
        # At turn 7 (max 8), offer soft close
        is_complete = True
        logger.info("NON_CRITICAL: approaching max turns – soft close.")
        if not response_text:
            soft_close = {
                "tr": "Teşekkürler, yeterli bilgiye sahibiz. Durumunuz kaydedildi. Daha fazla yardıma ihtiyacınız varsa 112'yi arayın.",
                "en": "Thank you. We have sufficient information. Your case recorded. Call 112 if needed.",
            }.get(lang, "Thank you. Case recorded. Call 112 if needed.")
            response_text = soft_close

    # Post-dispatch completion: after ANY dispatch (CRITICAL/URGENT/SILENT), we
    # allow the dispatcher LLM to keep asking useful follow-up questions
    # (micro-location, consciousness, secondary threats, etc.) for up to
    # POST_DISPATCH_MAX_TURNS user turns. The session only closes here when one
    # of these is true:
    #   (1) the LLM itself marks is_complete=True,
    #   (2) the AI has no question left in response_text (no "?"),
    #   (3) the post-dispatch turn counter hits the configured safety cap.
    # Defaulting to 3 follow-up turns prevents the old "close after exactly one
    # user reply" behaviour that cut the user off mid-answer.
    if (
        session.dispatch_status in ("DISPATCHED", "SILENT_DISPATCHED")
        and session.pending_update_after_dispatch
        and not dispatched_this_turn
    ):
        session.post_dispatch_turn_count += 1
        try:
            max_post_dispatch_turns = int(os.environ.get("POST_DISPATCH_MAX_TURNS", "3"))
        except ValueError:
            max_post_dispatch_turns = 3
        max_post_dispatch_turns = max(1, max_post_dispatch_turns)

        ai_still_asking = "?" in (response_text or "")
        llm_wants_close = bool(is_complete)
        safety_hit = session.post_dispatch_turn_count >= max_post_dispatch_turns

        if llm_wants_close or safety_hit or not ai_still_asking:
            session.pending_update_after_dispatch = False
            is_complete = True
            logger.info(
                "Post-dispatch close (turn=%d/%d, llm_complete=%s, ai_asking=%s, safety=%s, level=%s, target=%s)",
                session.post_dispatch_turn_count,
                max_post_dispatch_turns,
                llm_wants_close,
                ai_still_asking,
                safety_hit,
                triage_level,
                session.dispatch_target,
            )
            if not response_text:
                response_text = _COMPLETE_MSG.get(lang, _COMPLETE_MSG["en"])
        else:
            is_complete = False
            logger.info(
                "Post-dispatch continuing (turn=%d/%d, AI still has a question)",
                session.post_dispatch_turn_count,
                max_post_dispatch_turns,
            )

    # Merge new slots into session
    if extracted_slots:
        session.collected_slots.update(extracted_slots)
        logger.info("Slots merged: %s", list(extracted_slots.keys()))

    if not response_text:
        # 6. Dil bazlı fallback mesajı
        response_text = _LLM_FALLBACK.get(lang, _LLM_FALLBACK["en"])

    response_text = _normalize_response_language(response_text, lang)

    # Build triage result (classification from OpenAI; slots from Groq merge above)
    _conf = (session.initial_triage or {}).get("confidence")
    try:
        confidence_out = float(_conf) if _conf is not None else 1.00
    except (TypeError, ValueError):
        confidence_out = 1.00
    triage_result: Dict[str, Any] = {
        "triage_level": triage_level,
        "category": category,
        "confidence": confidence_out,
        "red_flags": red_flags,
        "slots": session.collected_slots,
        "llm_powered": True,
    }

    # Sentiment override
    triage_result = _merge_sentiment_into_triage(session, triage_result)
    session.triage_result = triage_result

    # Update image analysis with triage context
    if session.image_bytes:
        _run_image_analysis(
            session,
            text_category=category,
            text_triage_level=triage_result["triage_level"],
        )
        triage_result["image_analysis"] = session.image_analysis

    if is_turn_trace_enabled():
        trace_kv(
            "Triage (OpenAI + duygu birleşimi; görsel varsa bağlam güncellendi)",
            {
                "triage_level": (session.triage_result or {}).get("triage_level"),
                "category": (session.triage_result or {}).get("category"),
                "red_flags": (session.triage_result or {}).get("red_flags"),
                "witness_mode": session.witness_mode,
            },
        )

    # ------------------------------------------------------------------
    # Conversation complete → compose structured report
    # ------------------------------------------------------------------
    if is_complete:
        # Final-turn rule: the closing message must be informational
        # ("ekipler yönlendirildi / vaka kaydedildi") and never a question.
        # If the LLM left a question hanging or returned empty text, replace
        # it entirely with the deterministic closing message.
        _closing_msg = _final_closing_text(session, lang)
        if not response_text or "?" in response_text:
            response_text = _closing_msg
        report_local = _compose_session_report(
            triage_result=session.triage_result,
            slots=session.collected_slots,
            image_analysis=session.image_analysis,
            lang=lang,
        )
        session.is_complete = True
        session.pending_update_after_dispatch = False

        # Fix 4: TTS reads LLM guidance + the advice/instruction bullets from the report.
        # The full text (with report) is shown in the chat bubble, but audio = guidance + advice.
        from orchestrator.report_composer import compose_tts_instructions
        instructions_tts = compose_tts_instructions(session.triage_result, language=lang)
        tts_parts = [p for p in [response_text, instructions_tts] if p]
        tts_only = ". ".join(tts_parts) if tts_parts else _COMPLETE_MSG.get(lang, _COMPLETE_MSG["en"])
        final_text = (response_text + "\n\n" + report_local) if response_text else report_local

        if is_turn_trace_enabled():
            trace_orchestrator_outcome(
                triage_level=str((session.triage_result or {}).get("triage_level", "")),
                category=str((session.triage_result or {}).get("category", "")),
                dispatch_status=str(session.dispatch_status or ""),
                dispatch_target=session.dispatch_target,
                is_complete=True,
                user_turn_count=sum(1 for m in session.messages if m.get("role") == "user"),
            )

        return _reply(
            session,
            final_text,
            triage_result=session.triage_result,
            image_analysis=session.image_analysis,
            report=report_local,
            is_complete=True,
            user_transcript=asr_transcript,
            tts_text=tts_only,
        )

    # Conversation ongoing
    if is_turn_trace_enabled():
        trace_orchestrator_outcome(
            triage_level=str((session.triage_result or {}).get("triage_level", "")),
            category=str((session.triage_result or {}).get("category", "")),
            dispatch_status=str(session.dispatch_status or ""),
            dispatch_target=session.dispatch_target,
            is_complete=False,
            user_turn_count=sum(1 for m in session.messages if m.get("role") == "user"),
        )

    return _reply(
        session,
        response_text,
        triage_result=triage_result,
        image_analysis=session.image_analysis,
        user_transcript=asr_transcript,
    )


# ---------------------------------------------------------------------------
# Rule-based fallback (original dialog_manager logic)
# ---------------------------------------------------------------------------

def _handle_with_rules(
    session: Session,
    lang: str,
    asr_transcript: Optional[str],
) -> Dict[str, Any]:
    """Original slot-filling + rule-based triage flow (used when no LLM key)."""
    from orchestrator.dialog_manager import decide_next_action

    _extract_and_merge_slots(session)

    action = decide_next_action(session)
    logger.info("Dialog action (rules): %s", action)

    if is_turn_trace_enabled():
        trace_step("Kural tabanlı dialog_manager", f"decide_next_action → {action}")

    if action["action"] == "ask_question":
        question_en = action["question_en"]
        session.asked_questions.add(action["question_key"])
        question_local = (
            translate_from_english(question_en, target_lang=lang)
            if lang != "en"
            else question_en
        )
        return _reply(
            session,
            question_local,
            image_analysis=session.image_analysis,
            user_transcript=asr_transcript,
        )

    if action["action"] in ("run_triage", "complete"):
        triage = _run_triage(session)
        triage = _merge_sentiment_into_triage(session, triage)
        session.triage_result = triage

        if is_turn_trace_enabled():
            trace_kv(
                "Kural tabanlı triage (ML + mvp_rules / yedek)",
                {
                    "triage_level": triage.get("triage_level"),
                    "category": triage.get("category"),
                    "confidence": triage.get("confidence"),
                    "red_flags": triage.get("red_flags"),
                },
            )

        if session.image_bytes and session.image_analysis:
            _run_image_analysis(
                session,
                text_category=triage.get("category"),
                text_triage_level=triage.get("triage_level"),
            )
            triage["image_analysis"] = session.image_analysis

        if action["action"] == "run_triage":
            action2 = decide_next_action(session)
            if action2["action"] == "ask_question":
                question_en = action2["question_en"]
                session.asked_questions.add(action2["question_key"])
                question_local = (
                    translate_from_english(question_en, target_lang=lang)
                    if lang != "en"
                    else question_en
                )
                return _reply(
                    session,
                    question_local,
                    triage_result=triage,
                    image_analysis=session.image_analysis,
                    user_transcript=asr_transcript,
                )

    report_local = _compose_session_report(
        triage_result=session.triage_result,
        slots=session.collected_slots,
        image_analysis=session.image_analysis,
        lang=lang,
    )
    session.is_complete = True

    return _reply(
        session,
        report_local,
        triage_result=session.triage_result,
        image_analysis=session.image_analysis,
        report=report_local,
        is_complete=True,
        user_transcript=asr_transcript,
    )


# ---------------------------------------------------------------------------
# Reply helper
# ---------------------------------------------------------------------------

def _reply(
    session: Session,
    text: str,
    triage_result: Optional[Dict[str, Any]] = None,
    image_analysis: Optional[Dict[str, Any]] = None,
    report: Optional[str] = None,
    is_complete: bool = False,
    user_transcript: Optional[str] = None,
    tts_text: Optional[str] = None,
) -> Dict[str, Any]:
    session.messages.append({"role": "assistant", "text": text})

    # Fix 4: TTS only reads the advice/instructions portion, not the structured report.
    # When tts_text is provided (e.g. only the LLM's guidance), use that for audio.
    audio_source = tts_text if tts_text else text

    # Deferred TTS: when TTS_INLINE is disabled, skip synthesis in the hot path
    # and let the client request audio from /tts once it has the text.
    if _tts_inline_enabled():
        t0 = time.monotonic()
        audio_bytes = synthesize(audio_source, lang=session.language or "en")
        audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
        audio_url = _audio_to_data_url(audio_bytes)
        logger.info(
            "  [TIMING] TTS: %.2fs %s",
            time.monotonic() - t0,
            get_tts_runtime_info_str(),
        )
        tts_deferred = False
    else:
        audio_b64 = None
        audio_url = None
        tts_deferred = True
        logger.debug("TTS deferred (TTS_INLINE=false); client will fetch via /tts.")

    # FAZ 8-9: Build response with dispatch + resume info
    
    # Summary card for post-dispatch state
    summary_card = None
    if session.dispatch_status in ("DISPATCHED", "SILENT_DISPATCHED"):
        summary_card = {
            "category": session.dispatch_target or "other",
            "triage_level": (session.triage_result or {}).get("triage_level", "URGENT"),
            "dispatch_time": (session.dispatch_timestamp and 
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(session.dispatch_timestamp))),
            "dispatch_target": session.dispatch_target,
            "estimated_arrival_min": 5,  # TODO: Actual ETA calculation
            "collected_slots": session.collected_slots,
        }
    
    # Resume prompt if resuming after timeout
    resume_prompt = None
    if session.resumed_after_timeout:
        resume_prompt = {
            "tr": "Hoş geldiniz geri! Yardım çalışmaları devam ediyor. Yeni bilgi var mı?",
            "en": "Welcome back! Emergency response is in progress. Any updates?",
        }.get(session.language or "en", "Welcome back! Any updates?")
    
    # Followup status
    followup_status = None
    if session.dispatch_status in ("DISPATCHED", "SILENT_DISPATCHED"):
        if session.pending_update_after_dispatch and not session.is_complete:
            followup_status = "waiting_for_micro_location"
        else:
            followup_status = "dispatch_sent" if session.dispatch_timestamp else "dispatch_pending"
    elif session.dispatch_status == "PENDING":
        followup_status = "waiting_for_info" if not session.is_complete else "no_dispatch_needed"

    nearby_places = _resolve_nearby_places(session, triage_result or session.triage_result)

    # Persist the full session snapshot (messages + triage + dispatch + report)
    # to DynamoDB exactly once, at finalize-time. No-ops when CASES_DYNAMODB_TABLE
    # is unset; errors are logged but never break the live reply flow.
    if session.is_complete and not session.case_persisted:
        persist_case_if_configured(session, final_report=report)
        session.case_persisted = True

    return {
        "session_id": session.session_id,
        "assistant_text": text,
        "assistant_audio_url": audio_url,
        "assistant_audio_b64": audio_b64,
        "assistant_tts_text": audio_source,
        "tts_deferred": tts_deferred,
        "user_transcript": user_transcript,
        "triage_result": triage_result,
        "image_analysis": image_analysis,
        "report": report,
        "is_complete": is_complete,
        # FAZ 8-9 fields
        "chatbot_mode": "normal",  # TODO: "fallback" when TextAnalyze is ready
        "dispatch_status": session.dispatch_status,
        "summary_card": summary_card,
        "resume_prompt": resume_prompt,
        "followup_status": followup_status,
        "nearby_places": nearby_places,
    }


def _resolve_nearby_places(
    session: Session,
    triage_result: Optional[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    if os.environ.get("NEARBY_PLACES_ENABLED", "false").strip().lower() not in ("1", "true", "yes"):
        return None

    latitude = session.collected_slots.get("latitude")
    longitude = session.collected_slots.get("longitude")
    if latitude is None or longitude is None:
        return None

    preferred_type = _preferred_nearby_type((triage_result or {}).get("category"))
    try:
        from services.nearby_places_service import get_nearby_places

        return get_nearby_places(
            float(latitude),
            float(longitude),
            preferred_type=preferred_type,
            limit_per_type=5,
        )
    except Exception as exc:
        logger.warning("Nearby places could not be resolved: %s", exc)
        return []


def _preferred_nearby_type(category: Optional[str]) -> Optional[str]:
    if category == "crime":
        return "police"
    if category == "medical":
        return "hospital"
    return None


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def _run_sentiment_analysis(session: Session, text_en: str, audio_bytes: bytes) -> None:
    try:
        from services.sentiment_service import get_sentiment_service

        svc = get_sentiment_service()
        if not svc.is_loaded:
            return
        meta = {
            "deaths": session.meta.get("deaths", 0),
            "potential_death": session.meta.get("potential_death", 0),
            "false_alarm": session.meta.get("false_alarm", 0),
            "civilian_initiated": 1,
        }
        result = svc.predict(text_en, audio_bytes=audio_bytes, meta=meta)
        if result:
            session.sentiment_result = result
            logger.info(
                "Sentiment result: %s (panic=%s)",
                result.get("triage_level"),
                result.get("panic_level"),
            )
    except Exception as exc:
        logger.debug("Sentiment analysis skipped: %s", exc)


def _merge_sentiment_into_triage(
    session: Session, triage: Dict[str, Any]
) -> Dict[str, Any]:
    sent = session.sentiment_result
    if not sent or not sent.get("triage_level"):
        triage = _apply_temporal_consistency(session, triage)
        return triage
    severity_order = {"CRITICAL": 3, "URGENT": 2, "NON_URGENT": 1}
    text_level = triage.get("triage_level") or "URGENT"
    sent_level = sent.get("triage_level")
    text_sev = severity_order.get(text_level, 2)
    sent_sev = severity_order.get(sent_level, 2)
    if sent_sev > text_sev and sent.get("confidence", 0) >= 0.5:
        triage["triage_level"] = sent_level
        triage["confidence"] = sent.get("confidence")
        triage["sentiment_override"] = True
        logger.info(
            "Triage upgraded from %s to %s by sentiment", text_level, sent_level
        )
    triage["sentiment_result"] = sent
    triage = _apply_temporal_consistency(session, triage)
    return triage


# ---------------------------------------------------------------------------
# Temporal consistency — son N turdaki triage tahminleri uzerinden smoothing.
# Amac: tek-tur CRITICAL tahminini kucuk bir gecikme ile onaylamak; red_flag ve
# yuksek confidence varsa anlik onaya izin verir.
# ---------------------------------------------------------------------------

_TEMPORAL_HISTORY_MAX = 8  # Maksimum tutulacak snapshot sayisi


def _apply_temporal_consistency(
    session: Session, triage: Dict[str, Any]
) -> Dict[str, Any]:
    """Session.triage_history'ye yeni tahmini ekler, gecmisle uyum kurallari uygular."""
    if not triage:
        return triage

    new_level = str(triage.get("triage_level") or "").upper()
    new_conf = float(triage.get("confidence") or 0.0)
    new_rf = 1 if (triage.get("red_flags") or []) else 0

    session.triage_history.append({
        "triage_level": new_level,
        "confidence": new_conf,
        "red_flag_present": new_rf,
        "ts": time.time(),
    })
    if len(session.triage_history) > _TEMPORAL_HISTORY_MAX:
        session.triage_history = session.triage_history[-_TEMPORAL_HISTORY_MAX:]

    # sentiment_override veya red_flag varsa temporal downgrade YAPMA.
    if triage.get("sentiment_override") or new_rf == 1:
        return triage

    # CRITICAL lock aktifse aşağı yönlü yumuşatma yapma. Vaka zaten bir kez
    # CRITICAL olarak kilitlendiyse (ve muhtemelen sevk edildiyse) kullanıcının
    # sonraki nötr onay mesajları seviyeyi düşürmemelidir.
    if getattr(session, "critical_locked", False):
        if new_level != "CRITICAL":
            logger.info(
                "Temporal smooth skipped: CRITICAL lock active → keeping CRITICAL "
                "(model returned %s)", new_level,
            )
            triage["triage_level"] = "CRITICAL"
            triage["critical_locked"] = True
        return triage

    recent = session.triage_history[-3:]
    if len(recent) < 2:
        return triage

    # Yeni CRITICAL + gecmiste hic CRITICAL yok + confidence dusuk
    # => gecici olarak URGENT'a dusur (bir sonraki turda tekrar CRITICAL
    # gelirse sinyal tutarli demektir, o zaman commit).
    if new_level == "CRITICAL":
        prev_critical = sum(1 for r in recent[:-1] if r.get("triage_level") == "CRITICAL")
        if prev_critical == 0 and new_conf < 0.80:
            logger.info(
                "Temporal smooth: first CRITICAL turn (conf=%.2f) -> URGENT until confirmed",
                new_conf,
            )
            triage["triage_level"] = "URGENT"
            triage["temporal_smoothed"] = True
            return triage

    return triage


# ---------------------------------------------------------------------------
# Rule-based helpers (used in fallback mode)
# ---------------------------------------------------------------------------

def _extract_and_merge_slots(session: Session) -> None:
    try:
        from mvp_rules import extract_slots, load_rules
        from orchestrator.dialog_manager import merge_slots

        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        new_slots = extract_slots(session.text_en_accumulated, rules)
        merge_slots(session, new_slots)
    except Exception as exc:
        logger.warning("Slot extraction failed: %s", exc)


def _run_triage(session: Session) -> Dict[str, Any]:
    text_en = session.text_en_accumulated
    meta = session.meta
    slots = session.collected_slots
    result: Dict[str, Any] = {}

    try:
        from api.model_loader import apply_redflag_override, get_model_service

        svc = get_model_service()
        if svc.is_loaded:
            label, conf = svc.predict(text_en, meta=meta)
            label, red_flags = apply_redflag_override(text_en, label, meta)
            result = {
                "category": _infer_category_rule(text_en),
                "triage_level": label,
                "confidence": conf,
                "red_flags": list(set(red_flags) | set(slots.get("red_flags", []))),
                "slots": slots,
            }
    except Exception as exc:
        logger.warning("ML model prediction failed: %s – falling back to MVP", exc)

    if not result:
        try:
            from mvp_rules import predict_mvp

            result = predict_mvp(
                text=text_en,
                deaths=meta.get("deaths", 0),
                potential_death=meta.get("potential_death", 0),
                false_alarm=meta.get("false_alarm", 0),
            )
            result["confidence"] = None
            result["slots"] = slots
        except Exception as exc:
            logger.error("MVP rules also failed: %s", exc)
            result = {
                "category": "other",
                "triage_level": "URGENT",
                "confidence": None,
                "red_flags": [],
                "slots": slots,
            }

    result["red_flags"] = list(
        set(result.get("red_flags", []) or []) | set(slots.get("red_flags", []) or [])
    )
    result["needs_more_info"] = False
    result["recommended_questions"] = []
    return result


def _infer_category_rule(text_en: str) -> str:
    try:
        from mvp_rules import infer_category, load_rules

        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        return infer_category(text_en, rules)
    except Exception:
        return "other"


def _run_image_analysis(
    session: Session,
    text_category: Optional[str] = None,
    text_triage_level: Optional[str] = None,
) -> None:
    if not session.image_bytes:
        return
    try:
        from services.image_service import analyze_image

        if text_category is None and session.triage_result:
            text_category = session.triage_result.get("category")
        if text_triage_level is None and session.triage_result:
            text_triage_level = session.triage_result.get("triage_level")

        # Short-circuit: if the (image, category, level) triple is unchanged
        # since the last successful analysis, reuse the cached result instead
        # of running the vision model again.
        try:
            import hashlib

            image_id = hashlib.sha1(session.image_bytes).hexdigest()[:16]
        except Exception:
            image_id = str(len(session.image_bytes))
        cache_key = f"{image_id}|{text_category or ''}|{text_triage_level or ''}"
        if (
            session.image_analysis
            and session.image_analysis.get("available", True)
            and session.last_image_analysis_key == cache_key
        ):
            logger.debug("Image analysis cache hit (key=%s), skipping re-run.", cache_key)
            return

        t0 = time.monotonic()
        result = analyze_image(
            image_bytes=session.image_bytes,
            text_category=text_category,
            text_triage_level=text_triage_level,
        )
        session.image_analysis = result
        if result.get("available", True):
            session.last_image_analysis_key = cache_key
        logger.info(
            "  [TIMING] Image analysis: %.2fs — %s",
            time.monotonic() - t0,
            result.get("summary", ""),
        )
    except Exception as exc:
        logger.error("Image analysis failed: %s", exc)
        session.image_analysis = {
            "classification": None,
            "consistency": None,
            "summary": f"Image analysis failed: {exc}",
            "available": False,
        }
        session.last_image_analysis_key = None
