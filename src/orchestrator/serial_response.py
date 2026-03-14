"""
Serial response generator: produces a single, short reply from decision-tree outputs.

Used after LLM returns decision_steps; backend fills response_text deterministically
from triage_level, red_flags, is_complete, and next_question_key.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Language-keyed templates (same as orchestrator)
_DISPATCH_MSG: Dict[str, str] = {
    "tr": "Yardım ekipleri yönlendiriliyor, lütfen hatta kalın.",
    "en": "Emergency services are being dispatched. Please stay on the line.",
    "de": "Notfalldienste werden entsandt. Bitte bleiben Sie in der Leitung.",
    "fr": "Les secours sont en route. Veuillez rester en ligne.",
    "es": "Los servicios de emergencia están en camino. Por favor permanezca en línea.",
    "ar": "يتم إرسال الخدمات الطارئة. الرجاء البقاء على الخط.",
    "ru": "Экстренные службы вызваны. Пожалуйста, оставайтесь на линии.",
}

_COMPLETE_MSG: Dict[str, str] = {
    "tr": "Teşekkürler, yeterli bilgiye sahibim. Yardım yönlendiriliyor.",
    "en": "Thank you. I have enough information. Help is on the way.",
    "de": "Danke. Ich habe genug Informationen. Hilfe ist unterwegs.",
    "fr": "Merci. J'ai suffisamment d'informations. Les secours arrivent.",
    "es": "Gracias. Tengo suficiente información. La ayuda está en camino.",
    "ar": "شكراً. لدي معلومات كافية. المساعدة في الطريق.",
    "ru": "Спасибо. У меня достаточно информации. Помощь уже едет.",
}

_LLM_FALLBACK: Dict[str, str] = {
    "tr": "Anlayamadım, lütfen tekrar anlatır mısınız?",
    "en": "I didn't catch that, could you please describe the situation again?",
    "de": "Ich habe das nicht verstanden, bitte beschreiben Sie die Situation erneut.",
    "fr": "Je n'ai pas compris, pourriez-vous décrire la situation à nouveau ?",
    "es": "No entendí, ¿podría describir la situación nuevamente?",
    "ar": "لم أفهم، هل يمكنك وصف الوضع مرة أخرى؟",
    "ru": "Я не понял, не могли бы вы описать ситуацию снова?",
}

# One question per key (English); orchestrator or caller can translate
_NEXT_QUESTION_EN: Dict[str, str] = {
    "chief_complaint": "Can you describe what happened or what is wrong?",
    "caller_name": "What is your name?",
    "age": "How old is the person who needs help?",
    "severity_1_10": "On a scale of 1-10, how severe is the situation?",
    "sex": "Is the person male or female?",
    "duration_minutes": "How long has this been going on (in minutes)?",
    "location_hint": "Where are you right now? (home, street, car, etc.)",
    "breathing": "Is the person breathing right now?",
    "consciousness": "Are they conscious? Can they open their eyes or respond?",
    "duration_medical": "How long have they been in this state?",
    "bleeding": "Is there any bleeding? If yes, where and how much?",
    "trapped": "Is anyone trapped inside the building?",
    "fire_size": "How large is the fire? One room or spreading?",
    "smoke_inhalation": "Is anyone having trouble breathing from smoke?",
    "assailant_present": "Is the assailant still there?",
    "weapon": "Do you see any weapon?",
    "injuries": "Is anyone injured?",
}


def build_serial_response(
    slots: Dict[str, Any],
    category: str,
    triage_level: str,
    red_flags: list,
    is_complete: bool,
    next_question_key: Optional[str],
    lang: str,
) -> str:
    """
    Produce a single, short reply from decision-tree outputs.
    Priority: CRITICAL+red_flags -> dispatch; is_complete -> complete; else one question or fallback.
    """
    lang = (lang or "en").lower()[:2]
    if lang not in _DISPATCH_MSG:
        lang = "en"

    # 1. CRITICAL + red_flags -> dispatch message
    if triage_level == "CRITICAL" and red_flags:
        return _DISPATCH_MSG.get(lang, _DISPATCH_MSG["en"])

    # 2. Complete (and not needing dispatch) -> thank you / help on the way
    if is_complete:
        return _COMPLETE_MSG.get(lang, _COMPLETE_MSG["en"])

    # 3. Next question (one only)
    if next_question_key and next_question_key.strip():
        question_en = _NEXT_QUESTION_EN.get(
            next_question_key.strip().lower(),
            _NEXT_QUESTION_EN.get("chief_complaint", "Can you describe what happened?"),
        )
        if lang != "en":
            try:
                from services.translation_service import translate_from_english
                return translate_from_english(question_en, target_lang=lang)
            except Exception:
                return question_en
        return question_en

    # 4. Fallback: ask for chief_complaint if missing, else generic fallback
    if not slots.get("chief_complaint"):
        question_en = _NEXT_QUESTION_EN["chief_complaint"]
        if lang != "en":
            try:
                from services.translation_service import translate_from_english
                return translate_from_english(question_en, target_lang=lang)
            except Exception:
                return question_en
        return question_en

    return _LLM_FALLBACK.get(lang, _LLM_FALLBACK["en"])
