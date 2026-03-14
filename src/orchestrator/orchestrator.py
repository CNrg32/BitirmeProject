"""
Orchestrator – main entry point for session management and message handling.

Turn flow (LLM-powered):
  1.  ASR (if audio provided)
  2.  Language detection / lock
  3.  Slot extraction + response generation via Gemini (full conversation history)
  4.  Sentiment analysis on audio (enhances triage level if panic detected)
  5.  Image analysis (if image was attached)
  6.  TTS the assistant reply
  7.  Return structured response

If GEMINI_API_KEY is not set the system falls back to the original rule-based
dialog_manager + mvp_rules flow automatically.
"""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.translation_service import translate_to_english, translate_from_english, detect_language
from services.tts_service import synthesize
from orchestrator.session import Session, get_session_store
from orchestrator.report_composer import compose_report

logger = logging.getLogger(__name__)


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

def start_session(language: Optional[str] = None) -> Dict[str, Any]:
    store = get_session_store()
    session = store.create(language=language)
    if language:
        session.language_locked = True
    lang = session.language or "en"

    greeting = _GREETINGS.get(lang, _GREETINGS["en"])
    session.messages.append({"role": "assistant", "text": greeting})

    audio_bytes = synthesize(greeting, lang=lang)
    audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
    audio_url = _audio_to_data_url(audio_bytes)

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
        return {"error": "Session not found or expired."}

    # Guard: zaten tamamlanmış session'a yeni mesaj gelirse raporla cevap ver
    if session.is_complete:
        done_msg = {
            "tr": "Bu oturum tamamlandı. Yeni bir acil durum için lütfen yeniden başlatın.",
            "en": "This session is already complete. Please start a new session for a new emergency.",
        }.get(session.language or "en",
              "This session is already complete. Please start a new session.")
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
            logger.info("  [TIMING] ASR: %.2fs", time.monotonic() - t0)
            user_text = transcript
            asr_transcript = transcript
            if detected_lang and not session.language_locked:
                session.language = detected_lang
                session.language_locked = True
                lang = detected_lang
                logger.info("Language locked from first audio: %s", detected_lang)
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
            img_summary = session.image_analysis.get("summary", "Image received.")
            ack_en = f"Image received and analyzed. {img_summary} Please also describe the emergency verbally."
            ack_local = translate_from_english(ack_en, target_lang=lang) if lang != "en" else ack_en
            return _reply(session, ack_local, image_analysis=session.image_analysis, user_transcript=asr_transcript)
        return _reply(
            session,
            "I didn't receive any input. Could you please describe your emergency?",
            user_transcript=asr_transcript,
        )

    # ------------------------------------------------------------------
    # 4. Language detection from text (if not already locked)
    # ------------------------------------------------------------------
    if not session.language_locked:
        detected_text_lang = detect_language(user_text)
        if detected_text_lang:
            session.language = detected_text_lang
            session.language_locked = True
            lang = detected_text_lang
            logger.info("Language locked from first text input: %s", detected_text_lang)

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

    # Add user message to history
    session.messages.append({"role": "user", "text": user_text})

    # ------------------------------------------------------------------
    # 6. Sentiment analysis (needs audio for best results)
    # ------------------------------------------------------------------
    if audio_bytes:
        _run_sentiment_analysis(session, text_en, audio_bytes)

    # ------------------------------------------------------------------
    # 7. LLM turn (or fallback to rule-based)
    # ------------------------------------------------------------------
    from services.llm_service import get_llm_service
    llm = get_llm_service()

    if llm.is_available:
        return _handle_with_llm(session, lang, asr_transcript)
    else:
        return _handle_with_rules(session, lang, asr_transcript)


# ---------------------------------------------------------------------------
# LLM-powered turn
# ---------------------------------------------------------------------------

def _handle_with_llm(
    session: Session,
    lang: str,
    asr_transcript: Optional[str],
) -> Dict[str, Any]:
    from services.llm_service import get_llm_service

    llm = get_llm_service()

    t0 = time.monotonic()
    llm_result = llm.chat(history=session.messages, language=lang)
    logger.info("  [TIMING] LLM: %.2fs", time.monotonic() - t0)

    response_text: str = llm_result.get("response_text", "")
    extracted_slots: Dict[str, Any] = llm_result.get("extracted_slots", {})
    triage_level: str = llm_result.get("triage_level", "URGENT")
    category: str = llm_result.get("category", "other")
    is_complete: bool = llm_result.get("is_complete", False)
    red_flags: List[str] = llm_result.get("red_flags", [])

    # ------------------------------------------------------------------
    # Guard 1 — Category locking
    # Once a non-"other" category is established, never let the LLM flip it.
    # ------------------------------------------------------------------
    locked_category = (session.triage_result or {}).get("category", "other")
    if locked_category and locked_category != "other":
        category = locked_category
        logger.debug("Category locked to '%s' (LLM suggested '%s')",
                     locked_category, llm_result.get("category"))

    # ------------------------------------------------------------------
    # Guard 2 — Red-flag instant completion
    # Life-threatening red flags → dispatch immediately, stop asking.
    # ------------------------------------------------------------------
    if red_flags and triage_level == "CRITICAL" and not is_complete:
        is_complete = True
        logger.info("Red flags detected (%s) with CRITICAL triage – forcing completion.",
                    red_flags)
        if not response_text:
            response_text = _DISPATCH_MSG.get(lang, _DISPATCH_MSG["en"])

    # ------------------------------------------------------------------
    # Guard 3 — Maximum turn limit
    # After MAX_TURNS user messages, force completion to prevent infinite loops.
    # ------------------------------------------------------------------
    MAX_TURNS = 8
    user_turn_count = sum(1 for m in session.messages if m.get("role") == "user")
    if user_turn_count >= MAX_TURNS and not is_complete:
        is_complete = True
        logger.info("Max turns (%d) reached – forcing conversation complete.", MAX_TURNS)
        if not response_text:
            response_text = _COMPLETE_MSG.get(lang, _COMPLETE_MSG["en"])

    # Merge new slots into session
    if extracted_slots:
        session.collected_slots.update(extracted_slots)
        logger.info("Slots merged: %s", list(extracted_slots.keys()))

    if not response_text:
        # 6. Dil bazlı fallback mesajı
        response_text = _LLM_FALLBACK.get(lang, _LLM_FALLBACK["en"])

    # Build triage result
    triage_result: Dict[str, Any] = {
        "triage_level": triage_level,
        "category": category,
        "confidence": 0.85,
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

    # ------------------------------------------------------------------
    # Conversation complete → compose structured report
    # ------------------------------------------------------------------
    if is_complete:
        report_en = compose_report(
            triage_result=session.triage_result,
            slots=session.collected_slots,
            image_analysis=session.image_analysis,
        )
        report_local = (
            translate_from_english(report_en, target_lang=lang)
            if lang != "en"
            else report_en
        )
        session.is_complete = True

        final_text = (response_text + "\n\n" + report_local) if response_text else report_local

        return _reply(
            session,
            final_text,
            triage_result=session.triage_result,
            image_analysis=session.image_analysis,
            report=report_local,
            is_complete=True,
            user_transcript=asr_transcript,
        )

    # Conversation ongoing
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

    report_en = compose_report(
        triage_result=session.triage_result,
        slots=session.collected_slots,
        image_analysis=session.image_analysis,
    )
    report_local = (
        translate_from_english(report_en, target_lang=lang)
        if lang != "en"
        else report_en
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
) -> Dict[str, Any]:
    session.messages.append({"role": "assistant", "text": text})

    t0 = time.monotonic()
    audio_bytes = synthesize(text, lang=session.language or "en")
    audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
    audio_url = _audio_to_data_url(audio_bytes)
    logger.info("  [TIMING] TTS: %.2fs", time.monotonic() - t0)

    return {
        "session_id": session.session_id,
        "assistant_text": text,
        "assistant_audio_url": audio_url,
        "assistant_audio_b64": audio_b64,
        "user_transcript": user_transcript,
        "triage_result": triage_result,
        "image_analysis": image_analysis,
        "report": report,
        "is_complete": is_complete,
    }


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

        result = analyze_image(
            image_bytes=session.image_bytes,
            text_category=text_category,
            text_triage_level=text_triage_level,
        )
        session.image_analysis = result
        logger.info("Image analysis completed: %s", result.get("summary", ""))
    except Exception as exc:
        logger.error("Image analysis failed: %s", exc)
        session.image_analysis = {
            "classification": None,
            "consistency": None,
            "summary": f"Image analysis failed: {exc}",
            "available": False,
        }
