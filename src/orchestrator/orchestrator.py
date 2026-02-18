from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import sys

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.translation_service import translate_to_english, translate_from_english, detect_language
from services.tts_service import synthesize
from orchestrator.session import Session, get_session_store
from orchestrator.dialog_manager import decide_next_action, merge_slots
from orchestrator.report_composer import compose_report

logger = logging.getLogger(__name__)


_GREETINGS = {
    "en": "Emergency assistant here. Please describe your emergency.",
    "tr": "Acil durum asistanı burada. Lütfen acil durumunuzu anlatın.",
    "de": "Notfall-Assistent hier. Bitte beschreiben Sie Ihren Notfall.",
    "fr": "Assistant d'urgence ici. Veuillez décrire votre urgence.",
    "es": "Asistente de emergencia aquí. Por favor describa su emergencia.",
    "ar": "مساعد الطوارئ هنا. يرجى وصف حالتك الطارئة.",
    "ru": "Ассистент экстренной помощи. Пожалуйста, опишите вашу ситуацию.",
}


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

    return {
        "session_id": session.session_id,
        "greeting": greeting,
        "greeting_audio_b64": audio_b64,
    }


def handle_message(
    session_id: str,
    user_text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    image_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        return {"error": "Session not found or expired."}

    lang = session.language or "en"
    asr_transcript: Optional[str] = None
    t_msg_start = time.monotonic()

    if image_bytes:
        session.image_bytes = image_bytes
        _run_image_analysis(session)

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
            return _reply(session, "Sorry, I could not understand the audio. Please try again or type your message.", user_transcript=asr_transcript)

    if not user_text or not user_text.strip():
        if image_bytes and session.image_analysis:
            img_summary = session.image_analysis.get("summary", "Image received.")
            ack_en = f"Image received and analyzed. {img_summary} Please also describe the emergency verbally."
            ack_local = translate_from_english(ack_en, target_lang=lang) if lang != "en" else ack_en
            return _reply(session, ack_local, image_analysis=session.image_analysis, user_transcript=asr_transcript)
        return _reply(session, "I didn't receive any input. Could you please describe your emergency?", user_transcript=asr_transcript)

    if not audio_bytes and not session.language_locked:
        detected_text_lang = detect_language(user_text)
        if detected_text_lang:
            session.language = detected_text_lang
            session.language_locked = True
            lang = detected_text_lang
            logger.info("Language locked from first text input: %s", detected_text_lang)

    session.messages.append({"role": "user", "text": user_text})

    t0 = time.monotonic()
    text_en = translate_to_english(user_text, source_lang=lang) if lang != "en" else user_text
    if lang != "en":
        logger.info("  [TIMING] Translate→EN: %.2fs", time.monotonic() - t0)
    session.text_en_accumulated += " " + text_en
    session.text_en_accumulated = session.text_en_accumulated.strip()

    _extract_and_merge_slots(session, text_en)

    if audio_bytes:
        _run_sentiment_analysis(session, text_en, audio_bytes)

    action = decide_next_action(session)
    logger.info("Dialog action: %s", action)

    if action["action"] == "ask_question":
        question_en = action["question_en"]
        session.asked_questions.add(action["question_key"])
        question_local = translate_from_english(question_en, target_lang=lang) if lang != "en" else question_en
        return _reply(session, question_local, image_analysis=session.image_analysis, user_transcript=asr_transcript)

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
                return _reply(session, question_local, triage_result=triage,
                              image_analysis=session.image_analysis, user_transcript=asr_transcript)

    report_en = compose_report(
        triage_result=session.triage_result,
        slots=session.collected_slots,
        image_analysis=session.image_analysis,
    )

    report_local = translate_from_english(report_en, target_lang=lang) if lang != "en" else report_en

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
    logger.info("  [TIMING] TTS: %.2fs", time.monotonic() - t0)

    return {
        "session_id": session.session_id,
        "assistant_text": text,
        "assistant_audio_b64": audio_b64,
        "user_transcript": user_transcript,
        "triage_result": triage_result,
        "image_analysis": image_analysis,
        "report": report,
        "is_complete": is_complete,
    }


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
            logger.info("Sentiment result: %s (panic=%s)", result.get("triage_level"), result.get("panic_level"))
    except Exception as exc:
        logger.debug("Sentiment analysis skipped: %s", exc)


def _merge_sentiment_into_triage(session: Session, triage: Dict[str, Any]) -> Dict[str, Any]:
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
        logger.info("Triage upgraded from %s to %s by sentiment", text_level, sent_level)
    triage["sentiment_result"] = sent
    return triage


def _extract_and_merge_slots(session: Session, text_en: str) -> None:
    try:
        from mvp_rules import extract_slots, load_rules

        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        new_slots = extract_slots(text_en, rules)
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

    result["red_flags"] = list(set(result.get("red_flags", []) or []) | set(slots.get("red_flags", []) or []))

    result["needs_more_info"] = False
    result["recommended_questions"] = []

    return result


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


def _infer_category_rule(text_en: str) -> str:
    try:
        from mvp_rules import infer_category, load_rules

        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        return infer_category(text_en, rules)
    except Exception:
        return "other"
