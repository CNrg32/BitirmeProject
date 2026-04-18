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
import re
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
        return {"error": "Session not found or expired."}
    
    # ------------------------------------------------------------------
    # FAZ 7: Timeout & Silent Dispatch (3-minute inactivity rule)
    # If session is waiting for response and 3+ minutes passed with no activity:
    # CRITICAL/URGENT → auto-dispatch (silent dispatch)
    # User returning after timeout → resume mode
    # ------------------------------------------------------------------
    current_time = time.time()
    
    # Set timeout deadline on first message if not set (3 minutes = 180 seconds)
    if not session.timeout_deadline:
        session.timeout_deadline = session.last_user_activity_at + (3 * 60)
        logger.debug("Timeout deadline set: %.0f (current: %.0f)", 
                     session.timeout_deadline, current_time)
    
    # Check if timeout has been triggered
    if session.dispatch_status in ("PENDING", "FALLBACK_PENDING") and not session.resumed_after_timeout:
        if current_time > session.timeout_deadline:
            # Timeout triggered
            if session.triage_result:
                triage_level = session.triage_result.get("triage_level", "NON_URGENT")
                if triage_level in ("CRITICAL", "URGENT"):
                    # Silent dispatch for critical/urgent cases
                    session.dispatch_status = "SILENT_DISPATCHED"
                    session.dispatch_target = session.triage_result.get("category", "other")
                    session.dispatch_timestamp = current_time
                    logger.warning("Silent dispatch triggered: timeout after 3 min. vaka_id=%s, category=%s",
                                   session.session_id, session.dispatch_target)
                    
                    # Prepare timeout message
                    timeout_msg = {
                        "tr": "Yanıt alamadığım için yardım ekiplerini gönderdim. Lütfen güvende kalın.",
                        "en": "No response detected. Emergency services have been dispatched. Please stay safe.",
                    }.get(session.language or "en", "Emergency services dispatched.")
                    
                    return _reply(session, timeout_msg, triage_result=session.triage_result, 
                                is_complete=True, report=None)
    
    # User returning after timeout → resume mode
    if current_time > session.timeout_deadline and not session.resumed_after_timeout:
        session.resumed_after_timeout = True
        logger.info("Session resumed after timeout: vaka_id=%s", session.session_id)

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
            return _handle_image_only(session, lang, asr_transcript)
        return _reply(
            session,
            "I didn't receive any input. Could you please describe your emergency?",
            user_transcript=asr_transcript,
        )

    # ------------------------------------------------------------------
    # 4. Language detection from text (if not already locked)
    # ------------------------------------------------------------------
    hard_noise = _is_gibberish(user_text)
    llm_noise = None if hard_noise else _is_gibberish_with_llm(user_text, lang)
    is_noise = hard_noise if hard_noise else bool(llm_noise)

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

    # Update last user activity timestamp (for timeout tracking)
    session.last_user_activity_at = time.time()
    logger.debug("Last user activity: %.0f", session.last_user_activity_at)

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


def _handle_image_only(
    session: Session,
    lang: str,
    asr_transcript: Optional[str],
) -> Dict[str, Any]:
    image_analysis = session.image_analysis or {}
    visual = image_analysis.get("visual_triage") or {}
    action = visual.get("action")
    triage_level = visual.get("triage_level", "URGENT")
    category = visual.get("category", "other")

    if action in ("RECAPTURE_IMAGE", "MANUAL_FALLBACK"):
        session.image_attempt_count += 1
        if session.image_attempt_count >= 2:
            session.dispatch_status = "FALLBACK_PENDING"
            msg = {
                "tr": "Görseli güvenilir analiz edemedim. Lütfen Tıbbi, Polis, İtfaiye, Trafik/Kaza veya Diğer olarak seçip 1 cümle açıklama yazın.",
                "en": "I could not reliably analyze the image. Please choose Medical, Police, Fire, Traffic/Accident, or Other and add one short sentence.",
            }.get(lang, "Please choose a category and add one short sentence.")
        else:
            msg = {
                "tr": "Görsel net değil veya model şu an kullanılamıyor. Mümkünse daha net fotoğraf gönderin ya da olayı kısaca yazın.",
                "en": "The image is unclear or the model is unavailable. Please send a clearer photo or briefly describe the emergency.",
            }.get(lang, "Please send a clearer photo or describe the emergency.")
        return _reply(session, msg, image_analysis=image_analysis, user_transcript=asr_transcript)

    triage = _triage_from_visual(image_analysis)
    session.triage_result = triage
    session.collected_slots.update({
        "image_category": category,
        "image_triage_level": triage_level,
        "visual_flags": visual.get("visual_flags", []),
    })

    if action == "EARLY_DISPATCH":
        _mark_dispatch(session, category)
        msg = {
            "tr": "Görselde kritik risk tespit edildi. Ekipler yönlendiriliyor. Güvenli alana geçin; bina, kat, daire veya tam konumu yazabilirseniz ekiplere ileteceğim.",
            "en": "Critical risk was detected in the image. Emergency services are being dispatched. Move to a safe area; send building, floor, apartment, or exact location if you can.",
        }.get(lang, "Emergency services are being dispatched.")
        return _reply(session, msg, triage_result=triage, image_analysis=image_analysis)

    if action == "VERIFY_THEN_DISPATCH":
        msg = {
            "tr": "Görsel acil durum belirtisi gösteriyor. Yaralı, duman/alev, silah veya mahsur kalan biri var mı? Kısaca yazın.",
            "en": "The image suggests an emergency. Is there an injured person, smoke/fire, a weapon, or someone trapped? Please answer briefly.",
        }.get(lang, "Please briefly verify the emergency.")
        return _reply(session, msg, triage_result=triage, image_analysis=image_analysis)

    msg = {
        "tr": "Görselde net bir acil durum belirtisi görünmüyor. Yine de acil bir durum varsa lütfen kısaca açıklayın.",
        "en": "The image does not show a clear emergency. If there is still an emergency, please briefly describe it.",
    }.get(lang, "Please briefly describe the emergency.")
    return _reply(session, msg, triage_result=triage, image_analysis=image_analysis)


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
    # FAZ 3: Groq Triage (Turn 1 only)
    # On first user message: run triage (category + severity)
    # Lock the category so LLM doesn't change it across turns
    # ------------------------------------------------------------------
    user_turn_count = sum(1 for m in session.messages if m.get("role") == "user")
    
    if user_turn_count == 1 and session.initial_triage is None:
        # First turn: perform Groq triage
        logger.info("Turn 1: Running Groq triage...")
        t0 = time.monotonic()
        
        # Call Groq for initial triage
        triage_result = llm.chat(
            history=session.messages,
            language=lang,
            task="triage"  # Signal to LLM: perform triage, not dialog
        )
        logger.info("  [TIMING] LLM Triage: %.2fs", time.monotonic() - t0)
        
        # Save initial triage result
        initial_triage = {
            "category": triage_result.get("category", "other"),
            "triage_level": triage_result.get("triage_level", "URGENT"),
            "confidence": triage_result.get("confidence", 0.85),
            "red_flags": triage_result.get("red_flags", []),
        }
        session.initial_triage = initial_triage
        logger.info("Initial triage saved: category=%s, level=%s",
                    initial_triage["category"], initial_triage["triage_level"])
    
    # Lock category from initial triage (prevent LLM from changing it)
    locked_category = (session.initial_triage or {}).get("category", "other")

    t0 = time.monotonic()
    llm_result = llm.chat(
        history=session.messages,
        language=lang,
        session_context={  # Pass locked triage to LLM
            "initial_category": locked_category,
            "initial_triage_level": (session.initial_triage or {}).get("triage_level", "URGENT"),
            "dispatch_status": session.dispatch_status,
            "witness_mode": session.witness_mode,
        }
    )
    logger.info("  [TIMING] LLM Dialog: %.2fs", time.monotonic() - t0)

    response_text: str = llm_result.get("response_text", "")
    extracted_slots: Dict[str, Any] = llm_result.get("extracted_slots", {})
    triage_level: str = llm_result.get("triage_level", "URGENT")
    category: str = llm_result.get("category", "other")
    is_complete: bool = llm_result.get("is_complete", False)
    red_flags: List[str] = llm_result.get("red_flags", [])

    # ------------------------------------------------------------------
    # FAZ 5: Slot Attempt Tracking (2-Attempt Rule)
    # If a slot was asked but not filled, increment its attempt counter
    # ------------------------------------------------------------------
    if session.pending_question_key and session.pending_question_key not in extracted_slots:
        from orchestrator.dialog_manager import increment_slot_attempt
        increment_slot_attempt(session, session.pending_question_key)
        logger.debug("Slot %s not filled by LLM, incrementing attempt counter (current: %d)",
                     session.pending_question_key, session.slot_attempt_counts.get(session.pending_question_key, 0))
        session.pending_question_key = None  # Reset for next question

    # ------------------------------------------------------------------
    # Guard 1 — Category locking (Updated for Groq Primary)
    # Once Groq's initial triage is set, never let the LLM flip it.
    # LLM must respect the locked category from Turn 1.
    # ------------------------------------------------------------------
    if locked_category and locked_category != "other":
        category = locked_category
        triage_level = (session.initial_triage or {}).get("triage_level", triage_level)
        logger.debug("Category & triage locked from initial Groq triage: category=%s, level=%s",
                     locked_category, triage_level)

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
    # FAZ 1.5: Dispatch Lock (Safety-Critical)
    # Prevent redispatch of the same case within 48 hours
    # ------------------------------------------------------------------
    if is_complete and triage_level in ("CRITICAL", "URGENT"):
        if not can_redispatch(session, redispatch_ttl_seconds=48 * 3600):
            # Already dispatched recently, don't dispatch again
            logger.warning("Dispatch lock active: same case already dispatched. Preventing redispatch.")
            is_complete = False  # Keep conversation open, don't force completion
            response_text = {
                "tr": "Bu durum zaten kayda alındı. Daha fazla bilgi var mı?",
                "en": "This case has already been reported. Is there any additional information?",
            }.get(lang, "This case has already been reported.")
        else:
            # Can dispatch: set dispatch status and timestamp
            session.dispatch_status = "DISPATCHED"
            session.dispatch_target = category  # store which service to dispatch to
            session.dispatch_timestamp = time.time()
            logger.info("Dispatch lock set: vaka_id=%s, target=%s, TTL=48h",
                        session.session_id, category)

    # ------------------------------------------------------------------
    # Guard 3 — Maximum turn limit (Dynamic based on triage_level)
    # FAZ 6: Operasyonel Mantik — Escalation Control
    # CRITICAL: 1-2 turns max (immediate dispatch)
    # URGENT: 4-5 turns max (collect essential + quick dispatch)
    # NON_CRITICAL: 8 turns max (normal slot filling, soft close)
    # ------------------------------------------------------------------
    max_turns_map = {
        "CRITICAL": 2,      # Critical: almost no delay
        "URGENT": 4,        # Urgent: collect essentials quickly
        "NON_URGENT": 8,    # Non-urgent: normal dialog flow
    }
    max_turns = max_turns_map.get(triage_level, 8)
    
    user_turn_count = sum(1 for m in session.messages if m.get("role") == "user")
    
    # FAZ 6: CRITICAL escalation — force completion after max turns
    if triage_level == "CRITICAL" and user_turn_count >= max_turns:
        is_complete = True
        logger.info("CRITICAL escalation: max turns (%d) reached – forcing dispatch.", max_turns)
        if not response_text:
            response_text = _DISPATCH_MSG.get(lang, _DISPATCH_MSG["en"])
    
    # FAZ 6: URGENT escalation — collect key slots quickly, then dispatch
    elif triage_level == "URGENT" and user_turn_count >= max_turns:
        is_complete = True
        logger.info("URGENT escalation: max turns (%d) reached – dispatching.", max_turns)
        if not response_text:
            response_text = _DISPATCH_MSG.get(lang, _DISPATCH_MSG["en"])
    
    # FAZ 6: NON_CRITICAL soft close — offer graceful exit
    elif triage_level == "NON_URGENT" and user_turn_count >= max_turns - 1:
        # At turn 7 (max 8), offer soft close
        is_complete = True
        logger.info("NON_CRITICAL: approaching max turns – soft close.")
        if not response_text:
            soft_close = {
                "tr": "Teşekkürler, yeterli bilgiye sahibiz. Durumunuz kaydedildi. Daha fazla yardıma ihtiyacınız varsa 112'yi arayın.",
                "en": "Thank you. We have sufficient information. Your case recorded. Call 112 if needed.",
            }.get(lang, "Thank you. Case recorded. Call 112 if needed.")
            response_text = soft_close

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
        followup_status = "dispatch_sent" if session.dispatch_timestamp else "dispatch_pending"
    elif session.dispatch_status == "PENDING":
        followup_status = "waiting_for_info" if not session.is_complete else "no_dispatch_needed"

    nearby_places = _resolve_nearby_places(session, triage_result or session.triage_result)
    
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
