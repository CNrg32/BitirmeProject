from __future__ import annotations

import base64
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.schemas import (
    ImageAnalysisResult,
    PredictRequest,
    PredictResponse,
    SessionMessageRequest,
    SessionMessageResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionTranscribeRequest,
    SessionTranscribeResponse,
)
from api.model_loader import apply_redflag_override, get_model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Emergency Triage Service ...")

    svc = get_model_service()
    loaded = svc.load()
    if loaded:
        logger.info("ML text model loaded successfully.")
    else:
        logger.warning("ML text model NOT loaded – using rule-based fallback.")

    try:
        from services.image_service import get_image_model_service

        img_svc = get_image_model_service()
        img_loaded = img_svc.load()
        if img_loaded:
            logger.info("Image model loaded successfully.")
        else:
            logger.warning("Image model NOT loaded – image analysis unavailable.")
    except Exception as exc:
        logger.warning("Image model setup skipped: %s", exc)

    try:
        from api.model_loader import load_sentiment_model

        if load_sentiment_model():
            logger.info("Sentiment model loaded successfully.")
        else:
            logger.warning("Sentiment model NOT loaded – voice sentiment analysis disabled.")
    except Exception as exc:
        logger.warning("Sentiment model setup skipped: %s", exc)

    try:
        from services.asr_service import preload_model as preload_asr

        if preload_asr():
            logger.info("ASR (Whisper) model pre-loaded successfully.")
        else:
            logger.warning("ASR model NOT pre-loaded – will load on first request.")
    except Exception as exc:
        logger.warning("ASR model preload skipped: %s", exc)

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Emergency Triage Assistant API",
    version="1.0.0",
    description="Multilingual emergency triage API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    svc = get_model_service()

    img_loaded = False
    sentiment_loaded = False
    try:
        from services.image_service import get_image_model_service
        img_loaded = get_image_model_service().is_loaded
    except Exception:
        pass
    try:
        from services.sentiment_service import get_sentiment_service
        sentiment_loaded = get_sentiment_service().is_loaded
    except Exception:
        pass

    return {
        "status": "ok",
        "ml_model_loaded": svc.is_loaded,
        "image_model_loaded": img_loaded,
        "sentiment_model_loaded": sentiment_loaded,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text_en = req.text_en
    meta = req.meta
    slots_in = req.slots

    result = _run_predict(text_en, meta, slots_in)
    return PredictResponse(**result)


def _run_predict(text_en, meta, slots_in) -> dict:
    from mvp_rules import predict_mvp, extract_slots, load_rules

    svc = get_model_service()

    rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
    auto_slots = extract_slots(text_en, rules)

    merged_slots = {**auto_slots}
    if slots_in:
        for k, v in slots_in.model_dump(exclude_none=True).items():
            if v is not None:
                merged_slots[k] = v

    meta_dict = {"deaths": meta.deaths, "potential_death": meta.potential_death, "false_alarm": meta.false_alarm}
    if svc.is_loaded:
        label, conf = svc.predict(text_en, meta=meta_dict)
        label, red_flags = apply_redflag_override(text_en, label, meta_dict)
        from mvp_rules import infer_category
        category = infer_category(text_en, rules)
    else:
        mvp_out = predict_mvp(
            text=text_en,
            deaths=meta.deaths,
            potential_death=meta.potential_death,
            false_alarm=meta.false_alarm,
        )
        label = mvp_out["triage_level"]
        conf = None
        category = mvp_out["category"]
        red_flags = mvp_out.get("red_flags", [])

    red_flags = list(set(red_flags) | set(merged_slots.get("red_flags", [])))

    return {
        "category": category,
        "triage_level": label,
        "confidence": conf,
        "red_flags": red_flags,
        "slots": merged_slots,
        "needs_more_info": False,
        "recommended_questions": [],
    }


@app.post("/session/start", response_model=SessionStartResponse)
def session_start(req: SessionStartRequest):
    from orchestrator.orchestrator import start_session

    out = start_session(language=req.language if req.language else None)
    return SessionStartResponse(
        session_id=out["session_id"],
        greeting=out["greeting"],
        greeting_audio_url=None,
        greeting_audio_b64=out.get("greeting_audio_b64"),
    )


@app.post("/session/transcribe", response_model=SessionTranscribeResponse)
def session_transcribe(req: SessionTranscribeRequest):
    from orchestrator.session import get_session_store

    store = get_session_store()
    session = store.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found or expired.")

    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 audio data.")

    try:
        from services.asr_service import transcribe_audio

        lang_hint = session.language if session.language_locked else None
        transcript, detected_lang, _conf = transcribe_audio(
            audio_bytes=audio_bytes, language=lang_hint,
        )

        if detected_lang and not session.language_locked:
            session.language = detected_lang
            session.language_locked = True

        return SessionTranscribeResponse(
            session_id=req.session_id,
            transcript=transcript,
            detected_language=detected_lang,
        )
    except Exception as exc:
        logger.error("Transcribe failed: %s", exc)
        raise HTTPException(500, f"Transcription failed: {exc}")


@app.post("/session/message", response_model=SessionMessageResponse)
def session_message(req: SessionMessageRequest):
    from orchestrator.orchestrator import handle_message

    audio_bytes = None
    if req.audio_base64:
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
        except Exception:
            raise HTTPException(400, "Invalid base64 audio data.")

    image_bytes = None
    if req.image_base64:
        try:
            image_bytes = base64.b64decode(req.image_base64)
        except Exception:
            raise HTTPException(400, "Invalid base64 image data.")

    out = handle_message(
        session_id=req.session_id,
        user_text=req.text,
        audio_bytes=audio_bytes,
        image_bytes=image_bytes,
    )

    if "error" in out:
        raise HTTPException(404, out["error"])

    triage = None
    if out.get("triage_result"):
        triage = PredictResponse(**out["triage_result"])

    image_analysis = None
    if out.get("image_analysis"):
        image_analysis = ImageAnalysisResult(**out["image_analysis"])

    return SessionMessageResponse(
        session_id=out["session_id"],
        assistant_text=out["assistant_text"],
        assistant_audio_url=None,
        assistant_audio_b64=out.get("assistant_audio_b64"),
        user_transcript=out.get("user_transcript"),
        triage_result=triage,
        image_analysis=image_analysis,
        report=out.get("report"),
        is_complete=out.get("is_complete", False),
    )


@app.post("/analyze-image", response_model=ImageAnalysisResult)
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    text_category: Optional[str] = Form(None),
    text_triage_level: Optional[str] = Form(None),
):
    from services.image_service import analyze_image

    content = await image.read()
    if not content:
        raise HTTPException(400, "Empty image file.")

    result = analyze_image(
        image_bytes=content,
        text_category=text_category,
        text_triage_level=text_triage_level,
    )

    if not result.get("available"):
        raise HTTPException(
            503,
            detail=result.get("summary", "Image analysis model not loaded."),
        )

    return ImageAnalysisResult(**result)


@app.post("/asr")
async def asr_endpoint(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    from services.asr_service import transcribe_audio

    content = await audio.read()
    transcript, detected_lang, confidence = transcribe_audio(
        audio_bytes=content, language=language
    )
    return {
        "transcript": transcript,
        "detected_language": detected_lang,
        "confidence": confidence,
    }


@app.post("/translate")
def translate_endpoint(
    text: str = Form(...),
    source: str = Form("auto"),
    target: str = Form("en"),
):
    from services.translation_service import translate

    result = translate(text, source=source, target=target)
    return {"translated": result, "source": source, "target": target}


@app.post("/tts")
def tts_endpoint(
    text: str = Form(...),
    language: str = Form("en"),
):
    from services.tts_service import synthesize

    audio = synthesize(text, lang=language)
    if not audio:
        raise HTTPException(500, "TTS synthesis failed.")
    return Response(content=audio, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
