from __future__ import annotations

import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Load .env file (GEMINI_API_KEY etc.) before any other imports
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_ENV_FILE, override=False)
    except ImportError:
        pass  # python-dotenv not installed; use system env vars

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.schemas import (
    ImageAnalysisResult,
    NearbyPlacesRequest,
    NearbyPlacesResponse,
    PredictRequest,
    PredictResponse,
    SessionMessageRequest,
    SessionMessageResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionTranscribeRequest,
    SessionTranscribeResponse,
    TranscriptImportRequest,
    TranscriptImportResponse,
    TranscriptListResponse,
    TranscriptRecordResponse,
)
from api.model_loader import apply_redflag_override, get_model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def simulate_fallback_for_session(session_id: str, text: str, scenario: str) -> Dict[str, Any]:
    """Dev helper for Faz 10.1 fallback simulation."""
    from orchestrator.session import get_session_store
    from services.text_analyze_mock import analyze_text_mock

    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        return {"error": "Session not found or expired."}

    result = analyze_text_mock(text=text, scenario=scenario)
    status = result.get("status")
    if status in ("fail", "uncertain"):
        session.dispatch_status = "FALLBACK_PENDING"
        return {
            "session_id": session_id,
            "chatbot_mode": "fallback",
            "dispatch_status": session.dispatch_status,
            "text_analyze_result": result,
            "fallback_options": ["medical", "crime", "fire"],
            "prompt": "Kategori secin ve kisa aciklama yazin.",
        }

    return {
        "session_id": session_id,
        "chatbot_mode": "normal",
        "dispatch_status": session.dispatch_status,
        "text_analyze_result": result,
        "fallback_options": [],
        "prompt": "Normal akis devam ediyor.",
    }


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
        greeting_audio_url=out.get("greeting_audio_url"),
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
        latitude=req.latitude,
        longitude=req.longitude,
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
        assistant_audio_url=out.get("assistant_audio_url"),
        assistant_audio_b64=out.get("assistant_audio_b64"),
        user_transcript=out.get("user_transcript"),
        triage_result=triage,
        image_analysis=image_analysis,
        report=out.get("report"),
        is_complete=out.get("is_complete", False),
        chatbot_mode=out.get("chatbot_mode", "normal"),
        dispatch_status=out.get("dispatch_status", "PENDING"),
        summary_card=out.get("summary_card"),
        resume_prompt=out.get("resume_prompt"),
        followup_status=out.get("followup_status"),
        nearby_places=out.get("nearby_places"),
    )


@app.post("/nearby-places", response_model=NearbyPlacesResponse)
def nearby_places(req: NearbyPlacesRequest):
    from services.nearby_places_service import get_nearby_places

    places = get_nearby_places(
        latitude=req.latitude,
        longitude=req.longitude,
        preferred_type=req.preferred_type,
        limit_per_type=req.limit_per_type,
    )
    return NearbyPlacesResponse(nearby_places=places)


@app.post("/transcripts/import-auto-csv", response_model=TranscriptImportResponse)
def import_auto_transcripts(req: TranscriptImportRequest):
    try:
        from services.transcript_store import (
            DEFAULT_AUTO_TRANSCRIPTS_CSV,
            get_transcript_store,
            load_auto_transcripts_csv,
        )

        csv_path = req.csv_path or DEFAULT_AUTO_TRANSCRIPTS_CSV
        records = load_auto_transcripts_csv(csv_path)
        store = get_transcript_store()
        imported_count = store.put_many(records)
        return TranscriptImportResponse(
            imported_count=imported_count,
            table_name=getattr(store, "table_name", None),
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, f"CSV file not found: {exc.filename}")


@app.get("/transcripts", response_model=TranscriptListResponse)
def list_transcripts(limit: int = 50):
    try:
        from services.transcript_store import get_transcript_store

        safe_limit = min(max(limit, 1), 200)
        items = get_transcript_store().list(limit=safe_limit)
        return TranscriptListResponse(
            transcripts=[TranscriptRecordResponse(**item) for item in items]
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/transcripts/{transcript_id}", response_model=TranscriptRecordResponse)
def get_transcript(transcript_id: str):
    try:
        from services.transcript_store import get_transcript_store

        item = get_transcript_store().get(transcript_id)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))

    if item is None:
        raise HTTPException(404, "Transcript not found.")
    return TranscriptRecordResponse(**item)


@app.post("/test/simulate-fallback")
def test_simulate_fallback(
    session_id: str = Form(...),
    text: str = Form("mock test"),
    scenario: str = Form("uncertain"),
):
    """Faz 10.1 Test 1/3: Dev-only endpoint to simulate TextAnalyze fail/uncertain."""
    if os.environ.get("ENABLE_TEST_ENDPOINTS", "false").lower() != "true":
        raise HTTPException(404, "Test endpoints are disabled.")

    out = simulate_fallback_for_session(session_id=session_id, text=text, scenario=scenario)
    if "error" in out:
        raise HTTPException(404, out["error"])
    return out


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

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(_SRC))
