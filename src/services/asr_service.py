from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

_model = None
_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "base")


def _get_model():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel

            logger.info("Loading Whisper model '%s' …", _MODEL_SIZE)
            t0 = time.monotonic()
            _model = WhisperModel(
                _MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            logger.info(
                "Whisper model '%s' loaded in %.1fs.",
                _MODEL_SIZE,
                time.monotonic() - t0,
            )
        except ImportError:
            logger.warning(
                "faster-whisper is not installed. ASR will not be available. "
                "Install with: pip install faster-whisper"
            )
            raise
    return _model


def preload_model() -> bool:
    """Pre-load the Whisper model so the first request is fast."""
    try:
        _get_model()
        return True
    except Exception as exc:
        logger.warning("ASR model preload failed: %s", exc)
        return False


def transcribe_audio(
    audio_bytes: bytes | None = None,
    audio_path: str | Path | None = None,
    language: str | None = None,
) -> Tuple[str, str, float]:
    t_start = time.monotonic()
    model = _get_model()

    created_tmp_path = None
    if audio_bytes is not None and audio_path is None:
        if audio_bytes[:4] == b"\x1aE\xdf\xa3":
            suffix = ".webm"
        elif audio_bytes[:4] == b"OggS":
            suffix = ".ogg"
        elif audio_bytes[:4] == b"RIFF":
            suffix = ".wav"
        else:
            suffix = ".audio"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        audio_path = tmp.name
        created_tmp_path = tmp.name

    try:
        t_transcribe = time.monotonic()
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        texts = []
        total_prob = 0.0
        count = 0
        for seg in segments:
            texts.append(seg.text.strip())
            total_prob += seg.avg_logprob
            count += 1

        transcript = " ".join(texts)
        avg_confidence = (total_prob / count) if count else 0.0
        detected_lang = info.language or "en"

        t_end = time.monotonic()
        logger.info(
            "ASR done in %.2fs (transcribe=%.2fs) lang=%s text=%.60s…",
            t_end - t_start,
            t_end - t_transcribe,
            detected_lang,
            transcript,
        )

        return transcript, detected_lang, avg_confidence
    finally:
        if created_tmp_path and os.path.exists(created_tmp_path):
            try:
                os.unlink(created_tmp_path)
            except OSError:
                pass
