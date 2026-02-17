from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

_model = None
_MODEL_SIZE = "base"


def _get_model():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel

            logger.info("Loading Whisper model '%s' …", _MODEL_SIZE)
            _model = WhisperModel(
                _MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            logger.info("Whisper model loaded.")
        except ImportError:
            logger.warning(
                "faster-whisper is not installed. ASR will not be available. "
                "Install with: pip install faster-whisper"
            )
            raise
    return _model


def transcribe_audio(
    audio_bytes: bytes | None = None,
    audio_path: str | Path | None = None,
    language: str | None = None,
) -> Tuple[str, str, float]:
    model = _get_model()

    created_tmp_path = None
    if audio_bytes is not None and audio_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        audio_path = tmp.name
        created_tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        texts = []
        total_prob = 0.0
        count = 0
        for seg in segments:
            texts.append(seg.text.strip())
            total_prob += seg.avg_logprob
            count += 1

        transcript = " ".join(texts)
        # avg_logprob is negative; expose a simple 0-1 style value: clamp and invert for "confidence"
        avg_confidence = (total_prob / count) if count else 0.0
        detected_lang = info.language or "en"

        return transcript, detected_lang, avg_confidence
    finally:
        if created_tmp_path and os.path.exists(created_tmp_path):
            try:
                os.unlink(created_tmp_path)
            except OSError:
                pass
