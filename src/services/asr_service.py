from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

_model = None
# Default `small`: better TR/EN recognition than `base`; use `ASR_MODEL_SIZE=base` on weak CPUs.
_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")
_DEVICE = (os.getenv("ASR_DEVICE") or "").strip().lower() or None  # cpu, cuda, mps
_COMPUTE_TYPE = (os.getenv("ASR_COMPUTE_TYPE") or "").strip().lower() or None  # int8, float16


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r — using default %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


# Higher beam improves stability; lower is faster (default 5 per quality/consistency tradeoff).
_BEAM_SIZE = max(1, _env_int("ASR_BEAM_SIZE", 5))
_CONDITION_PREV = _env_bool("ASR_CONDITION_ON_PREVIOUS_TEXT", False)
_VAD_FILTER = _env_bool("ASR_VAD_FILTER", True)
# When True and ffmpeg is on PATH, decode to 16 kHz mono WAV before Whisper (recommended for mobile/webm).
_ASR_PREPROCESS = _env_bool("ASR_PREPROCESS", True)


def _effective_device() -> str:
    if _DEVICE == "cuda":
        return "cuda"
    if _DEVICE == "mps":
        return "mps"
    return "cpu"


def _effective_compute_type() -> str:
    dev = _effective_device()
    if _COMPUTE_TYPE in ("float16", "int8"):
        return _COMPUTE_TYPE
    if dev != "cpu" and not _COMPUTE_TYPE:
        return "float16"
    return "int8"


def get_asr_runtime_info() -> dict:
    """Snapshot for logging and tests (model id, decode options, preprocess)."""
    return {
        "model_size": _MODEL_SIZE,
        "device": _effective_device(),
        "compute_type": _effective_compute_type(),
        "beam_size": _BEAM_SIZE,
        "condition_on_previous_text": _CONDITION_PREV,
        "vad_filter": _VAD_FILTER,
        "preprocess_16k_mono": _ASR_PREPROCESS,
    }


def get_asr_runtime_info_str() -> str:
    d = get_asr_runtime_info()
    return (
        f"model={d['model_size']} device={d['device']} beam={d['beam_size']} "
        f"vad={d['vad_filter']} condition_prev={d['condition_on_previous_text']} "
        f"preprocess={d['preprocess_16k_mono']}"
    )


def _get_model():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel

            device = _effective_device()
            compute_type = _effective_compute_type()

            logger.info("Loading Whisper model '%s' (device=%s, compute_type=%s) …", _MODEL_SIZE, device, compute_type)
            t0 = time.monotonic()
            _model = WhisperModel(
                _MODEL_SIZE,
                device=device,
                compute_type=compute_type,
            )
            logger.info(
                "Whisper model '%s' loaded in %.1fs.",
                _MODEL_SIZE,
                time.monotonic() - t0,
            )
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. ASR is unavailable until you install it "
                "(e.g. `pip install faster-whisper` or `pip install -r requirements.txt` "
                "from the project root)."
            ) from exc
    return _model


def _transcode_to_wav_16k_mono(src_path: str) -> tuple[str, bool]:
    """Return (path_for_whisper, is_temp). If preprocess fails, returns (src_path, False)."""
    if not _ASR_PREPROCESS:
        return src_path, False
    if not shutil.which("ffmpeg"):
        logger.debug("ASR preprocess: ffmpeg not on PATH; using original file")
        return src_path, False
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        src_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return out_path, True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("ASR ffmpeg preprocess failed (%s); using original file", exc)
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return src_path, False


def _prepare_audio_path(
    audio_bytes: bytes | None,
    audio_path: str | Path | None,
) -> tuple[str | Path | None, str | None, str | None]:
    """Returns (path_for_model, created_raw_temp, preprocessed_wav_temp)."""
    created_raw_path = None
    preprocessed_path = None

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
        created_raw_path = tmp.name

    if audio_path is None:
        return None, created_raw_path, preprocessed_path

    use_path, did_preprocess = _transcode_to_wav_16k_mono(str(audio_path))
    if did_preprocess:
        preprocessed_path = use_path
        return use_path, created_raw_path, preprocessed_path

    return audio_path, created_raw_path, preprocessed_path


def _transcribe_opts() -> dict:
    return {
        "beam_size": _BEAM_SIZE,
        "vad_filter": _VAD_FILTER,
        "condition_on_previous_text": _CONDITION_PREV,
    }


def preload_model() -> bool:
    """Pre-load the Whisper model so the first request is fast."""
    try:
        _get_model()
        return True
    except ImportError as exc:
        logger.warning("ASR preload skipped — %s", exc)
        return False
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

    path_for_model, created_raw_path, preprocessed_path = _prepare_audio_path(audio_bytes, audio_path)
    if path_for_model is None:
        return "", "en", 0.0

    opts = _transcribe_opts()
    try:
        t_transcribe = time.monotonic()
        segments, info = model.transcribe(
            str(path_for_model),
            language=language,
            beam_size=opts["beam_size"],
            vad_filter=opts["vad_filter"],
            condition_on_previous_text=opts["condition_on_previous_text"],
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
            "ASR done in %.2fs (transcribe=%.2fs) %s lang=%s text=%.60s…",
            t_end - t_start,
            t_end - t_transcribe,
            get_asr_runtime_info_str(),
            detected_lang,
            transcript,
        )

        return transcript, detected_lang, avg_confidence
    finally:
        for p in (preprocessed_path, created_raw_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def transcribe_audio_streaming(
    audio_bytes: bytes | None = None,
    audio_path: str | Path | None = None,
    language: str | None = None,
    on_partial_text: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str, float]:
    """
    Transcribe audio and optionally call on_partial_text(partial_transcript) after each
    segment. This allows incremental processing (e.g. early slot extraction) while
    transcription is still running. Returns the same (full_transcript, detected_lang,
    avg_confidence) as transcribe_audio.
    """
    t_start = time.monotonic()
    model = _get_model()

    path_for_model, created_raw_path, preprocessed_path = _prepare_audio_path(audio_bytes, audio_path)
    if path_for_model is None:
        return "", "en", 0.0

    opts = _transcribe_opts()
    try:
        t_transcribe = time.monotonic()
        segments, info = model.transcribe(
            str(path_for_model),
            language=language,
            beam_size=opts["beam_size"],
            vad_filter=opts["vad_filter"],
            condition_on_previous_text=opts["condition_on_previous_text"],
        )

        texts = []
        total_prob = 0.0
        count = 0
        for seg in segments:
            texts.append(seg.text.strip())
            total_prob += seg.avg_logprob
            count += 1
            if on_partial_text:
                partial = " ".join(texts).strip()
                if partial:
                    try:
                        on_partial_text(partial)
                    except Exception as exc:
                        logger.debug("on_partial_text callback failed: %s", exc)

        transcript = " ".join(texts)
        avg_confidence = (total_prob / count) if count else 0.0
        detected_lang = info.language or "en"

        t_end = time.monotonic()
        logger.info(
            "ASR streaming done in %.2fs (transcribe=%.2fs) %s lang=%s text=%.60s…",
            t_end - t_start,
            t_end - t_transcribe,
            get_asr_runtime_info_str(),
            detected_lang,
            transcript,
        )

        return transcript, detected_lang, avg_confidence
    finally:
        for p in (preprocessed_path, created_raw_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
