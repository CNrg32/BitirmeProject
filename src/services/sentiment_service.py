from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _BASE / "out_models"

PANIC_KEYWORDS = [
    "help", "fuck", "shit", "blood", "dying", "gun", "please",
    "urgent", "kill", "fire", "can't breathe", "stabbed", "shot",
    "ambulance", "dead", "hurry", "emergency", "weapon", "scream",
    "attack", "assault", "overdose", "unconscious", "trapped",
]

TEXT_FEATURES = [
    "panic_word_ratio", "panic_word_count", "word_count", "avg_word_length",
    "sentence_count", "exclamation_ratio", "question_ratio", "uppercase_ratio",
    "repetition_ratio", "emergency_phrase_count", "sentiment_score",
    "estimated_wpm", "wpm",
]
AUDIO_FEATURES = [
    "caller_pitch_mean", "spectral_flux", "silence_ratio", "audio_duration",
]
META_FEATURES = [
    "deaths", "potential_death", "false_alarm", "civilian_initiated",
]
TRIAGE_LABELS = ["CRITICAL", "URGENT", "NON_URGENT"]


def extract_text_features(text: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    panic_count = sum(1 for w in words if any(k in w for k in PANIC_KEYWORDS))
    feats["panic_word_ratio"] = panic_count / word_count if word_count > 0 else 0.0
    feats["panic_word_count"] = float(panic_count)

    feats["word_count"] = float(word_count)
    feats["avg_word_length"] = (
        float(np.mean([len(w) for w in words])) if words else 0.0
    )

    sentences = [
        s.strip()
        for s in text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    feats["sentence_count"] = float(len(sentences))

    feats["exclamation_ratio"] = (
        text.count("!") / word_count if word_count > 0 else 0.0
    )
    feats["question_ratio"] = (
        text.count("?") / word_count if word_count > 0 else 0.0
    )

    alpha_chars = [c for c in text if c.isalpha()]
    feats["uppercase_ratio"] = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0.0
    )

    if word_count > 1:
        unique_words = set(words)
        feats["repetition_ratio"] = 1.0 - (len(unique_words) / word_count)
    else:
        feats["repetition_ratio"] = 0.0

    emergency_phrases = [
        "need help", "send help", "hurry up", "right now", "come quick",
        "shots fired", "officer down", "code 3", "respond to",
        "not breathing", "no pulse", "unconscious", "bleeding out",
    ]
    feats["emergency_phrase_count"] = float(
        sum(1 for p in emergency_phrases if p in text_lower)
    )

    return feats


def get_sentiment_score(text: str) -> float:
    """Sentiment score: -1 (panic) to +1 (calm). Fallback to keyword-based if no HF."""
    try:
        from transformers import pipeline as hf_pipeline
        _pipe = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        result = _pipe(text[:512])[0]
        score = result["score"]
        return -score if result["label"] == "NEGATIVE" else score
    except Exception:
        pass

    text_lower = text.lower()
    neg_count = sum(1 for k in PANIC_KEYWORDS if k in text_lower)
    pos_words = ["okay", "fine", "alright", "thank", "stable", "calm"]
    pos_count = sum(1 for k in pos_words if k in text_lower)
    total = neg_count + pos_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def extract_audio_features_from_bytes(audio_bytes: bytes) -> Dict[str, float]:
    """Extract audio features from raw bytes (WAV/MP3/FLAC etc.)."""
    feats = {
        "caller_pitch_mean": 0.0,
        "spectral_flux": 0.0,
        "silence_ratio": 0.0,
        "audio_duration": 0.0,
        "audio_found": False,
    }
    try:
        import librosa
    except ImportError:
        logger.debug("librosa not available, skipping audio features")
        return feats

    if not audio_bytes or len(audio_bytes) < 1000:
        return feats

    try:
        buf = io.BytesIO(audio_bytes)
        y, sr = librosa.load(buf, duration=60, sr=None)
        feats["audio_found"] = True
        feats["audio_duration"] = float(librosa.get_duration(y=y, sr=sr))

        S = np.abs(librosa.stft(y))
        pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
        threshold = np.mean(magnitudes) * 1.5
        caller_pitches = pitches[magnitudes > threshold]
        caller_pitches = caller_pitches[caller_pitches > 60]
        if len(caller_pitches) > 0:
            feats["caller_pitch_mean"] = float(np.mean(caller_pitches))

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        feats["spectral_flux"] = float(np.mean(onset_env))

        intervals = librosa.effects.split(y, top_db=25)
        non_silent_dur = sum(x[1] - x[0] for x in intervals) / sr
        total_dur = feats["audio_duration"]
        feats["silence_ratio"] = (
            1.0 - (non_silent_dur / total_dur) if total_dur > 0 else 0.0
        )
    except Exception as e:
        logger.debug("Audio feature extraction failed: %s", e)

    return feats


def build_feature_vector(
    text_en: str,
    audio_bytes: Optional[bytes] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Build full feature dict for sentiment model (text + audio + meta).
    Caller uses feature_order from model meta to build the final vector.
    """
    meta = meta or {}
    feats: Dict[str, float] = {}

    # Text features
    text_f = extract_text_features(text_en)
    feats.update(text_f)
    feats["sentiment_score"] = get_sentiment_score(text_en)
    feats["estimated_wpm"] = (
        feats["word_count"] / max(feats["sentence_count"], 1) * 10
    )

    # Audio features
    if audio_bytes:
        audio_f = extract_audio_features_from_bytes(audio_bytes)
        feats.update({k: audio_f[k] for k in AUDIO_FEATURES})
        if audio_f["audio_found"] and audio_f["audio_duration"] > 0:
            feats["wpm"] = (feats["word_count"] / audio_f["audio_duration"]) * 60
        else:
            feats["wpm"] = feats["estimated_wpm"]
    else:
        feats["caller_pitch_mean"] = 0.0
        feats["spectral_flux"] = 0.0
        feats["silence_ratio"] = 0.0
        feats["audio_duration"] = 0.0
        feats["wpm"] = feats["estimated_wpm"]

    # Meta
    for col in META_FEATURES:
        val = meta.get(col, 0)
        try:
            feats[col] = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            feats[col] = 0.0

    return feats


class SentimentService:
    """Load sentiment model + scaler and run inference from text + optional audio."""

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.feature_order: List[str] = []
        self._loaded = False

    def load(self, model_dir: Path | str | None = None) -> bool:
        model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        model_path = model_dir / "sentiment_voting_model.joblib"
        scaler_path = model_dir / "sentiment_scaler.joblib"
        meta_path = model_dir / "sentiment_model_meta.json"

        if not model_path.exists() or not scaler_path.exists():
            logger.warning(
                "Sentiment model not found at %s – sentiment analysis disabled.",
                model_dir,
            )
            return False

        try:
            import joblib
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                self.feature_order = meta.get("features", TEXT_FEATURES + AUDIO_FEATURES + META_FEATURES)
            else:
                self.feature_order = TEXT_FEATURES + AUDIO_FEATURES + META_FEATURES
            self._loaded = True
            logger.info("Sentiment model loaded from %s", model_dir)
            return True
        except Exception as e:
            logger.exception("Failed to load sentiment model: %s", e)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(
        self,
        text_en: str,
        audio_bytes: Optional[bytes] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run sentiment model. Returns dict with triage_level, confidence, panic_level, etc.
        Returns None if model not loaded or prediction fails.
        """
        if not self._loaded:
            return None

        try:
            feats = build_feature_vector(text_en, audio_bytes, meta)
            X = np.array(
                [[feats.get(f, 0.0) for f in self.feature_order]],
                dtype=np.float64,
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.transform(X)
            pred_label = self.model.predict(X_scaled)[0]
            proba = self.model.predict_proba(X_scaled)[0]
            classes = list(self.model.classes_)
            idx = classes.index(pred_label) if pred_label in classes else 0
            confidence = float(proba[idx])

            sentiment_score = feats.get("sentiment_score", 0.0)
            panic_level = "high" if sentiment_score < -0.3 else ("medium" if sentiment_score < 0.2 else "low")

            return {
                "triage_level": pred_label,
                "confidence": confidence,
                "panic_level": panic_level,
                "sentiment_score": sentiment_score,
                "proba": {c: float(p) for c, p in zip(classes, proba)},
            }
        except Exception as e:
            logger.exception("Sentiment prediction failed: %s", e)
            return None


_sentiment_service: Optional[SentimentService] = None


def get_sentiment_service() -> SentimentService:
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentService()
    return _sentiment_service
