from __future__ import annotations

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
TRIAGE_LABELS = ["CRITICAL", "URGENT", "NON_URGENT"]
TEXT_ONLY_FEATURES = list(TEXT_FEATURES)
EXPECTED_FEATURE_PIPELINE_VERSION = "feature-pipeline-v2.0"
MIN_META_SCHEMA_VERSION = 2
RISK_WEIGHTS = {
    "CRITICAL": 1.0,
    "URGENT": 0.6,
    "NON_URGENT": 0.2,
}

FEATURE_RISK_DIRECTIONS = {
    "panic_word_ratio": 1.0,
    "panic_word_count": 1.0,
    "emergency_phrase_count": 1.0,
    "exclamation_ratio": 0.6,
    "uppercase_ratio": 0.4,
    "repetition_ratio": 0.5,
    "wpm": 0.4,
    "sentiment_score": -1.0,
}


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


# Cached HuggingFace sentiment pipeline (single load for performance)
_hf_sentiment_pipeline = None


def _get_hf_sentiment_pipeline():
    global _hf_sentiment_pipeline
    if _hf_sentiment_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _hf_sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )
        except Exception:
            pass
    return _hf_sentiment_pipeline


def get_sentiment_score(text: str) -> float:
    """Sentiment score: -1 (panic) to +1 (calm). Fallback to keyword-based if no HF."""
    pipe = _get_hf_sentiment_pipeline()
    if pipe is not None:
        try:
            result = pipe(text[:512])[0]
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


def build_feature_vector(
    text_en: str,
    audio_bytes: Optional[bytes] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Build feature dict for text-only sentiment model.
    Caller uses feature_order from model meta to build the final vector.
    """
    feats: Dict[str, float] = {}

    # Text features
    text_f = extract_text_features(text_en)
    feats.update(text_f)
    feats["sentiment_score"] = get_sentiment_score(text_en)
    feats["estimated_wpm"] = (
        feats["word_count"] / max(feats["sentence_count"], 1) * 10
    )
    feats["wpm"] = feats["estimated_wpm"]

    return feats


class SentimentService:
    """Load sentiment model + scaler and run text-only inference."""

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.feature_order: List[str] = []
        self.model_version: str = "unknown"
        self.feature_pipeline_version: str = "unknown"
        self.meta_schema_version: int = 0
        self._loaded = False

    def _validate_meta(self, meta: Dict[str, Any]) -> bool:
        known_features = set(TEXT_ONLY_FEATURES)
        meta_features = meta.get("features") or []

        if meta_features:
            unknown_features = sorted(set(meta_features) - known_features)
            if unknown_features:
                logger.warning(
                    "Sentiment model meta has unknown features: %s",
                    unknown_features,
                )
                return False

        schema_version = int(meta.get("meta_schema_version", 0) or 0)
        if schema_version and schema_version < MIN_META_SCHEMA_VERSION:
            logger.warning(
                "Sentiment model meta schema too old (%s < %s)",
                schema_version,
                MIN_META_SCHEMA_VERSION,
            )
            return False

        pipeline_version = str(meta.get("feature_pipeline_version", "") or "")
        if pipeline_version and pipeline_version != EXPECTED_FEATURE_PIPELINE_VERSION:
            logger.warning(
                "Sentiment feature pipeline mismatch (model=%s, runtime=%s)",
                pipeline_version,
                EXPECTED_FEATURE_PIPELINE_VERSION,
            )
            return False

        return True

    def _compute_risk_score(self, classes: List[str], proba: np.ndarray) -> float:
        weighted = 0.0
        for class_name, class_proba in zip(classes, proba):
            weighted += float(class_proba) * RISK_WEIGHTS.get(class_name, 0.5)
        return max(0.0, min(100.0, weighted * 100.0))

    def _panic_level_from_risk(self, risk_score: float) -> str:
        if risk_score >= 70.0:
            return "high"
        if risk_score >= 40.0:
            return "medium"
        return "low"

    def _build_feature_contributions(
        self,
        feats: Dict[str, float],
        x_scaled: np.ndarray,
    ) -> List[Dict[str, Any]]:
        indexed_scaled = {
            name: float(x_scaled[0][idx])
            for idx, name in enumerate(self.feature_order)
        }

        contributions: List[Dict[str, Any]] = []
        for feature_name, direction in FEATURE_RISK_DIRECTIONS.items():
            if feature_name not in indexed_scaled:
                continue

            scaled_val = indexed_scaled[feature_name]
            raw_val = float(feats.get(feature_name, 0.0))
            impact = scaled_val * float(direction)

            contributions.append(
                {
                    "feature": feature_name,
                    "value": raw_val,
                    "scaled_value": scaled_val,
                    "direction": "risk_up" if direction > 0 else "risk_down",
                    "impact": float(impact),
                }
            )

        contributions.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return contributions[:6]

    def load(self, model_dir: Path | str | None = None) -> bool:
        model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        text_only_model_path = model_dir / "sentiment_voting_model_text_only.joblib"
        text_only_scaler_path = model_dir / "sentiment_scaler_text_only.joblib"
        text_only_meta_path = model_dir / "sentiment_model_meta_text_only.json"

        if text_only_model_path.exists() and text_only_scaler_path.exists():
            model_path = text_only_model_path
            scaler_path = text_only_scaler_path
            meta_path = text_only_meta_path
        else:
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

                if not self._validate_meta(meta):
                    return False

                self.feature_order = meta.get("features", TEXT_ONLY_FEATURES)
                self.model_version = str(meta.get("model_version", "unknown"))
                self.feature_pipeline_version = str(meta.get("feature_pipeline_version", "unknown"))
                self.meta_schema_version = int(meta.get("meta_schema_version", 0) or 0)
            else:
                self.feature_order = TEXT_ONLY_FEATURES
                self.model_version = "legacy"
                self.feature_pipeline_version = "legacy"
                self.meta_schema_version = 0
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
            feats = build_feature_vector(text_en, None, None)
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
            risk_score = self._compute_risk_score(classes, proba)
            panic_level = self._panic_level_from_risk(risk_score)

            triage_boost_reason = None
            if pred_label == "CRITICAL" and confidence >= 0.5:
                triage_boost_reason = "text_sentiment_high_risk"
            elif panic_level == "high":
                triage_boost_reason = "panic_indicators_high"

            feature_contributions = self._build_feature_contributions(feats, X_scaled)

            return {
                "triage_level": pred_label,
                "confidence": confidence,
                "risk_score": risk_score,
                "panic_level": panic_level,
                "triage_boost_reason": triage_boost_reason,
                "feature_contributions": feature_contributions,
                "model_version": self.model_version,
                "feature_pipeline_version": self.feature_pipeline_version,
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
