from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import re

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _BASE / "out_models"

PANIC_KEYWORDS = [
    "yardım", "imdat", "acil", "çabuk", "kan", "kanama", "nefes",
    "ölüyor", "ölüyorum", "öldü", "silah", "bıçak", "yangın", "ambulans",
    "bilinci", "bayıldı", "kriz", "saldırı", "yaralı", "vuruldu", "boğuluyor",
    "yanıyor", "zehirlendi", "aşırı doz", "nefes alamıyor", "öldür",
    "acıyor", "ağrıyor", "düştü", "fenalaştı", "baygın", "yığıldı",
    "göğüs", "sıkışıyor", "kalkamıyor", "çarptı",
]

EMERGENCY_PHRASES = [
    "yardım edin", "yardım et", "ambulans gönder", "hemen gelin", "çabuk gelin",
    "nefes almıyor", "nefes alamıyor", "nabız yok", "bilinci kapalı", "çok kan var",
    "kan fışkırıyor", "silah var", "bıçaklandı", "yangın var", "çocuğum nefes almıyor",
    "yere düştü", "baygın yatıyor", "göğüs ağrısı", "nefes darlığı var",
    "başını çarptı", "fenalaştı", "çok acıyor", "kalkamıyor",
    "sahipsiz paket", "şüpheli paket", "bomba ihbarı", "içinden ses geliyor",
    "çocuk ağlaması", "kapıyı açan yok", "kıvılcım çıkarıyor", "yangın çıkmasından korkuyorum",
]

POSITIVE_WORDS = [
    "iyi", "tamam", "sakin", "stabil", "düzeldi", "geçti", "kontrol altında",
]

NEGATION_WORDS = {
    "değil", "yok", "yoktur", "hayır", "değilim", "değilsin",
}

RISK_ENTITIES = [
    "bebek", "çocuk", "anne", "baba", "hamile",
    "bıçak", "silah", "kurşun", "vuruldu", "bıçaklandı", "yaralı",
    "kan", "kanama", "kanıyor", "kan fışkırıyor",
    "bayıldı", "bilinci kapalı", "koma", "yığıldı", "baygın",
    "intihar", "aşırı doz", "zehir", "uyuşturucu",
    "yangın", "yanık", "yanıyor", "boğulma", "boğuluyor",
    "nefes alamıyor", "nefes almıyor", "kalp krizi", "düştü", "fenalaştı",
    "göğüs ağrısı", "nefes darlığı", "başını çarptı", "kalkamıyor",
    "sahipsiz paket", "şüpheli paket", "bomba", "infilak",
    "çocuk ağlaması", "kavga", "kapıyı açan yok",
    "elektrik teli", "kıvılcım", "kısa devre",
]

ASCII_FOLD_MAP = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "c", "Ğ": "g", "İ": "i", "I": "i", "Ö": "o", "Ş": "s", "Ü": "u",
})

NORMALIZED_PANIC_KEYWORDS = [item.translate(ASCII_FOLD_MAP).lower() for item in PANIC_KEYWORDS]
NORMALIZED_EMERGENCY_PHRASES = [item.translate(ASCII_FOLD_MAP).lower() for item in EMERGENCY_PHRASES]
NORMALIZED_POSITIVE_WORDS = [item.translate(ASCII_FOLD_MAP).lower() for item in POSITIVE_WORDS]
NORMALIZED_NEGATION_WORDS = {item.translate(ASCII_FOLD_MAP).lower() for item in NEGATION_WORDS}
NORMALIZED_RISK_ENTITIES = [item.translate(ASCII_FOLD_MAP).lower() for item in RISK_ENTITIES]
PHYSICAL_THREAT_TERMS = ["silah", "bicak", "saldiri"]

PANIC_KEYWORD_MULTIPLIERS = {
    "oluyor": 2.0,   # from "ölüyor"
    "bicak": 2.0,    # from "bıçak"
    "silah": 2.0,
    "yangin": 2.0,   # from "yangın"
}

SHORT_CRITICAL_PHRASES = {
    "nefes alamıyor": 0.78,
    "nefes almıyor": 0.86,
    "baygın yatıyor": 0.78,
    "bilinci kapalı": 0.86,
    "çok kan var": 0.82,
    "kan fışkırıyor": 0.85,
    "kalp krizi": 0.86,
    "yığıldı": 0.74,
    "tepki vermiyor": 0.80,
    "araba takla attı": 0.82,
    "takla attı": 0.74,
    "araba takla ati": 0.82,
    "takla ati": 0.74,
    "kan kaybettim": 0.80,
    "çok kan kaybettim": 0.88,
    "bilincim kapanıyor": 0.88,
    "bilincim kapanıyo": 0.86,
    "sesimi duyuramıyorum": 0.70,
}

SHORT_URGENT_PHRASES = {
    "yere düştü": 0.54,
    "düştü": 0.44,
    "fenalaştı": 0.58,
    "baygın": 0.62,
    "nefes darlığı": 0.64,
    "göğüs ağrısı": 0.60,
    "göğsüm sıkışıyor": 0.58,
    "gogsum sikisiyo": 0.58,
    "başını çarptı": 0.58,
    "kalkamıyor": 0.58,
    "çok acıyor": 0.34,
    "acıyor": 0.26,
    "ağrıyor": 0.26,
    "kapının önünde": 0.42,
    "siyah giyimli": 0.40,
    "elinde bir şey var": 0.46,
    "ışıkları kapattım": 0.52,
    "sessizce bekliyorum": 0.44,
}

NORMALIZED_SHORT_CRITICAL_PHRASES = {
    key.translate(ASCII_FOLD_MAP).lower(): value
    for key, value in SHORT_CRITICAL_PHRASES.items()
}
NORMALIZED_SHORT_URGENT_PHRASES = {
    key.translate(ASCII_FOLD_MAP).lower(): value
    for key, value in SHORT_URGENT_PHRASES.items()
}

HARD_FLOOR_080_PHRASES = [
    "inme",
    "kan kaybi",
    "kan kaybediyorum",
    "kan kaybettim",
]

SUSPICIOUS_PACKAGE_TERMS = [
    "sahipsiz paket",
    "supheli paket",
    "bomba",
    "infilak",
]


def _normalize_match_text(text: str) -> str:
    normalized = text.translate(ASCII_FOLD_MAP).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _collapse_elongations(text: str) -> str:
    return re.sub(r"([a-z])\1+", r"\1", text)


def _weighted_panic_count(words: List[str]) -> float:
    weighted_count = 0.0
    for token in words:
        matched = [k for k in NORMALIZED_PANIC_KEYWORDS if k in token]
        if not matched:
            continue
        token_weight = max(PANIC_KEYWORD_MULTIPLIERS.get(k, 1.0) for k in matched)
        weighted_count += token_weight
    return weighted_count

TEXT_FEATURES = [
    "panic_word_ratio", "panic_word_count", "word_count", "avg_word_length",
    "sentence_count", "exclamation_ratio", "question_ratio", "uppercase_ratio",
    "repetition_ratio", "emergency_phrase_count", "sentiment_score",
    # V3.0 New features
    "vowel_lengthening_ratio", "negation_adjusted_panic_ratio", "ner_risk_density",
    "sentence_entropy_inverse",
]
TRIAGE_LABELS = ["CRITICAL", "URGENT", "NON_URGENT"]
TEXT_ONLY_FEATURES = list(TEXT_FEATURES)
EXPECTED_FEATURE_PIPELINE_VERSION = "feature-pipeline-v3.0"
MIN_META_SCHEMA_VERSION = 3
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
    "sentiment_score": -1.0,
}


def extract_text_features(text: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    normalized_text = _normalize_match_text(text)
    collapsed_text = _collapse_elongations(normalized_text)
    words = collapsed_text.split()
    word_count = len(words)

    panic_count = _weighted_panic_count(words)
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

    feats["emergency_phrase_count"] = float(
        sum(1 for p in NORMALIZED_EMERGENCY_PHRASES if p in collapsed_text)
    )

    # === V3.0 NEW FEATURES ===

    # 1. VOWEL_LENGTHENING_RATIO: Detects stretched vowels like "helppppp", "pleaseee"
    vowel_patterns = re.findall(r"([aeiou])\1{2,}", normalized_text)
    feats["vowel_lengthening_ratio"] = float(
        len(vowel_patterns) / max(word_count, 1)
    )

    # 2. NEGATION_ADJUSTED_PANIC_RATIO: Reduce panic score near negation words
    tokens = collapsed_text.split()
    adjusted_panic_count = 0
    for i, token in enumerate(tokens):
        is_panic = any(k in token for k in NORMALIZED_PANIC_KEYWORDS)
        if is_panic:
            has_negation_nearby = False
            for j in range(max(0, i-3), min(len(tokens), i+4)):
                if tokens[j] in NORMALIZED_NEGATION_WORDS or any(neg in tokens[j] for neg in NORMALIZED_NEGATION_WORDS):
                    has_negation_nearby = True
                    break
            if not has_negation_nearby:
                adjusted_panic_count += 1
    feats["negation_adjusted_panic_ratio"] = float(
        adjusted_panic_count / max(word_count, 1)
    )

    # 3. NER_RISK_DENSITY: Count high-risk entities
    risk_count = sum(1 for e in NORMALIZED_RISK_ENTITIES if e in collapsed_text)
    ner_density = float(risk_count / max(word_count, 1))
    if any(term in collapsed_text for term in PHYSICAL_THREAT_TERMS):
        ner_density *= 1.5
    feats["ner_risk_density"] = float(min(max(ner_density, 0.0), 1.0))

    # 4. SENTENCE_ENTROPY_INVERSE: Low entropy + high panic = higher risk
    if len(sentences) > 0:
        entropies = []
        for sent in sentences:
            sent_words = sent.lower().split()
            if sent_words:
                unique = len(set(sent_words))
                diversity = float(unique / len(sent_words))
                entropies.append(diversity)
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        feats["sentence_entropy_inverse"] = 1.0 - avg_entropy
    else:
        feats["sentence_entropy_inverse"] = 0.0

    return feats


def get_sentiment_score(text: str) -> float:
    """Keyword-based sentiment score for Turkish emergency text: -1 panic, +1 calm."""
    normalized_text = _collapse_elongations(_normalize_match_text(text))
    neg_count = sum(1 for k in NORMALIZED_PANIC_KEYWORDS if k in normalized_text)
    pos_count = sum(1 for k in NORMALIZED_POSITIVE_WORDS if k in normalized_text)
    total = neg_count + pos_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def compute_phrase_boost(text: str) -> float:
    text_lower = _collapse_elongations(_normalize_match_text(text))
    boost = 0.0

    for phrase, score in NORMALIZED_SHORT_CRITICAL_PHRASES.items():
        if phrase in text_lower:
            boost = max(boost, score)

    for phrase, score in NORMALIZED_SHORT_URGENT_PHRASES.items():
        if phrase in text_lower:
            boost = max(boost, score)

    if "dustu" in text_lower and ("baygin" in text_lower or "kalkamiyor" in text_lower):
        boost = max(boost, 0.66)
    if "nefes" in text_lower and ("fenalasti" in text_lower or "darligi" in text_lower):
        boost = max(boost, 0.70)
    if "cok" in text_lower and ("aciyor" in text_lower or "agriyor" in text_lower):
        boost = max(boost, 0.34)
    if "baygin" in text_lower and ("tepki vermiyor" in text_lower or "tepki vermiyo" in text_lower):
        boost = max(boost, 0.78)
    if "nefes alamiyo" in text_lower or "nefes alamiyor" in text_lower:
        boost = max(boost, 0.78)
    if "araba" in text_lower and ("takla atti" in text_lower or "takla ati" in text_lower):
        boost = max(boost, 0.82)
    if "kan kaybettim" in text_lower and ("bilincim" in text_lower or "duyuramiyorum" in text_lower):
        boost = max(boost, 0.90)
    if "kapinin onunde" in text_lower and ("elinde bir sey var" in text_lower or "siyah giyimli" in text_lower):
        boost = max(boost, 0.56)
    if "isiklari kapattim" in text_lower and ("sessizce" in text_lower or "bekliyorum" in text_lower):
        boost = max(boost, 0.60)

    if (
        ("sahipsiz" in text_lower or "supheli" in text_lower)
        and "paket" in text_lower
        and ("ses" in text_lower or "bomba" in text_lower)
    ):
        boost = max(boost, 0.88)

    if (
        "cocuk" in text_lower
        and ("aglama" in text_lower or "agliyor" in text_lower)
        and "kavga" in text_lower
    ):
        boost = max(boost, 0.62)

    if (
        ("elektrik" in text_lower or "teller" in text_lower or "priz" in text_lower)
        and ("kivilcim" in text_lower or "ark" in text_lower)
        and "yangin" in text_lower
    ):
        boost = max(boost, 0.64)

    # Calm phrase should not suppress severe physical risk.
    if "her sey yolunda" in text_lower and (
        "takla atti" in text_lower or "takla ati" in text_lower or "kan kaybettim" in text_lower
    ):
        boost = max(boost, 0.80)

    return boost


def compute_hard_risk_floor(text: str) -> float:
    text_lower = _collapse_elongations(_normalize_match_text(text))
    floor = 0.0

    if any(phrase in text_lower for phrase in HARD_FLOOR_080_PHRASES):
        floor = max(floor, 0.80)

    # Security-critical suspicious package scenario.
    if (
        any(term in text_lower for term in SUSPICIOUS_PACKAGE_TERMS)
        or (
            ("sahipsiz" in text_lower or "supheli" in text_lower)
            and "paket" in text_lower
            and ("ses" in text_lower or "bina" in text_lower or "giris" in text_lower)
        )
    ):
        floor = max(floor, 0.90)

    # Domestic violence / child safety anomaly scenario.
    if (
        "cocuk" in text_lower
        and ("aglama" in text_lower or "agliyor" in text_lower)
        and "kavga" in text_lower
        and "kapi" in text_lower
        and ("acan yok" in text_lower or "acilmiyor" in text_lower)
    ):
        floor = max(floor, 0.68)

    # Electrical ignition risk scenario.
    if (
        ("elektrik" in text_lower or "teller" in text_lower or "priz" in text_lower)
        and ("kivilcim" in text_lower or "ark" in text_lower or "kisa devre" in text_lower)
        and ("yangin" in text_lower or "alev" in text_lower or "yanik" in text_lower)
    ):
        floor = max(floor, 0.66)

    return floor


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
    return feats


# ─── V3.0 Helper Functions for panic_risk_score calculation ───────────────────

def compute_text_panic_score(feats: Dict[str, float]) -> float:
    """
    Compute text-based panic score from text features (V3.0).
    Optimized for STT + direct text input (no timing features).
    Formula: weighted sum of 7 linguistic features [0, 1].
    """
    weights = {
        "panic_word_ratio": 0.16,                    # panic keywords (weighted)
        "emergency_phrase_count": 0.90,              # 3x multiplier
        "repetition_ratio": 0.08,                    # word repetition
        "negation_adjusted_panic_ratio": 0.08,       # panic with negation context
        "ner_risk_density": 0.60,                    # 3x multiplier
        "sentence_entropy_inverse": 0.10,            # lexical diversity inverse
    }
    # Base weighted score: 0.92
    
    panic_score = 0.0
    
    for feature, weight in weights.items():
        if feature not in feats:
            continue
        
        val = feats.get(feature, 0.0)
        
        # Normalize based on feature type
        if feature in ["panic_word_ratio", "repetition_ratio",
                       "negation_adjusted_panic_ratio", "ner_risk_density", "sentence_entropy_inverse"]:
            normalized = min(max(val, 0.0), 1.0)  # Clamp [0,1]
        elif feature == "emergency_phrase_count":
            normalized = min(val / 3.0, 1.0)  # Threshold: 3 phrases
        else:
            normalized = 0.0
        
        panic_score += weight * normalized

    # Vowel lengthening contributes as a direct bonus to base score for panic writing style.
    vowel_val = min(max(float(feats.get("vowel_lengthening_ratio", 0.0)), 0.0), 1.0)
    if vowel_val > 0.0:
        vowel_bonus = min(0.06 + 0.12 * vowel_val, 0.18)
        panic_score += vowel_bonus
    
    return min(max(panic_score, 0.0), 1.0)  # Clamp [0, 1]




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
        candidate_triplets = [
            (
                model_dir / "panic_risk_model_tr_v3.joblib",
                model_dir / "panic_risk_scaler_tr_v3.joblib",
                model_dir / "panic_risk_model_meta_tr_v3.json",
            ),
            (
                model_dir / "sentiment_voting_model_text_only.joblib",
                model_dir / "sentiment_scaler_text_only.joblib",
                model_dir / "sentiment_model_meta_text_only.json",
            ),
            (
                model_dir / "sentiment_voting_model.joblib",
                model_dir / "sentiment_scaler.joblib",
                model_dir / "sentiment_model_meta.json",
            ),
        ]

        model_path = None
        scaler_path = None
        meta_path = None
        for candidate_model, candidate_scaler, candidate_meta in candidate_triplets:
            if candidate_model.exists() and candidate_scaler.exists():
                model_path = candidate_model
                scaler_path = candidate_scaler
                meta_path = candidate_meta
                break

        if model_path is None or scaler_path is None:
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
            X = pd.DataFrame(
                [{f: feats.get(f, 0.0) for f in self.feature_order}],
                columns=self.feature_order,
            )
            X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
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

            # === V3.0: Compute panic_risk_score (70/30 hybrid: text-only) ===
            # Text panic score from linguistic features
            text_panic_score = compute_text_panic_score(feats)
            
            # 70/30 hybrid formula with direct phrase boost for short Turkish emergency text.
            panic_risk_score = 0.70 * text_panic_score + 0.30 * (risk_score / 100.0)
            phrase_boost = compute_phrase_boost(text_en)
            panic_risk_score = max(panic_risk_score, phrase_boost)
            hard_floor = compute_hard_risk_floor(text_en)
            panic_risk_score = max(panic_risk_score, hard_floor)
            panic_risk_score = max(0.0, min(1.0, panic_risk_score))  # Clamp [0, 1]

            return {
                "panic_risk_score": panic_risk_score,
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
