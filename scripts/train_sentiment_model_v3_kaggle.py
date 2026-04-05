#!/usr/bin/env python3
"""
Sentiment Model Training - V3.0 (Text-Only, Turkish Final Dataset)
==================================================================
Egit: VotingClassifier (RF + GB + LR)
Veri: synthetic_triage_cases_final_tr.csv
Features: 15 text-based features (no audio, no timing)
Output: panic_risk_model_tr_v3.joblib + metadata
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "labels"
MODEL_DIR = _PROJECT_ROOT / "out_models"
TURKISH_TRAIN_CSV = DATA_DIR / "synthetic_triage_cases_final_tr.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRIAGE_LABELS = ["CRITICAL", "URGENT", "NON_URGENT"]
RANDOM_SEED = 42

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

FEATURE_ORDER = [
    "panic_word_ratio",
    "panic_word_count",
    "word_count",
    "avg_word_length",
    "sentence_count",
    "exclamation_ratio",
    "question_ratio",
    "uppercase_ratio",
    "repetition_ratio",
    "emergency_phrase_count",
    "sentiment_score",
    "vowel_lengthening_ratio",
    "negation_adjusted_panic_ratio",
    "ner_risk_density",
    "sentence_entropy_inverse",
]


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
    feats["avg_word_length"] = float(np.mean([len(w) for w in words])) if words else 0.0

    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    feats["sentence_count"] = float(len(sentences))
    feats["exclamation_ratio"] = text.count("!") / word_count if word_count > 0 else 0.0
    feats["question_ratio"] = text.count("?") / word_count if word_count > 0 else 0.0

    alpha_chars = [c for c in text if c.isalpha()]
    feats["uppercase_ratio"] = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0.0
    )

    if word_count > 1:
        feats["repetition_ratio"] = 1.0 - (len(set(words)) / word_count)
    else:
        feats["repetition_ratio"] = 0.0

    feats["emergency_phrase_count"] = float(sum(1 for p in NORMALIZED_EMERGENCY_PHRASES if p in collapsed_text))

    neg_count = sum(1 for k in NORMALIZED_PANIC_KEYWORDS if k in collapsed_text)
    pos_count = sum(1 for k in NORMALIZED_POSITIVE_WORDS if k in collapsed_text)
    total = neg_count + pos_count
    feats["sentiment_score"] = (pos_count - neg_count) / total if total > 0 else 0.0

    vowel_patterns = re.findall(r"([aeiou])\1{2,}", normalized_text)
    feats["vowel_lengthening_ratio"] = float(len(vowel_patterns) / max(word_count, 1))

    adjusted_panic_count = 0
    for i, token in enumerate(words):
        is_panic = any(k in token for k in NORMALIZED_PANIC_KEYWORDS)
        if is_panic:
            has_negation_nearby = False
            for j in range(max(0, i - 3), min(len(words), i + 4)):
                if words[j] in NORMALIZED_NEGATION_WORDS or any(neg in words[j] for neg in NORMALIZED_NEGATION_WORDS):
                    has_negation_nearby = True
                    break
            if not has_negation_nearby:
                adjusted_panic_count += 1
    feats["negation_adjusted_panic_ratio"] = float(adjusted_panic_count / max(word_count, 1))

    risk_count = sum(1 for e in NORMALIZED_RISK_ENTITIES if e in collapsed_text)
    ner_density = float(risk_count / max(word_count, 1))
    if any(term in collapsed_text for term in PHYSICAL_THREAT_TERMS):
        ner_density *= 1.5
    feats["ner_risk_density"] = float(min(max(ner_density, 0.0), 1.0))

    if sentences:
        entropies = []
        for sent in sentences:
            sent_words = sent.lower().split()
            if sent_words:
                entropies.append(float(len(set(sent_words)) / len(sent_words)))
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        feats["sentence_entropy_inverse"] = 1.0 - avg_entropy
    else:
        feats["sentence_entropy_inverse"] = 0.0

    return feats


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    if "text_en" not in df.columns and "text" in df.columns:
        df["text_en"] = df["text"]
    if "label_triage_gold" not in df.columns and "label_triage" in df.columns:
        df["label_triage_gold"] = df["label_triage"]

    df = df[df["label_triage_gold"].isin(TRIAGE_LABELS)].copy()
    df = df[df["text_en"].astype(str).str.strip().str.len() > 2].copy()

    print(f"Valid rows: {len(df)}")
    print("\nLabel distribution:")
    print(df["label_triage_gold"].value_counts())
    return df


def build_features(df: pd.DataFrame):
    print("\nExtracting features...")
    features_list = []
    for idx, row in df.iterrows():
        feats = extract_text_features(str(row["text_en"]).strip())
        features_list.append(feats)
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(df)} completed")

    X = pd.DataFrame(features_list)
    for feat in FEATURE_ORDER:
        if feat not in X.columns:
            X[feat] = 0.0
    X = X[FEATURE_ORDER]
    y = df["label_triage_gold"].astype(str).to_numpy(dtype=object)

    print(f"Features shape: {X.shape}")
    return X, y


def train_models(X: pd.DataFrame, y):
    print("\nTraining models...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    voting_clf = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
    )
    voting_clf.fit(X_train, y_train)

    y_pred = voting_clf.predict(X_test)
    y_pred_train = voting_clf.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    critical_recall = recall_score(y_test, y_pred, labels=["CRITICAL"], average="macro")
    cv_scores = cross_val_score(voting_clf, X_scaled, y, cv=5, scoring="accuracy")

    print(f"\nTrain Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred, target_names=TRIAGE_LABELS))
    print(f"CRITICAL Recall: {critical_recall:.4f}")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        "model": voting_clf,
        "scaler": scaler,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "critical_recall": critical_recall,
        "cv_scores": cv_scores,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def save_model(results: dict, output_dir: Path):
    model_path = output_dir / "panic_risk_model_tr_v3.joblib"
    scaler_path = output_dir / "panic_risk_scaler_tr_v3.joblib"
    meta_path = output_dir / "panic_risk_model_meta_tr_v3.json"

    joblib.dump(results["model"], model_path)
    joblib.dump(results["scaler"], scaler_path)

    meta = {
        "model_type": "VotingClassifier",
        "model_version": "v3.0-text-only-tr",
        "feature_pipeline_version": "feature-pipeline-v3.0",
        "dataset": TURKISH_TRAIN_CSV.name,
        "features": FEATURE_ORDER,
        "triage_labels": TRIAGE_LABELS,
        "train_accuracy": float(results["train_accuracy"]),
        "test_accuracy": float(results["test_accuracy"]),
        "critical_recall": float(results["critical_recall"]),
        "cv_mean_accuracy": float(results["cv_scores"].mean()),
        "cv_std_accuracy": float(results["cv_scores"].std()),
        "random_seed": RANDOM_SEED,
        "meta_schema_version": 3,
        "components": ["RandomForest", "GradientBoosting", "LogisticRegression"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Metadata saved: {meta_path}")


def main():
    print("=" * 80)
    print("SENTIMENT MODEL TRAINING V3.0 (Turkish Final Dataset)")
    print("=" * 80)
    df = load_data(TURKISH_TRAIN_CSV)
    X, y = build_features(df)
    results = train_models(X, y)
    save_model(results, MODEL_DIR)
    print("\nTraining complete.")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
