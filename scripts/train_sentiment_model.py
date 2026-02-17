# Duygu analizi + aciliyet modeli egitimi (VotingClassifier: RF + GB + LR)
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "labels"
AUDIO_DIR = _PROJECT_ROOT / "data" / "raw" / "911_recordings" / "audio"
OUTPUT_DIR = _PROJECT_ROOT / "output" / "out_dataset"
MODEL_DIR = _PROJECT_ROOT / "out_models"
ANALYSIS_DIR = _PROJECT_ROOT / "output" / "Sentiment Analysis Outputs"

CASES_CSV = DATA_DIR / "911_cases_v1.csv"
TRANSCRIPTS_CSV = DATA_DIR / "auto_transcripts.csv"

TRIAGE_LABELS = ["CRITICAL", "URGENT", "NON_URGENT"]
RANDOM_SEED = 42

PANIC_KEYWORDS = [
    "help", "fuck", "shit", "blood", "dying", "gun", "please",
    "urgent", "kill", "fire", "can't breathe", "stabbed", "shot",
    "ambulance", "dead", "hurry", "emergency", "weapon", "scream",
    "attack", "assault", "overdose", "unconscious", "trapped",
]


def extract_text_features(text: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    panic_count = sum(1 for w in words if any(k in w for k in PANIC_KEYWORDS))
    feats["panic_word_ratio"] = panic_count / word_count if word_count > 0 else 0
    feats["panic_word_count"] = panic_count

    feats["word_count"] = word_count
    feats["avg_word_length"] = (
        np.mean([len(w) for w in words]) if words else 0
    )

    sentences = [
        s.strip()
        for s in text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    feats["sentence_count"] = len(sentences)

    feats["exclamation_ratio"] = (
        text.count("!") / word_count if word_count > 0 else 0
    )
    feats["question_ratio"] = (
        text.count("?") / word_count if word_count > 0 else 0
    )

    alpha_chars = [c for c in text if c.isalpha()]
    feats["uppercase_ratio"] = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars
        else 0
    )

    if word_count > 1:
        unique_words = set(words)
        feats["repetition_ratio"] = 1 - (len(unique_words) / word_count)
    else:
        feats["repetition_ratio"] = 0

    emergency_phrases = [
        "need help", "send help", "hurry up", "right now", "come quick",
        "shots fired", "officer down", "code 3", "respond to",
        "not breathing", "no pulse", "unconscious", "bleeding out",
    ]
    feats["emergency_phrase_count"] = sum(
        1 for p in emergency_phrases if p in text_lower
    )

    return feats


def extract_audio_features(file_path: Path) -> Dict[str, float]:
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
        return feats

    if not file_path.exists():
        return feats

    try:
        y, sr = librosa.load(str(file_path), duration=60, sr=None)
        feats["audio_found"] = True
        feats["audio_duration"] = librosa.get_duration(y=y, sr=sr)

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
        non_silent_dur = sum([x[1] - x[0] for x in intervals]) / sr
        total_dur = feats["audio_duration"]
        feats["silence_ratio"] = (
            1 - (non_silent_dur / total_dur) if total_dur > 0 else 0
        )
    except Exception:
        pass

    return feats


def get_sentiment_score(text: str) -> float:
    """Sentiment skoru: -1 (panik) ile +1 (sakin) arasi. HuggingFace yoksa keyword-based."""
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


def load_and_prepare_data() -> pd.DataFrame:
    if not CASES_CSV.exists():
        raise FileNotFoundError(f"Data file not found: {CASES_CSV}")

    df_cases = pd.read_csv(CASES_CSV)
    print(f"Loaded {len(df_cases)} rows from {CASES_CSV.name}")

    # Triage label kolonu sec (gold > weak)
    if "label_triage_gold" in df_cases.columns:
        df_cases["triage_label"] = df_cases["label_triage_gold"].fillna(
            df_cases.get("label_triage_weak", pd.Series(dtype=str))
        )
        print("Label source: label_triage_gold (fallback: label_triage_weak)")
    elif "label_triage_weak" in df_cases.columns:
        df_cases["triage_label"] = df_cases["label_triage_weak"]
        print("Label source: label_triage_weak")
    else:
        raise ValueError("No triage label column found in CSV!")

    # Metin kolonu bul
    text_col = None
    for col in ["transcript_en", "text_clean", "text_raw"]:
        if col in df_cases.columns:
            non_empty = df_cases[col].dropna().str.strip().str.len().gt(0).sum()
            if non_empty > 0:
                text_col = col
                print(f"Text source: {col} ({non_empty} non-empty rows)")
                break

    if text_col is None and TRANSCRIPTS_CSV.exists():
        print("No text column found, merging with auto_transcripts.csv...")
        df_text = pd.read_csv(TRANSCRIPTS_CSV)
        if "path" in df_text.columns:
            df_text["file_name"] = df_text["path"].apply(
                lambda x: Path(str(x)).name
            )
        df_cases = pd.merge(
            df_cases, df_text[["file_name", "text"]], on="file_name", how="left"
        )
        text_col = "text"

    if text_col is None:
        raise ValueError("No text column found!")

    df_cases["transcript"] = df_cases[text_col].astype(str).fillna("")

    df_valid = df_cases[df_cases["triage_label"].isin(TRIAGE_LABELS)].copy()
    df_valid = df_valid[df_valid["transcript"].str.strip().str.len() > 2].copy()

    print(f"Valid labeled rows with text: {len(df_valid)}")
    print(f"Label distribution:\n{df_valid['triage_label'].value_counts()}")

    return df_valid


def build_feature_dataset(df_valid: pd.DataFrame) -> pd.DataFrame:
    print("\nFeature extraction starting...")
    start_time = time.time()

    training_data: List[Dict] = []
    total = len(df_valid)

    for idx, (_, row) in enumerate(df_valid.iterrows()):
        transcript = str(row["transcript"])

        feats = extract_text_features(transcript)

        feats["sentiment_score"] = get_sentiment_score(transcript)

        feats["estimated_wpm"] = (
            feats["word_count"] / max(feats["sentence_count"], 1) * 10
        )

        # Audio features (optional)
        file_name = str(row.get("file_name", ""))
        audio_path = AUDIO_DIR / file_name
        audio_feats = extract_audio_features(audio_path)
        feats.update(audio_feats)

        # WPM: use real audio duration if available
        if audio_feats["audio_found"] and audio_feats["audio_duration"] > 0:
            feats["wpm"] = (
                feats["word_count"] / audio_feats["audio_duration"]
            ) * 60
        else:
            feats["wpm"] = feats["estimated_wpm"]

        # Metadata features
        for meta_col in [
            "deaths", "potential_death", "false_alarm", "civilian_initiated",
        ]:
            val = row.get(meta_col, 0)
            try:
                feats[meta_col] = float(val) if pd.notna(val) else 0
            except (ValueError, TypeError):
                feats[meta_col] = 0

        feats["target_label"] = row["triage_label"]
        feats["file_name"] = file_name
        feats["transcript"] = transcript
        training_data.append(feats)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            elapsed = time.time() - start_time
            print(f"  {idx + 1}/{total} completed ({elapsed:.1f}s)")

    df_features = pd.DataFrame(training_data)
    df_features = df_features.fillna(0)
    df_features = df_features.replace([np.inf, -np.inf], 0)

    # Save extracted features
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = OUTPUT_DIR / "sentiment_training_dataset.csv"
    df_features.to_csv(dataset_path, index=False)
    print(f"\nFeature dataset saved: {dataset_path}")
    print(f"Total samples: {len(df_features)}")
    print(
        f"Audio files found: {df_features['audio_found'].sum()}"
        if "audio_found" in df_features.columns
        else ""
    )

    return df_features


# ─── Training ────────────────────────────────────────────────────────────────
# Feature groups
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


def train_models(df_features: pd.DataFrame) -> Dict:
    """Train RF, GB, LR, and VotingClassifier; return best model + metadata."""
    print("\nModel training starting...\n")

    # Select available features
    all_features = TEXT_FEATURES + AUDIO_FEATURES + META_FEATURES
    available_features = [f for f in all_features if f in df_features.columns]

    # Drop audio features if no audio files found
    if df_features.get("audio_found", pd.Series([False])).sum() == 0:
        available_features = [
            f for f in available_features if f not in AUDIO_FEATURES
        ]
        print("No audio files found - using text + metadata features only.")

    print(f"Features ({len(available_features)}): {available_features}\n")

    X = df_features[available_features]
    y = df_features["target_label"]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Label distribution (train): {dict(y_train.value_counts())}")

    # --- Model 1: Random Forest (Balanced) ---
    print("\n--- Model 1: Random Forest (Balanced) ---")
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf_clf.fit(X_train_scaled, y_train)
    rf_pred = rf_clf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(classification_report(y_test, rf_pred, labels=TRIAGE_LABELS))

    # --- Model 2: Gradient Boosting ---
    print("--- Model 2: Gradient Boosting ---")
    gb_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_SEED,
    )
    gb_clf.fit(X_train_scaled, y_train)
    gb_pred = gb_clf.predict(X_test_scaled)
    gb_acc = accuracy_score(y_test, gb_pred)
    print(f"Gradient Boosting Accuracy: {gb_acc*100:.2f}%")
    print(classification_report(y_test, gb_pred, labels=TRIAGE_LABELS))

    # --- Model 3: Logistic Regression ---
    print("--- Model 3: Logistic Regression ---")
    lr_clf = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    lr_clf.fit(X_train_scaled, y_train)
    lr_pred = lr_clf.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")
    print(classification_report(y_test, lr_pred, labels=TRIAGE_LABELS))

    # --- Voting Classifier (Ensemble) ---
    print("\n=== Voting Classifier (Ensemble) ===")
    voting_clf = VotingClassifier(
        estimators=[("rf", rf_clf), ("gb", gb_clf), ("lr", lr_clf)],
        voting="soft",
    )
    voting_clf.fit(X_train_scaled, y_train)
    voting_pred = voting_clf.predict(X_test_scaled)
    voting_acc = accuracy_score(y_test, voting_pred)
    critical_recall = recall_score(
        y_test, voting_pred, labels=["CRITICAL"], average=None,
    )[0]

    print(f"Voting Classifier Accuracy: {voting_acc*100:.2f}%")
    print(f"CRITICAL Recall: {critical_recall*100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, voting_pred, labels=TRIAGE_LABELS))

    # Best model selection
    model_scores = {
        "RandomForest": rf_acc,
        "GradientBoosting": gb_acc,
        "LogisticRegression": lr_acc,
        "VotingClassifier": voting_acc,
    }
    best_name = max(model_scores, key=model_scores.get)
    print(f"\nBest model: {best_name} ({model_scores[best_name]*100:.2f}%)")

    # --- Save ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "sentiment_voting_model.joblib"
    scaler_path = MODEL_DIR / "sentiment_scaler.joblib"
    meta_path = MODEL_DIR / "sentiment_model_meta.json"

    joblib.dump(voting_clf, model_path)
    joblib.dump(scaler, scaler_path)

    model_meta = {
        "model_type": "VotingClassifier (RF + GB + LR)",
        "features": available_features,
        "triage_labels": TRIAGE_LABELS,
        "accuracy": round(voting_acc, 4),
        "critical_recall": round(critical_recall, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "all_model_scores": {k: round(v, 4) for k, v in model_scores.items()},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2, ensure_ascii=False)

    print(f"\nModel saved    : {model_path}")
    print(f"Scaler saved   : {scaler_path}")
    print(f"Metadata saved : {meta_path}")

    # --- Visualization (optional, requires matplotlib/seaborn) ---
    _save_visualizations(
        y_test, voting_pred, voting_acc, model_scores,
        rf_clf, available_features, df_features,
    )

    return model_meta


def _save_visualizations(
    y_test, voting_pred, voting_acc, model_scores,
    rf_clf, available_features, df_features,
):
    """Generate and save performance plots (fails silently if matplotlib unavailable)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available - skipping visualization.")
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Sentiment & Urgency Model Performance", fontsize=14, fontweight="bold",
    )

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, voting_pred, labels=TRIAGE_LABELS)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TRIAGE_LABELS, yticklabels=TRIAGE_LABELS, ax=axes[0, 0],
    )
    axes[0, 0].set_title(f"Voting Classifier (Acc: {voting_acc*100:.1f}%)")
    axes[0, 0].set_ylabel("True")
    axes[0, 0].set_xlabel("Predicted")

    # 2. Model comparison
    models = list(model_scores.keys())
    scores = [v * 100 for v in model_scores.values()]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    bars = axes[0, 1].barh(models, scores, color=colors)
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].set_title("Model Accuracy Comparison")
    axes[0, 1].set_xlabel("Accuracy (%)")
    for bar, score in zip(bars, scores):
        axes[0, 1].text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%", va="center", fontweight="bold",
        )

    # 3. Feature importance (Random Forest)
    importances = rf_clf.feature_importances_
    feat_imp = pd.Series(importances, index=available_features).sort_values(
        ascending=True,
    )
    feat_imp.plot(kind="barh", ax=axes[1, 0], color="#607D8B")
    axes[1, 0].set_title("Feature Importance (RF)")
    axes[1, 0].set_xlabel("Importance")

    # 4. Label distribution
    label_counts = df_features["target_label"].value_counts()
    label_colors = {
        "CRITICAL": "#F44336", "URGENT": "#FF9800", "NON_URGENT": "#4CAF50",
    }
    pie_colors = [label_colors.get(l, "#999") for l in label_counts.index]
    axes[1, 1].pie(
        label_counts.values, labels=label_counts.index, autopct="%1.1f%%",
        colors=pie_colors, startangle=90,
    )
    axes[1, 1].set_title("Label Distribution")

    plt.tight_layout()
    fig_path = ANALYSIS_DIR / "model_performance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {fig_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    df_valid = load_and_prepare_data()
    df_features = build_feature_dataset(df_valid)
    train_models(df_features)


if __name__ == "__main__":
    main()
