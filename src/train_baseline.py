
"""
Baseline training: TF-IDF + Logistic Regression

Trains:
  - medical_vs_nonmedical (binary) from label_category_gold
  - triage_level (3-class) from label_triage_gold (optionally only medical)

Saves:
  - out_models/triage_lr.joblib
  - out_models/tfidf.joblib
  - out_models/label_maps.json
Reports:
  - classification report + confusion matrix
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("out_dataset")
TRAIN_PATH = DATA_DIR / "triage_dataset_train.parquet"
VAL_PATH = DATA_DIR / "triage_dataset_val.parquet"
TEST_PATH = DATA_DIR / "triage_dataset_test.parquet"

OUT_MODELS = Path("out_models")
OUT_MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["text_en"] = df["text_en"].astype(str).fillna("").str.strip()
    df = df[df["text_en"].str.len() > 0].copy()
    return df

def print_cm(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=pred)")
    print("labels:", labels)
    print(cm)

def main():
    train = load_split(TRAIN_PATH)
    val = load_split(VAL_PATH)
    test = load_split(TEST_PATH)

    # TRIAGE 3-class
    triage_labels = ["CRITICAL","URGENT","NON_URGENT"]
    train_t = train[train["label_triage_gold"].isin(triage_labels)].copy()
    val_t = val[val["label_triage_gold"].isin(triage_labels)].copy()
    test_t = test[test["label_triage_gold"].isin(triage_labels)].copy()

    # Optional: train only on medical cases
    # Uncomment if you want:
    # train_t = train_t[train_t["label_category_gold"]=="medical"]
    # val_t = val_t[val_t["label_category_gold"]=="medical"]
    # test_t = test_t[test_t["label_category_gold"]=="medical"]

    X_train, y_train = train_t["text_en"], train_t["label_triage_gold"]
    X_val, y_val = val_t["text_en"], val_t["label_triage_gold"]
    X_test, y_test = test_t["text_en"], test_t["label_triage_gold"]

    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=60000,
        min_df=2,
        strip_accents="unicode",
        lowercase=True
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_SEED
    )

    model = Pipeline([("tfidf", tfidf), ("clf", clf)])
    model.fit(X_train, y_train)

    # Validate
    val_pred = model.predict(X_val)
    print("\n=== VAL REPORT (triage) ===")
    print(classification_report(y_val, val_pred, labels=triage_labels))
    print_cm(y_val, val_pred, triage_labels)

    # Test
    test_pred = model.predict(X_test)
    print("\n=== TEST REPORT (triage) ===")
    print(classification_report(y_test, test_pred, labels=triage_labels))
    print_cm(y_test, test_pred, triage_labels)

    # Save artifacts
    joblib.dump(model.named_steps["tfidf"], OUT_MODELS / "tfidf.joblib")
    joblib.dump(model.named_steps["clf"], OUT_MODELS / "triage_lr.joblib")
    with open(OUT_MODELS / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump({"triage_labels": triage_labels}, f, indent=2)

    print("\nSaved models to:", OUT_MODELS)

if __name__ == "__main__":
    main()
