# DistilBERT fine-tune ile triaj seviyesi egitimi (CRITICAL / URGENT / NON_URGENT)
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "output" / "out_dataset"
TRAIN_PATH = DATA_DIR / "triage_dataset_train.parquet"
VAL_PATH = DATA_DIR / "triage_dataset_val.parquet"
TEST_PATH = DATA_DIR / "triage_dataset_test.parquet"

OUT_MODELS = _PROJECT_ROOT / "out_models"
BERT_DIR = OUT_MODELS / "triage_distilbert"
OUT_MODELS.mkdir(parents=True, exist_ok=True)

TRIAGE_LABELS = ["CRITICAL", "URGENT", "NON_URGENT"]
LABEL2ID = {l: i for i, l in enumerate(TRIAGE_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(TRIAGE_LABELS)}

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 8
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
PATIENCE = 3
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

logger.info("Using device: %s", DEVICE)


class TriageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["text_en"] = df["text_en"].astype(str).fillna("").str.strip()
    df = df[df["text_en"].str.len() > 0].copy()
    df = df[df["label_triage_gold"].isin(TRIAGE_LABELS)].copy()
    return df


def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=len(TRIAGE_LABELS))
    total = len(labels)
    weights = total / (len(TRIAGE_LABELS) * counts.astype(float) + 1e-6)
    weights[LABEL2ID["CRITICAL"]] *= 1.5
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, dataloader, device, class_names=TRIAGE_LABELS):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            n_batches += 1

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    return np.array(all_preds), np.array(all_labels), avg_loss


def print_report(y_true, y_pred, split_name=""):
    labels_str = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    y_true_str = [ID2LABEL[i] for i in y_true]
    y_pred_str = [ID2LABEL[i] for i in y_pred]

    print(f"\n{'='*60}")
    print(f"  {split_name} REPORT")
    print(f"{'='*60}")
    print(classification_report(y_true_str, y_pred_str, labels=labels_str, digits=3))

    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels_str)
    print("Confusion matrix (rows=true, cols=pred)")
    print(f"Labels: {labels_str}")
    print(cm)

    crit_recall = recall_score(y_true_str, y_pred_str, labels=["CRITICAL"], average=None)[0]
    print(f"\n** CRITICAL recall: {crit_recall:.4f} **")
    if crit_recall < 0.95:
        print("   ⚠ WARNING: CRITICAL recall below 95% safety threshold!")
    return crit_recall


def main():
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    from transformers import get_linear_schedule_with_warmup

    logger.info("Loading data...")
    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)

    logger.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

    train_labels = train_df["label_triage_gold"].map(LABEL2ID).values
    val_labels = val_df["label_triage_gold"].map(LABEL2ID).values
    test_labels = test_df["label_triage_gold"].map(LABEL2ID).values

    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TriageDataset(train_df["text_en"].values, train_labels, tokenizer)
    val_dataset = TriageDataset(val_df["text_en"].values, val_labels, tokenizer)
    test_dataset = TriageDataset(test_df["text_en"].values, test_labels, tokenizer)

    sample_weights_per_class = compute_class_weights(train_labels)
    sample_weights = sample_weights_per_class[train_labels]
    if "sample_weight" in train_df.columns:
        ds_weights = train_df["sample_weight"].fillna(1.0).values.astype(float)
        sample_weights = sample_weights.numpy() * ds_weights
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Loading model: %s", MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TRIAGE_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(DEVICE)

    class_weights = compute_class_weights(train_labels).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_crit_recall = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        n_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            n_train_batches += 1

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    "Epoch %d/%d, Batch %d/%d, Loss: %.4f",
                    epoch + 1, EPOCHS, batch_idx + 1, len(train_loader),
                    loss.item(),
                )

        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        val_preds, val_true, val_loss = evaluate(model, val_loader, DEVICE)
        crit_recall = print_report(val_true, val_preds, f"EPOCH {epoch+1} VAL")
        logger.info(
            "Epoch %d: train_loss=%.4f, val_loss=%.4f, CRITICAL_recall=%.4f",
            epoch + 1, avg_train_loss, val_loss, crit_recall,
        )

        if crit_recall > best_crit_recall:
            best_crit_recall = crit_recall
            patience_counter = 0
            logger.info("New best CRITICAL recall: %.4f – saving model...", crit_recall)
            model.save_pretrained(BERT_DIR)
            tokenizer.save_pretrained(BERT_DIR)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch + 1, PATIENCE)
                break

    logger.info("Loading best model for test evaluation...")
    model = DistilBertForSequenceClassification.from_pretrained(BERT_DIR)
    model.to(DEVICE)

    test_preds, test_true, test_loss = evaluate(model, test_loader, DEVICE)
    crit_recall_test = print_report(test_true, test_preds, "FINAL TEST")
    logger.info("Test loss: %.4f, Test CRITICAL recall: %.4f", test_loss, crit_recall_test)

    label_maps = {
        "triage_labels": TRIAGE_LABELS,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "model_type": "distilbert",
        "model_path": str(BERT_DIR),
    }
    with open(OUT_MODELS / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)

    wrapper = DistilBertTriageWrapper(str(BERT_DIR), TRIAGE_LABELS, LABEL2ID, ID2LABEL)
    joblib.dump(wrapper, OUT_MODELS / "triage_pipeline.joblib")

    logger.info("All artifacts saved to %s", OUT_MODELS)
    logger.info("Done! Best CRITICAL recall (val): %.4f, Test CRITICAL recall: %.4f",
                best_crit_recall, crit_recall_test)

    logger.info("\n--- TF-IDF baseline karsilastirma ---")
    _run_baseline_comparison()


class DistilBertTriageWrapper:
    """sklearn pipeline arayuzunu taklit eden wrapper (API uyumlulugu icin)."""
    def __init__(self, model_path, labels, label2id, id2label):
        self.model_path = model_path
        self.labels = labels
        self.label2id = label2id
        self.id2label = id2label
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            self._tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self._model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
            self._model.to(self._device)
            self._model.eval()

    @property
    def classes_(self):
        return np.array(self.labels)

    def predict(self, texts):
        proba = self.predict_proba(texts)
        return [self.labels[i] for i in np.argmax(proba, axis=1)]

    def predict_proba(self, texts):
        self._ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        encodings = self._tokenizer(
            list(texts),
            max_length=MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self._device)
        attention_mask = encodings["attention_mask"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            proba = torch.softmax(outputs.logits, dim=-1)
        return proba.cpu().numpy()


def _run_baseline_comparison():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    train = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)

    X_train = train["text_en"]
    y_train = train["label_triage_gold"]
    X_test = test["text_en"]
    y_test = test["label_triage_gold"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n{'='*60}")
    print(f"  TF-IDF + LR BASELINE (for comparison)")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, labels=TRIAGE_LABELS, digits=3))
    crit_recall = recall_score(y_test, y_pred, labels=["CRITICAL"], average=None)[0]
    print(f"** Baseline CRITICAL recall: {crit_recall:.4f} **")


if __name__ == "__main__":
    main()
