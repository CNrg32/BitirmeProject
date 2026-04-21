#!/usr/bin/env python3
"""
train_triage_xlmr.py
====================
XLM-RoBERTa base cok-basli triage siniflandirici egitimi.

Girdi: output/out_dataset/triage_history_{train,val,test}.parquet
Cikti: out_models/triage_xlmr/ (model.pt + config.json + tokenizer/)

Davranis odakli tasarim:
  - CRITICAL class boost *YOK* (inverse-frequency agirliklar + label smoothing).
  - Erken durdurma: val macro F1(triage) + recall(CRITICAL).
  - Multi-task loss: 1.0*triage + 0.5*category + 0.3*redflag.
  - WeightedRandomSampler ile sinif dengesi saglanir.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from services.triage_model import (  # noqa: E402
    CATEGORY_LABELS,
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_LEN,
    QUALITY_LABELS,
    TRIAGE_LABELS,
    XlmrMultiTaskTriage,
)

DATA_DIR = _PROJECT_ROOT / "output" / "out_dataset"
TRAIN_PATH = DATA_DIR / "triage_history_train.parquet"
VAL_PATH = DATA_DIR / "triage_history_val.parquet"
TEST_PATH = DATA_DIR / "triage_history_test.parquet"

OUT_DIR = _PROJECT_ROOT / "out_models" / "triage_xlmr"

TRIAGE2ID = {l: i for i, l in enumerate(TRIAGE_LABELS)}
ID2TRIAGE = {i: l for l, i in TRIAGE2ID.items()}
CAT2ID = {l: i for i, l in enumerate(CATEGORY_LABELS)}
ID2CAT = {i: l for l, i in CAT2ID.items()}
QUAL2ID = {l: i for i, l in enumerate(QUALITY_LABELS)}
ID2QUAL = {i: l for l, i in QUAL2ID.items()}


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TriageHistoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int) -> None:
        self.texts = df["text"].astype(str).tolist()
        self.triage = df["triage_level"].map(TRIAGE2ID).astype(int).tolist()
        self.category = df["category"].map(CAT2ID).astype(int).tolist()
        self.redflag = df["red_flag_present"].astype(int).tolist()
        if "input_quality" in df.columns:
            qseries = df["input_quality"].fillna("meaningful").astype(str)
            qseries = qseries.where(qseries.isin(QUALITY_LABELS), "meaningful")
            self.quality = qseries.map(QUAL2ID).astype(int).tolist()
        else:
            self.quality = [QUAL2ID["meaningful"]] * len(self.texts)
        self.weights = df["sample_weight"].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "triage_label": torch.tensor(self.triage[idx], dtype=torch.long),
            "category_label": torch.tensor(self.category[idx], dtype=torch.long),
            "redflag_label": torch.tensor(self.redflag[idx], dtype=torch.float32),
            "quality_label": torch.tensor(self.quality[idx], dtype=torch.long),
        }


def _inverse_freq_weights(labels: list[int], n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes)
    total = float(len(labels))
    return torch.tensor(total / (n_classes * counts.clip(min=1)), dtype=torch.float32)


def _sampler_weights(df: pd.DataFrame) -> torch.Tensor:
    triage_ids = df["triage_level"].map(TRIAGE2ID).astype(int).to_numpy()
    class_w = _inverse_freq_weights(triage_ids.tolist(), len(TRIAGE_LABELS)).numpy()
    per_sample = class_w[triage_ids]
    ds_w = df["sample_weight"].astype(float).to_numpy()
    out = per_sample * ds_w
    return torch.tensor(out, dtype=torch.float32)


def _evaluate(model: XlmrMultiTaskTriage, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true_t, y_pred_t = [], []
    y_true_c, y_pred_c = [], []
    y_true_r, y_pred_r = [], []
    y_true_q, y_pred_q = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            y_pred_t.extend(torch.argmax(out["logits_triage"], dim=-1).cpu().tolist())
            y_true_t.extend(batch["triage_label"].tolist())
            y_pred_c.extend(torch.argmax(out["logits_category"], dim=-1).cpu().tolist())
            y_true_c.extend(batch["category_label"].tolist())
            y_pred_r.extend((torch.sigmoid(out["logits_redflag"]) >= 0.5).long().cpu().tolist())
            y_true_r.extend(batch["redflag_label"].long().tolist())
            y_pred_q.extend(torch.argmax(out["logits_quality"], dim=-1).cpu().tolist())
            y_true_q.extend(batch["quality_label"].tolist())
    triage_f1 = f1_score(y_true_t, y_pred_t, average="macro")
    crit_recall = recall_score(
        [ID2TRIAGE[i] for i in y_true_t],
        [ID2TRIAGE[i] for i in y_pred_t],
        labels=["CRITICAL"], average=None,
    )[0]
    cat_f1 = f1_score(y_true_c, y_pred_c, average="macro")
    rf_f1 = f1_score(y_true_r, y_pred_r, average="binary")
    qual_f1 = f1_score(y_true_q, y_pred_q, average="macro")
    return {
        "triage_f1": float(triage_f1),
        "crit_recall": float(crit_recall),
        "cat_f1": float(cat_f1),
        "rf_f1": float(rf_f1),
        "qual_f1": float(qual_f1),
        "y_true_t": y_true_t, "y_pred_t": y_pred_t,
        "y_true_c": y_true_c, "y_pred_c": y_pred_c,
        "y_true_r": y_true_r, "y_pred_r": y_pred_r,
        "y_true_q": y_true_q, "y_pred_q": y_pred_q,
    }


def _print_full_report(eval_out: dict, split_name: str) -> None:
    print(f"\n{'='*60}\n  {split_name} REPORT\n{'='*60}")
    y_true_s = [ID2TRIAGE[i] for i in eval_out["y_true_t"]]
    y_pred_s = [ID2TRIAGE[i] for i in eval_out["y_pred_t"]]
    print("TRIAGE:")
    print(classification_report(y_true_s, y_pred_s, labels=TRIAGE_LABELS, digits=3))
    print(confusion_matrix(y_true_s, y_pred_s, labels=TRIAGE_LABELS))

    y_true_c = [ID2CAT[i] for i in eval_out["y_true_c"]]
    y_pred_c = [ID2CAT[i] for i in eval_out["y_pred_c"]]
    print("\nCATEGORY:")
    print(classification_report(y_true_c, y_pred_c, labels=CATEGORY_LABELS, digits=3))

    print("\nRED_FLAG:")
    print(classification_report(eval_out["y_true_r"], eval_out["y_pred_r"], digits=3))

    y_true_q = [QUALITY_LABELS[i] for i in eval_out["y_true_q"]]
    y_pred_q = [QUALITY_LABELS[i] for i in eval_out["y_pred_q"]]
    print("\nINPUT_QUALITY:")
    print(classification_report(y_true_q, y_pred_q, labels=QUALITY_LABELS, digits=3, zero_division=0))

    print(f"\n** macro F1(triage)={eval_out['triage_f1']:.4f}  CRITICAL recall={eval_out['crit_recall']:.4f} "
          f"macro F1(cat)={eval_out['cat_f1']:.4f}  F1(rf)={eval_out['rf_f1']:.4f}  "
          f"macro F1(quality)={eval_out['qual_f1']:.4f} **")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--alpha-triage", type=float, default=1.0)
    p.add_argument("--alpha-category", type=float, default=0.5)
    p.add_argument("--alpha-redflag", type=float, default=0.3)
    p.add_argument("--alpha-quality", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--limit-train", type=int, default=0,
                   help="(debug) sadece N ornek ile egit, 0=tum.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = _pick_device()
    logger.info("Device: %s", device)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if args.limit_train > 0:
        train_df = train_df.sample(n=min(args.limit_train, len(train_df)), random_state=42).reset_index(drop=True)

    logger.info("Train=%d Val=%d Test=%d", len(train_df), len(val_df), len(test_df))

    train_ds = TriageHistoryDataset(train_df, tokenizer, args.max_len)
    val_ds = TriageHistoryDataset(val_df, tokenizer, args.max_len)
    test_ds = TriageHistoryDataset(test_df, tokenizer, args.max_len)

    sample_w = _sampler_weights(train_df)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = XlmrMultiTaskTriage(base_model_name=args.base_model).to(device)

    triage_class_w = _inverse_freq_weights(
        train_df["triage_level"].map(TRIAGE2ID).astype(int).tolist(),
        len(TRIAGE_LABELS),
    ).to(device)
    cat_class_w = _inverse_freq_weights(
        train_df["category"].map(CAT2ID).astype(int).tolist(),
        len(CATEGORY_LABELS),
    ).to(device)
    rf_pos_weight = torch.tensor(
        [max(1.0, (train_df["red_flag_present"] == 0).sum() /
             max(1, (train_df["red_flag_present"] == 1).sum()))],
        dtype=torch.float32, device=device,
    )
    if "input_quality" in train_df.columns:
        q_ids = (train_df["input_quality"].fillna("meaningful").astype(str)
                 .where(lambda s: s.isin(QUALITY_LABELS), "meaningful")
                 .map(QUAL2ID).astype(int).tolist())
        qual_class_w = _inverse_freq_weights(q_ids, len(QUALITY_LABELS)).to(device)
    else:
        qual_class_w = torch.ones(len(QUALITY_LABELS), device=device)

    loss_triage = nn.CrossEntropyLoss(weight=triage_class_w, label_smoothing=args.label_smoothing)
    loss_category = nn.CrossEntropyLoss(weight=cat_class_w, label_smoothing=args.label_smoothing)
    loss_redflag = nn.BCEWithLogitsLoss(pos_weight=rf_pos_weight)
    loss_quality = nn.CrossEntropyLoss(weight=qual_class_w, label_smoothing=args.label_smoothing)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    patience_ctr = 0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            l_t = loss_triage(out["logits_triage"], batch["triage_label"].to(device))
            l_c = loss_category(out["logits_category"], batch["category_label"].to(device))
            l_r = loss_redflag(out["logits_redflag"], batch["redflag_label"].to(device))
            l_q = loss_quality(out["logits_quality"], batch["quality_label"].to(device))
            loss = (args.alpha_triage * l_t + args.alpha_category * l_c
                    + args.alpha_redflag * l_r + args.alpha_quality * l_q)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()

            running += loss.item()
            n_batches += 1
            if (step + 1) % 50 == 0:
                logger.info(
                    "Epoch %d/%d Step %d/%d loss=%.4f (t=%.3f c=%.3f rf=%.3f q=%.3f)",
                    epoch + 1, args.epochs, step + 1, len(train_loader),
                    loss.item(), l_t.item(), l_c.item(), l_r.item(), l_q.item(),
                )

        avg_loss = running / max(1, n_batches)
        val_out = _evaluate(model, val_loader, device)
        _print_full_report(val_out, f"EPOCH {epoch+1} VAL")
        # Track combined score: 0.6 * triage macro F1 + 0.4 * CRITICAL recall
        score = 0.6 * val_out["triage_f1"] + 0.4 * val_out["crit_recall"]
        logger.info(
            "Epoch %d: train_loss=%.4f score=%.4f triage_f1=%.4f crit_recall=%.4f",
            epoch + 1, avg_loss, score, val_out["triage_f1"], val_out["crit_recall"],
        )

        if score > best_score:
            best_score = score
            patience_ctr = 0
            logger.info("New best score %.4f, saving...", score)
            model.save(
                OUT_DIR, tokenizer=tokenizer,
                extra_meta={
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "best_val_score": score,
                    "best_val_triage_f1": val_out["triage_f1"],
                    "best_val_crit_recall": val_out["crit_recall"],
                },
            )
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Reloading best model for TEST ...")
    model = XlmrMultiTaskTriage.load(OUT_DIR, device=device)
    test_out = _evaluate(model, test_loader, device)
    _print_full_report(test_out, "FINAL TEST")


if __name__ == "__main__":
    main()
