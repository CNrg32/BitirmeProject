#!/usr/bin/env python3
"""
prepare_triage_history_dataset.py
=================================
Triage_history dataset olusturur — history-aware (progressive prefix) ornekler
+ JSONL dialog orneklerini tek bir parquet setine birlestirir.

Girdi:
    output/out_dataset/triage_dataset_{train,val,test}.parquet
    data/llm_finetune_train.jsonl
    data/llm_groq_lora_train.jsonl
    data/llm_groq_lora_val.jsonl

Cikti:
    output/out_dataset/triage_history_{train,val,test}.parquet
    Sutunlar:
        text                — [USER] .. [ASSISTANT] .. formatinda duz string
        triage_level        — CRITICAL / URGENT / NON_URGENT
        category            — medical / fire / crime / other
        red_flag_present    — 0/1
        sample_weight       — float
        source              — parquet_prefix / jsonl_dialog / ...
        split               — train / val / test

Parquet prefix'leme: her transkripti cumlelere bol, uzunluga gore 1-3 prefix
ornegi uret (~%50, %75, %100). Her prefix icin ayni gold etiket atanir; model
"sinyal netlesince commit et" davranisi ogreniyor (CRITICAL ornekler datasette
agir degil; conservative dataset NON_URGENT/URGENT counterweight sagliyor).

JSONL dialog: messages[0] user metnini al, messages[1] JSON'dan triage_level,
category, red_flags -> tek sample.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "output" / "out_dataset"
JSONL_FILES = [
    PROJECT_ROOT / "data" / "llm_finetune_train.jsonl",
    PROJECT_ROOT / "data" / "llm_groq_lora_train.jsonl",
    PROJECT_ROOT / "data" / "llm_groq_lora_val.jsonl",
]
MULTITURN_JSONL = PROJECT_ROOT / "data" / "labels" / "multiturn_dialogs_v1.jsonl"

OUT_TRAIN = DATA_DIR / "triage_history_train.parquet"
OUT_VAL = DATA_DIR / "triage_history_val.parquet"
OUT_TEST = DATA_DIR / "triage_history_test.parquet"

VALID_TRIAGE = {"CRITICAL", "URGENT", "NON_URGENT"}
VALID_CATEGORIES = {"medical", "fire", "crime", "other"}
VALID_QUALITIES = {"meaningful", "gibberish", "out_of_scope"}

PREFIX_FRACTIONS = [0.5, 0.75, 1.0]
MIN_SENTENCES_FOR_PREFIX = 3
JSONL_VAL_FRACTION = 0.1
JSONL_SAMPLE_WEIGHT = 1.5

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return parts or [text]


def _wrap_user(text: str) -> str:
    return f"[USER] {text.strip()}"


def _prefix_samples(text: str, label_triage: str, label_cat: str, rf: int,
                    weight: float, source: str, input_quality: str) -> list[dict[str, Any]]:
    sentences = _split_sentences(text)
    n = len(sentences)
    rows: list[dict[str, Any]] = []
    if n < MIN_SENTENCES_FOR_PREFIX:
        rows.append({
            "text": _wrap_user(" ".join(sentences)),
            "triage_level": label_triage,
            "category": label_cat,
            "red_flag_present": int(rf),
            "input_quality": input_quality,
            "sample_weight": float(weight),
            "source": source,
        })
        return rows
    for frac in PREFIX_FRACTIONS:
        k = max(1, int(round(frac * n)))
        prefix = " ".join(sentences[:k])
        rows.append({
            "text": _wrap_user(prefix),
            "triage_level": label_triage,
            "category": label_cat,
            "red_flag_present": int(rf),
            "input_quality": input_quality,
            "sample_weight": float(weight),
            "source": source,
        })
    return rows


def _process_parquet_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"triage_dataset_{split_name}.parquet"
    df = pd.read_parquet(path)
    df["text_en"] = df["text_en"].astype(str).fillna("").str.strip()
    df = df[df["text_en"].str.len() > 0].copy()
    df = df[df["label_triage_gold"].isin(VALID_TRIAGE)].copy()

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        triage = r["label_triage_gold"]
        cat = r.get("label_category_gold") or "other"
        if cat not in VALID_CATEGORIES:
            cat = "other"
        rf = int(r.get("red_flags_gold") or 0)
        weight = float(r.get("sample_weight") or 1.0)
        source = f"parquet_{r.get('source','unknown')}"
        iq = str(r.get("input_quality") or "meaningful").strip().lower()
        if iq not in VALID_QUALITIES:
            iq = "meaningful"
        rows.extend(_prefix_samples(
            r["text_en"], triage, cat, 1 if rf > 0 else 0, weight, source, iq
        ))
    out = pd.DataFrame(rows)
    out["split"] = split_name
    return out


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_assistant_label(content: str) -> tuple[str, str, int] | None:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    triage = str(data.get("triage_level") or "").upper().strip()
    cat = str(data.get("category") or "other").lower().strip()
    rfs = data.get("red_flags") or []
    if triage not in VALID_TRIAGE:
        return None
    if cat not in VALID_CATEGORIES:
        cat = "other"
    return triage, cat, 1 if isinstance(rfs, list) and len(rfs) > 0 else 0


def _process_jsonl(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for jp in JSONL_FILES:
        for i, obj in enumerate(_iter_jsonl(jp)):
            msgs = obj.get("messages") or []
            if len(msgs) < 2:
                continue
            user_msg = next((m for m in msgs if m.get("role") == "user"), None)
            asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
            if not user_msg or not asst_msg:
                continue
            user_text = str(user_msg.get("content") or "").strip()
            if not user_text:
                continue
            parsed = _parse_assistant_label(str(asst_msg.get("content") or ""))
            if parsed is None:
                continue
            triage, cat, rf = parsed
            rows.append({
                "text": _wrap_user(user_text),
                "triage_level": triage,
                "category": cat,
                "red_flag_present": rf,
                "input_quality": "meaningful",
                "sample_weight": JSONL_SAMPLE_WEIGHT,
                "source": f"jsonl_{jp.stem}",
            })
    if not rows:
        empty = pd.DataFrame(columns=[
            "text", "triage_level", "category", "red_flag_present",
            "input_quality", "sample_weight", "source", "split",
        ])
        return empty, empty, empty

    df = pd.DataFrame(rows)
    # Dedupe across files — sometimes groq lora train/val overlap with finetune_train
    df = df.drop_duplicates(subset=["text", "triage_level", "category"], keep="first").reset_index(drop=True)

    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).values
    df = df.iloc[rng].reset_index(drop=True)

    n = len(df)
    n_val = int(round(n * JSONL_VAL_FRACTION))
    n_test = int(round(n * JSONL_VAL_FRACTION))
    val_df = df.iloc[:n_val].copy()
    test_df = df.iloc[n_val:n_val + n_test].copy()
    train_df = df.iloc[n_val + n_test:].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return train_df, val_df, test_df


def _process_multiturn(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """multiturn_dialogs_v1.jsonl: zaten turn-snapshot formatli, direkt okur."""
    if not MULTITURN_JSONL.exists():
        empty = pd.DataFrame(columns=[
            "text", "triage_level", "category", "red_flag_present",
            "input_quality", "sample_weight", "source", "split",
        ])
        return empty, empty, empty
    rows: list[dict[str, Any]] = []
    for obj in _iter_jsonl(MULTITURN_JSONL):
        text = str(obj.get("text") or "").strip()
        triage = str(obj.get("triage_level") or "").upper()
        cat = str(obj.get("category") or "other").lower()
        rf = int(obj.get("red_flag_present") or 0)
        iq = str(obj.get("input_quality") or "meaningful").lower()
        if triage not in VALID_TRIAGE:
            continue
        if cat not in VALID_CATEGORIES:
            cat = "other"
        if iq not in VALID_QUALITIES:
            iq = "meaningful"
        rows.append({
            "text": text,
            "triage_level": triage,
            "category": cat,
            "red_flag_present": rf,
            "input_quality": iq,
            "sample_weight": float(obj.get("sample_weight") or 2.0),
            "source": str(obj.get("source") or "multiturn"),
            "dialog_id": str(obj.get("dialog_id") or ""),
        })
    if not rows:
        empty = pd.DataFrame(columns=[
            "text", "triage_level", "category", "red_flag_present",
            "input_quality", "sample_weight", "source", "split",
        ])
        return empty, empty, empty
    df = pd.DataFrame(rows)

    # Dialog-bazli split: ayni dialog'un tum turlari ayni split'e dussun.
    unique_dialogs = sorted(df["dialog_id"].unique())
    rng = pd.Series(unique_dialogs).sample(frac=1.0, random_state=seed).tolist()
    n = len(rng)
    n_val = max(1, int(round(n * JSONL_VAL_FRACTION)))
    n_test = max(1, int(round(n * JSONL_VAL_FRACTION)))
    val_ids = set(rng[:n_val])
    test_ids = set(rng[n_val:n_val + n_test])
    train_ids = set(rng[n_val + n_test:])

    def _pick(ids: set[str]) -> pd.DataFrame:
        sub = df[df["dialog_id"].isin(ids)].drop(columns=["dialog_id"]).reset_index(drop=True)
        return sub

    train_df = _pick(train_ids).assign(split="train")
    val_df = _pick(val_ids).assign(split="val")
    test_df = _pick(test_ids).assign(split="test")
    return train_df, val_df, test_df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build history-aware triage dataset.")
    p.add_argument("--no-jsonl", action="store_true", help="Skip JSONL dialog sources.")
    p.add_argument("--no-multiturn", action="store_true", help="Skip multiturn_dialogs_v1.jsonl.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        print(f"[parquet] processing split={split} ...")
        df = _process_parquet_split(split)
        print(f"  parquet {split}: {len(df)} prefix rows")
        parts[split].append(df)

    if not args.no_jsonl:
        jt, jv, jtest = _process_jsonl(seed=args.seed)
        print(f"[jsonl] train={len(jt)} val={len(jv)} test={len(jtest)}")
        parts["train"].append(jt)
        parts["val"].append(jv)
        parts["test"].append(jtest)

    if not args.no_multiturn:
        mt, mv, mtest = _process_multiturn(seed=args.seed)
        print(f"[multiturn] train={len(mt)} val={len(mv)} test={len(mtest)}")
        parts["train"].append(mt)
        parts["val"].append(mv)
        parts["test"].append(mtest)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outs = {"train": OUT_TRAIN, "val": OUT_VAL, "test": OUT_TEST}
    for split, files in parts.items():
        merged = pd.concat([f for f in files if len(f) > 0], ignore_index=True)
        merged = merged.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
        print(f"\n=== {split.upper()} ({len(merged)} rows) ===")
        print(merged["triage_level"].value_counts().to_string())
        print("category:")
        print(merged["category"].value_counts().to_string())
        print("input_quality:")
        print(merged["input_quality"].value_counts().to_string())
        merged.to_parquet(outs[split], index=False)
        print(f"  -> {outs[split]}")


if __name__ == "__main__":
    main()
