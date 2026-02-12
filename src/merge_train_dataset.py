
"""
Merge gold-reviewed 911 data with synthetic triage cases, and create train/val/test splits.

Inputs:
  - gold_suggestions_911.csv  (you review/fix *_gold columns)
  - synthetic_triage_cases_v1.csv

Output:
  - triage_dataset_merged.parquet
  - triage_dataset_train.parquet
  - triage_dataset_val.parquet
  - triage_dataset_test.parquet

Notes:
  - Uses label_*_gold as ground truth when present.
  - For 911 data, prefers transcript_en if you later fill it; otherwise uses text_clean.
  - Keeps "source" column for analysis.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

GOLD_CSV = Path("../data/labels/gold_suggestions_911.csv")
SYNTH_CSV = Path("../data/labels/synthetic_triage_cases_v1.csv")
OUT_DIR = Path("Outputs/out_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15

def _pick_text_row(r: pd.Series) -> str:
    # If you later add transcript_en into the gold CSV, use it.
    for col in ["transcript_en", "text_clean", "text_raw"]:
        if col in r and isinstance(r[col], str) and r[col].strip():
            return r[col].strip()
    return ""

def load_gold() -> pd.DataFrame:
    df = pd.read_csv(GOLD_CSV)
    df["source"] = "kaggle_911"
    df["case_id"] = df["id"].apply(lambda x: f"911_{int(x)}" if pd.notna(x) else "911_unknown")
    df["text_en"] = df.apply(_pick_text_row, axis=1)

    # Keep only rows with non-empty text and a triage label
    df["label_triage_gold"] = df["label_triage_gold"].astype(str).str.strip()
    df = df[df["text_en"].str.len() > 0]
    df = df[df["label_triage_gold"].isin(["CRITICAL","URGENT","NON_URGENT"])].copy()

    df["label_category_gold"] = df["label_category_gold"].astype(str).str.strip()
    df.loc[~df["label_category_gold"].isin(["medical","crime","fire","other"]), "label_category_gold"] = "other"

    # red flags as int
    df["red_flags_gold"] = pd.to_numeric(df["red_flags_gold"], errors="coerce").fillna(0).astype(int)

    keep_cols = [
        "case_id","source","text_en",
        "label_category_gold","label_triage_gold","red_flags_gold",
        "deaths","potential_death","false_alarm","state","date",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    return df[keep_cols]

def load_synth() -> pd.DataFrame:
    df = pd.read_csv(SYNTH_CSV)
    df = df.rename(columns={"case_id":"case_id", "text_en":"text_en"})
    df["source"] = df.get("source","synthetic_v1")
    # Ensure required cols exist
    df["label_category_gold"] = df["label_category_gold"].astype(str).str.strip()
    df.loc[~df["label_category_gold"].isin(["medical","crime","fire","other"]), "label_category_gold"] = "other"
    df["label_triage_gold"] = df["label_triage_gold"].astype(str).str.strip()
    df = df[df["label_triage_gold"].isin(["CRITICAL","URGENT","NON_URGENT"])].copy()
    df["red_flags_gold"] = pd.to_numeric(df["red_flags_gold"], errors="coerce").fillna(0).astype(int)
    for c in ["deaths","potential_death","false_alarm","state","date"]:
        if c not in df.columns:
            df[c] = None
    keep_cols = [
        "case_id","source","text_en",
        "label_category_gold","label_triage_gold","red_flags_gold",
        "deaths","potential_death","false_alarm","state","date",
    ]
    return df[keep_cols]

def stratified_split(df: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Stratify on triage label (primary) + source (secondary) to keep mix
    strat = (df["label_triage_gold"].astype(str) + "|" + df["source"].astype(str))
    df = df.copy()
    df["_strat"] = strat

    test_idx = []
    val_idx = []
    train_idx = []

    for key, grp in df.groupby("_strat"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(TEST_FRACTION * n))) if n >= 8 else max(0, int(round(TEST_FRACTION * n)))
        n_val  = max(1, int(round(VAL_FRACTION * n))) if n >= 8 else max(0, int(round(VAL_FRACTION * n)))

        test_part = idx[:n_test]
        val_part = idx[n_test:n_test+n_val]
        train_part = idx[n_test+n_val:]

        test_idx.extend(test_part.tolist())
        val_idx.extend(val_part.tolist())
        train_idx.extend(train_part.tolist())

    train = df.loc[train_idx].drop(columns=["_strat"])
    val = df.loc[val_idx].drop(columns=["_strat"])
    test = df.loc[test_idx].drop(columns=["_strat"])
    return train, val, test

def main():
    gold = load_gold()
    synth = load_synth()

    merged = pd.concat([gold, synth], ignore_index=True)
    merged = merged.drop_duplicates(subset=["case_id"]).reset_index(drop=True)

    # Save merged
    merged_path = OUT_DIR / "triage_dataset_merged.parquet"
    merged.to_parquet(merged_path, index=False)

    train, val, test = stratified_split(merged, seed=RANDOM_SEED)

    train.to_parquet(OUT_DIR / "triage_dataset_train.parquet", index=False)
    val.to_parquet(OUT_DIR / "triage_dataset_val.parquet", index=False)
    test.to_parquet(OUT_DIR / "triage_dataset_test.parquet", index=False)

    print("Saved:")
    print(" ", merged_path)
    print(" ", OUT_DIR / "triage_dataset_train.parquet", len(train))
    print(" ", OUT_DIR / "triage_dataset_val.parquet", len(val))
    print(" ", OUT_DIR / "triage_dataset_test.parquet", len(test))
    print("\nLabel distribution (train):")
    print(train["label_triage_gold"].value_counts())

if __name__ == "__main__":
    main()
