# 911 gold verisi + sentetik triaj verilerini birlestirip train/val/test split olusturur
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Paths relative to project root (script lives in scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_CSV = _PROJECT_ROOT / "data" / "labels" / "911_cases_v1.csv"
AUTO_TRANSCRIPTS_CSV = _PROJECT_ROOT / "data" / "labels" / "auto_transcripts.csv"
SYNTH_V1_CSV = _PROJECT_ROOT / "data" / "labels" / "synthetic_triage_cases_v1.csv"
SYNTH_V2_CSV = _PROJECT_ROOT / "data" / "labels" / "synthetic_triage_cases_v2.csv"
OUT_DIR = _PROJECT_ROOT / "output" / "out_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15

VALID_TRIAGE = {"CRITICAL", "URGENT", "NON_URGENT"}
VALID_CATEGORIES = {"medical", "crime", "fire", "other"}

MIN_TEXT_LEN = 60       # cok kisa metinleri (baslik/metadata) ele
MAX_TEXT_LEN = 1500     # cok uzun dispatch transcript'lerini kes

REAL_DATA_WEIGHT = 2.0
SYNTH_V2_WEIGHT = 1.5
SYNTH_V1_WEIGHT = 1.0

MINORITY_FRACTION_OF_MAJORITY = 0.6

# Extract call id from path like "data/raw/911_recordings/audio/call_123.mp3"
_CALL_PATH_RE = re.compile(r"call_(\d+)\.mp3", re.IGNORECASE)


def _load_transcript_by_id() -> dict[int, str]:
    """auto_transcripts.csv'den id -> transcript eslesmesi yukler."""
    out: dict[int, str] = {}
    if not AUTO_TRANSCRIPTS_CSV.exists():
        return out
    try:
        at = pd.read_csv(AUTO_TRANSCRIPTS_CSV)
        if "path" not in at.columns or "text" not in at.columns:
            return out
        for _, row in at.iterrows():
            path = str(row.get("path", ""))
            text = row.get("text")
            if pd.isna(text) or not path:
                continue
            m = _CALL_PATH_RE.search(path)
            if m:
                cid = int(m.group(1))
                out[cid] = str(text).strip()
        return out
    except Exception:
        return out


def _clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > MAX_TEXT_LEN:
        truncated = text[:MAX_TEXT_LEN]
        last_period = truncated.rfind(".")
        last_excl = truncated.rfind("!")
        last_q = truncated.rfind("?")
        cut_point = max(last_period, last_excl, last_q)
        if cut_point > MAX_TEXT_LEN * 0.5:
            text = text[:cut_point + 1]
        else:
            text = truncated
    return text


def load_gold() -> pd.DataFrame:
    df = pd.read_csv(GOLD_CSV)

    df["source"] = "kaggle_911"
    df["case_id"] = df["id"].apply(
        lambda x: f"911_{int(float(x))}" if pd.notna(x) and str(x).strip() != "" else "911_unknown"
    )

    df["text_en"] = df["text_clean"].astype(str).str.strip()
    transcript_by_id = _load_transcript_by_id()
    if transcript_by_id:
        def _best_text(row):
            try:
                cid = int(float(row["id"]))
                if cid in transcript_by_id and transcript_by_id[cid]:
                    return transcript_by_id[cid]
            except (ValueError, TypeError):
                pass
            return row["text_en"]
        df["text_en"] = df.apply(_best_text, axis=1)

    df["text_en"] = df["text_en"].apply(_clean_text)

    df["label_triage_gold"] = df["label_triage_gold"].astype(str).str.strip()
    df = df[df["label_triage_gold"].isin(VALID_TRIAGE)].copy()
    df = df[df["text_en"].str.len() >= MIN_TEXT_LEN].copy()

    df["label_category_gold"] = df["label_category_gold"].astype(str).str.strip()
    df.loc[~df["label_category_gold"].isin(VALID_CATEGORIES), "label_category_gold"] = "other"

    df["red_flags_gold"] = pd.to_numeric(df["red_flags_gold"], errors="coerce").fillna(0).astype(int)

    keep_cols = [
        "case_id", "source", "text_en",
        "label_category_gold", "label_triage_gold", "red_flags_gold",
        "deaths", "potential_death", "false_alarm", "state", "date",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    return df[keep_cols]

def load_synth(csv_path: Path, source_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"  [WARN] {csv_path} not found, skipping.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["source"] = df.get("source", source_name)

    df["label_category_gold"] = df["label_category_gold"].astype(str).str.strip()
    df.loc[~df["label_category_gold"].isin(VALID_CATEGORIES), "label_category_gold"] = "other"
    df["label_triage_gold"] = df["label_triage_gold"].astype(str).str.strip()
    df = df[df["label_triage_gold"].isin(VALID_TRIAGE)].copy()
    df["red_flags_gold"] = pd.to_numeric(df["red_flags_gold"], errors="coerce").fillna(0).astype(int)

    df["text_en"] = df["text_en"].apply(_clean_text)
    df = df[df["text_en"].str.len() >= 20].copy()

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


def _oversample_minority_triage(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Azinlik siniflarini oversampling ile cogunluk sinifina yaklastir."""
    rng = np.random.default_rng(seed)
    counts = df["label_triage_gold"].value_counts()
    if len(counts) < 2:
        return df
    majority_count = int(counts.iloc[0])
    target_minority = max(1, int(majority_count * MINORITY_FRACTION_OF_MAJORITY))
    parts = []
    for label in VALID_TRIAGE:
        sub = df[df["label_triage_gold"] == label].copy()
        n = len(sub)
        if n == 0:
            continue
        if n < target_minority:
            idx = rng.choice(sub.index, size=target_minority, replace=True)
            oversampled = df.loc[idx].copy()
            oversampled = oversampled.reset_index(drop=True)
            oversampled["case_id"] = [f"{oversampled.iloc[i]['case_id']}_os{i}" for i in range(len(oversampled))]
            parts.append(oversampled)
        else:
            parts.append(sub.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else df


def main():
    print("Loading 911 gold data...")
    gold = load_gold()
    print(f"  911 gold rows after cleaning: {len(gold)}")
    print(f"  Triage: {gold['label_triage_gold'].value_counts().to_dict()}")

    gold = _oversample_minority_triage(gold, seed=RANDOM_SEED)
    gold["sample_weight"] = REAL_DATA_WEIGHT
    print(f"  After oversampling: {len(gold)}")

    print("\nLoading synthetic v1...")
    synth_v1 = load_synth(SYNTH_V1_CSV, "synthetic_v1")
    synth_v1["sample_weight"] = SYNTH_V1_WEIGHT
    print(f"  Synth v1 rows: {len(synth_v1)}")

    print("\nLoading synthetic v2...")
    synth_v2 = load_synth(SYNTH_V2_CSV, "synthetic_v2")
    synth_v2["sample_weight"] = SYNTH_V2_WEIGHT
    print(f"  Synth v2 rows: {len(synth_v2)}")

    parts = [gold]
    if len(synth_v1) > 0:
        parts.append(synth_v1)
    if len(synth_v2) > 0:
        parts.append(synth_v2)
    merged = pd.concat(parts, ignore_index=True)

    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["text_en"], keep="first").reset_index(drop=True)
    print(f"\nAfter text dedup: {before_dedup} → {len(merged)}")

    merged = merged.drop_duplicates(subset=["case_id"], keep="first").reset_index(drop=True)
    print(f"\n=== MERGED DATASET ===")
    print(f"Total rows: {len(merged)}")
    print(f"Triage distribution:")
    print(merged["label_triage_gold"].value_counts().to_string())
    print(f"\nCategory distribution:")
    print(merged["label_category_gold"].value_counts().to_string())
    print(f"\nSource distribution:")
    print(merged["source"].value_counts().to_string())
    print(f"\nText length stats:")
    lengths = merged["text_en"].str.len()
    print(f"  Mean: {lengths.mean():.0f}, Median: {lengths.median():.0f}, Min: {lengths.min()}, Max: {lengths.max()}")

    merged_path = OUT_DIR / "triage_dataset_merged.parquet"
    merged.to_parquet(merged_path, index=False)

    train, val, test = stratified_split(merged, seed=RANDOM_SEED)

    train.to_parquet(OUT_DIR / "triage_dataset_train.parquet", index=False)
    val.to_parquet(OUT_DIR / "triage_dataset_val.parquet", index=False)
    test.to_parquet(OUT_DIR / "triage_dataset_test.parquet", index=False)

    print(f"\nSaved:")
    print(f"  {merged_path}")
    print(f"  train: {len(train)} rows")
    print(f"  val:   {len(val)} rows")
    print(f"  test:  {len(test)} rows")
    print(f"\nTrain triage distribution:")
    print(train["label_triage_gold"].value_counts().to_string())
    print(f"\nTrain source distribution:")
    print(train["source"].value_counts().to_string())

if __name__ == "__main__":
    main()
