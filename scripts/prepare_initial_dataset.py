"""
Prepare initial 911 labeled dataset (911_cases_v1.csv) from metadata.

Reads 911_metadata_final.csv and:
  1. Combines title + description into raw/clean text columns
  2. Adds placeholder columns for ASR transcripts
  3. Generates weak labels (category, triage, red flags) using rule-based classification
  4. Adds empty gold-label columns for human review
  5. Outputs 911_cases_v1.csv

Usage:
  python scripts/prepare_initial_dataset.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ─── Paths ───────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_CSV = _PROJECT_ROOT / "data" / "labels" / "911_metadata_final.csv"
OUT_CSV = _PROJECT_ROOT / "data" / "labels" / "911_cases_v1.csv"

# ─── Keyword Lists ──────────────────────────────────────────────────────────
CRIME_KW = [
    "robbery", "burglary", "theft", "gun", "gunshot", "shooting", "shot",
    "stab", "stabbing", "knife", "hostage", "assault", "fight", "domestic",
    "rape", "murder", "homicide", "suspect",
]
FIRE_KW = [
    "fire", "smoke", "burning", "explosion", "bomb", "gas leak",
    "house fire", "building fire", "wildfire",
]
MED_KW = [
    "unconscious", "not breathing", "cant breathe", "can't breathe",
    "difficulty breathing", "shortness of breath", "chest pain",
    "heart attack", "cardiac", "stroke", "seizure", "overdose", "poison",
    "allergic", "anaphylaxis", "bleeding", "hemorrhage", "pregnant",
    "labor", "choking", "fainted", "collapse", "fever", "diabetic",
]

RED_FLAG_KW = [
    "not breathing", "unconscious", "cardiac arrest", "heart attack",
    "stroke", "severe bleeding", "anaphylaxis", "choking", "gunshot",
    "shooting", "stabbing", "house fire", "explosion",
]
URGENT_KW = [
    "chest pain", "difficulty breathing", "shortness of breath", "seizure",
    "overdose", "car accident", "crash", "broken", "fracture", "bleeding",
]
NON_URGENT_KW = [
    "minor", "small cut", "non emergency", "noise complaint",
]


# ─── Helper Functions ────────────────────────────────────────────────────────
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def has_any(text: str, kws: list[str]) -> bool:
    return any(kw in text for kw in kws)


def infer_category(text: str) -> str:
    """Rule-based category inference from text."""
    if has_any(text, FIRE_KW):
        return "fire"
    if has_any(text, CRIME_KW):
        return "crime"
    if has_any(text, MED_KW):
        return "medical"
    return "other"


def infer_triage(row: pd.Series, text: str) -> tuple[str, int]:
    """Rule-based triage level + red flag inference."""
    deaths = row.get("deaths")
    potential_death = row.get("potential_death")
    false_alarm = row.get("false_alarm")

    red_flag = 1 if (
        has_any(text, RED_FLAG_KW)
        or (pd.notna(potential_death) and float(potential_death) == 1.0)
    ) else 0

    # CRITICAL: deaths, potential death, or red flags
    if (
        (pd.notna(deaths) and float(deaths) > 0)
        or (pd.notna(potential_death) and float(potential_death) == 1.0)
        or red_flag == 1
    ):
        return "CRITICAL", red_flag

    # URGENT: urgent keywords or crime/fire category
    if has_any(text, URGENT_KW) or infer_category(text) in ["crime", "fire"]:
        return "URGENT", red_flag

    # NON_URGENT: false alarm or non-urgent keywords
    if (
        (pd.notna(false_alarm) and float(false_alarm) == 1.0)
        or has_any(text, NON_URGENT_KW)
    ):
        return "NON_URGENT", red_flag

    # Default: conservative URGENT
    return "URGENT", red_flag


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    if not IN_CSV.exists():
        print(f"Input file not found: {IN_CSV}")
        print("This script requires 911_metadata_final.csv in data/labels/")
        return

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} rows from {IN_CSV.name}")

    # Raw and cleaned text
    df["text_raw"] = (
        df["title"].fillna("") + " " + df["description"].fillna("")
    ).str.strip()
    df["text_clean"] = df["text_raw"].apply(normalize_text)

    # Placeholders for ASR pipeline
    df["transcript_src"] = ""
    df["transcript_en"] = ""
    df["asr_confidence"] = ""

    # Weak labels (rule-based)
    df["label_category_weak"] = df["text_clean"].apply(infer_category)
    triage_and_flag = df.apply(
        lambda r: infer_triage(r, r["text_clean"]), axis=1,
    )
    df["label_triage_weak"] = [x[0] for x in triage_and_flag]
    df["red_flags_weak"] = [x[1] for x in triage_and_flag]
    df["weak_label_source"] = "rules_v1"

    # Gold columns (empty, for human review)
    df["label_category_gold"] = ""
    df["label_triage_gold"] = ""
    df["red_flags_gold"] = ""
    df["reviewed_by"] = ""
    df["review_notes"] = ""

    # Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} ({len(df)} rows)")

    # Summary
    print("\nCategory distribution (weak):")
    print(df["label_category_weak"].value_counts())
    print("\nTriage distribution (weak):")
    print(df["label_triage_weak"].value_counts())


if __name__ == "__main__":
    main()
