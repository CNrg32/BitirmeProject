#!/usr/bin/env python3
"""
eval_triage_local.py
====================
Egitilen yerel XLM-R triage modelini iki perspektiften degerlendirir:

  1. Genel test seti (output/out_dataset/triage_history_test.parquet)
     -> classification report (triage + category + red_flag)
  2. "Minor complaint stress test": sadece minor_complaints_v1 ve
     ambiguous_symptoms_v1 kaynakli test ornekleri -> model bu ornekleri
     CRITICAL olarak isaretliyor mu? Oran <5% ise basarili kabul edilir.

Ayrica LocalTriageService conservative rule'larinin ham model ciktisini nasil
etkiledigini gosterir (ham vs guarded).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from services.triage_local_service import LocalTriageService  # noqa: E402
from services.triage_model import (  # noqa: E402
    CATEGORY_LABELS,
    TRIAGE_LABELS,
)

TEST_PATH = _PROJECT_ROOT / "output" / "out_dataset" / "triage_history_test.parquet"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=str(_PROJECT_ROOT / "out_models" / "triage_xlmr"))
    p.add_argument("--show-raw", action="store_true",
                   help="conservative rule oncesi ham argmax ciktisi da raporlanir")
    return p.parse_args()


def _fake_history(text: str) -> list[dict]:
    """Flat text'i tek-turn user history'sine donustur ([USER] prefix'i LocalService ekliyor)."""
    if text.startswith("[USER] "):
        text = text[len("[USER] "):]
    return [{"role": "user", "text": text}]


def _evaluate(svc: LocalTriageService, df: pd.DataFrame, label_col: str = "triage_level") -> dict:
    y_true, y_pred_guarded = [], []
    y_pred_raw = []
    confs = []
    for _, row in df.iterrows():
        history = _fake_history(str(row["text"]))
        probs, cat_probs, rf_prob = svc._predict_raw(_flatten(history))  # type: ignore[attr-defined]
        raw_idx = int(np.argmax(probs))
        y_pred_raw.append(svc._triage_labels[raw_idx])
        guarded, conf = svc._apply_conservative_rules(list(probs), float(rf_prob), user_turns=1)
        y_pred_guarded.append(guarded)
        y_true.append(row[label_col])
        confs.append(conf)
    return {
        "y_true": y_true,
        "y_pred_raw": y_pred_raw,
        "y_pred_guarded": y_pred_guarded,
        "confs": confs,
    }


def _flatten(history: list[dict]) -> str:
    parts = []
    for msg in history:
        role = "ASSISTANT" if msg.get("role") == "assistant" else "USER"
        parts.append(f"[{role}] {msg.get('text','').strip()}")
    return " ".join(parts)


def main() -> None:
    args = _parse_args()
    svc = LocalTriageService(model_dir=Path(args.model_dir))
    if not svc.is_available:
        print(f"Model not found at {args.model_dir}")
        sys.exit(1)
    svc._ensure_loaded()  # type: ignore[attr-defined]

    df = pd.read_parquet(TEST_PATH)
    print(f"\n== FULL TEST ({len(df)} rows) ==")
    out = _evaluate(svc, df)
    print("\nTRIAGE (guarded):")
    print(classification_report(out["y_true"], out["y_pred_guarded"], labels=TRIAGE_LABELS, digits=3))
    print(confusion_matrix(out["y_true"], out["y_pred_guarded"], labels=TRIAGE_LABELS))

    if args.show_raw:
        print("\nTRIAGE (raw argmax, no guardrails):")
        print(classification_report(out["y_true"], out["y_pred_raw"], labels=TRIAGE_LABELS, digits=3))

    # minor-complaint stress test
    stress_df = df[df["source"].isin([
        "jsonl_llm_finetune_train",
        "jsonl_llm_groq_lora_train",
        "jsonl_llm_groq_lora_val",
        "parquet_minor_complaints_v1",
        "parquet_ambiguous_symptoms_v1",
    ])].copy()

    if len(stress_df) > 0:
        print(f"\n== STRESS TEST: conservative datasets ({len(stress_df)} rows) ==")
        s_out = _evaluate(svc, stress_df)
        crit_rate = sum(1 for y in s_out["y_pred_guarded"] if y == "CRITICAL") / max(1, len(stress_df))
        non_rate = sum(1 for y in s_out["y_pred_guarded"] if y == "NON_URGENT") / max(1, len(stress_df))
        urg_rate = sum(1 for y in s_out["y_pred_guarded"] if y == "URGENT") / max(1, len(stress_df))
        print(f"CRITICAL rate: {crit_rate:.3f} (target <= 0.05 for minor cases)")
        print(f"URGENT rate:   {urg_rate:.3f}")
        print(f"NON_URGENT:    {non_rate:.3f}")
        print(classification_report(s_out["y_true"], s_out["y_pred_guarded"],
                                    labels=TRIAGE_LABELS, digits=3, zero_division=0))
    else:
        print("No stress-test rows found in test split.")


if __name__ == "__main__":
    main()
