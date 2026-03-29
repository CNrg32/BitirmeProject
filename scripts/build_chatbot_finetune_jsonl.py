from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def _first_non_empty(row: pd.Series, cols: Iterable[str], default: str = "") -> str:
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v:
                return v
    return default


def _normalize_triage(value: str) -> str:
    v = (value or "").strip().upper()
    if v in {"CRITICAL", "URGENT", "NON_URGENT"}:
        return v
    return "URGENT"


def _normalize_category(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"medical", "fire", "crime", "other"}:
        return v
    return "other"


def _normalize_red_flag(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y"}


def _next_question_by_category(category: str) -> str:
    if category == "medical":
        return "Is the person conscious right now?"
    if category == "fire":
        return "Are there any people trapped inside?"
    if category == "crime":
        return "Is the assailant still at the scene?"
    return "Can you briefly describe what happened?"


def _build_example(text: str, category: str, triage: str, red_flag: bool, language: str = "en") -> Dict[str, Any]:
    extracted_slots: Dict[str, Any] = {"chief_complaint": text[:180]}
    response_text = _next_question_by_category(category)

    is_complete = triage == "CRITICAL" and red_flag
    red_flags = ["life_threatening_sign"] if red_flag else []

    return {
        "language": language,
        "history": [
            {"role": "user", "text": text},
        ],
        "target": {
            "response_text": response_text,
            "extracted_slots": extracted_slots,
            "triage_level": triage,
            "category": category,
            "is_complete": is_complete,
            "red_flags": red_flags,
        },
    }


def _load_records(csv_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        text = _first_non_empty(row, ["text_en", "transcript_en", "text_clean", "text", "description"])
        if len(text) < 12:
            continue

        category = _normalize_category(
            _first_non_empty(row, ["label_category_gold", "label_category_weak"], default="other")
        )
        triage = _normalize_triage(
            _first_non_empty(row, ["label_triage_gold", "label_triage_weak"], default="URGENT")
        )
        red_flag = _normalize_red_flag(
            _first_non_empty(row, ["red_flags_gold", "red_flags_weak"], default="0")
        )

        records.append(_build_example(text=text, category=category, triage=triage, red_flag=red_flag))

    return records


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chatbot fine-tune JSONL files from existing triage CSV datasets.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/labels/synthetic_triage_cases_v1.csv",
            "data/labels/synthetic_triage_cases_v2.csv",
            "data/labels/911_cases_v1.csv",
        ],
        help="Input CSV files.",
    )
    parser.add_argument("--output-train", default="data/labels/chatbot_finetune_train.jsonl")
    parser.add_argument("--output-val", default="data/labels/chatbot_finetune_val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on total samples (0 means no cap).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    all_rows: List[Dict[str, Any]] = []
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            continue
        all_rows.extend(_load_records(path))

    if not all_rows:
        raise RuntimeError("No training records produced. Check input paths and column names.")

    random.shuffle(all_rows)
    if args.max_samples > 0:
        all_rows = all_rows[: args.max_samples]

    split_idx = max(1, int(len(all_rows) * (1.0 - args.val_ratio)))
    train_rows = all_rows[:split_idx]
    val_rows = all_rows[split_idx:]
    if not val_rows:
        val_rows = train_rows[-1:]
        train_rows = train_rows[:-1]

    out_train = Path(args.output_train)
    out_val = Path(args.output_val)
    _write_jsonl(out_train, train_rows)
    _write_jsonl(out_val, val_rows)

    print(f"Total: {len(all_rows)} | Train: {len(train_rows)} | Val: {len(val_rows)}")
    print(f"Train file: {out_train}")
    print(f"Val file: {out_val}")


if __name__ == "__main__":
    main()
