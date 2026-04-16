#!/usr/bin/env python3
"""
Export emergency chatbot examples to Groq/LoRA-friendly JSONL.

This script reads `data/llm_fine_tune_examples.json` and writes chat-style
records to `data/llm_groq_lora_train.jsonl`.

Usage:
  PYTHONPATH=src python scripts/export_groq_lora_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "llm_fine_tune_examples.json"
OUTPUT_FILE = DATA_DIR / "llm_groq_lora_train.jsonl"


BASE_SYSTEM_PROMPT = (
    "You are a professional emergency dispatcher assistant. "
    "Return ONLY a valid JSON object with keys: response_text, extracted_slots, "
    "triage_level, category, is_complete, red_flags."
)


def _load_examples() -> list[dict]:
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}", file=sys.stderr)
        return []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get("examples", [])
    if isinstance(payload, list):
        return payload
    return []


def _to_record(example: dict) -> dict | None:
    user = (example.get("user") or "").strip()
    assistant_json = example.get("assistant_json") or example.get("assistant")
    if not user or not isinstance(assistant_json, dict):
        return None

    assistant_content = json.dumps(assistant_json, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main() -> None:
    examples = _load_examples()
    if not examples:
        print("No examples found to export.", file=sys.stderr)
        sys.exit(1)

    records: list[str] = []
    for ex in examples:
        record = _to_record(ex)
        if record is None:
            continue
        records.append(json.dumps(record, ensure_ascii=False))

    if not records:
        print("No valid records generated.", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in records:
            f.write(line + "\n")

    print(f"Exported {len(records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
