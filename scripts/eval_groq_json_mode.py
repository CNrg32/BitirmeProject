#!/usr/bin/env python3
"""
Quick evaluation for Groq emergency triage model outputs.

Checks:
- JSON parse success rate
- Required keys presence

Usage:
  GROQ_API_KEY=... PYTHONPATH=src python scripts/eval_groq_json_mode.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "llm_fine_tune_examples.json"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
REQUIRED_KEYS = {
    "response_text",
    "extracted_slots",
    "triage_level",
    "category",
    "is_complete",
    "red_flags",
}


def _load_examples() -> list[dict[str, Any]]:
    if not INPUT_FILE.exists():
        return []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("examples", [])
    return data if isinstance(data, list) else []


def main() -> None:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        print("Missing GROQ_API_KEY", file=sys.stderr)
        sys.exit(1)

    model = (
        os.environ.get("GROQ_FINE_TUNED_MODEL", "").strip()
        or os.environ.get("GROQ_MODEL", "").strip()
        or DEFAULT_MODEL
    )

    try:
        from groq import Groq  # type: ignore
    except Exception:
        print("groq package missing: pip install groq", file=sys.stderr)
        sys.exit(1)

    examples = _load_examples()
    if not examples:
        print(f"No examples found in {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=groq_key)
    parse_ok = 0
    schema_ok = 0
    total = 0

    system_prompt = (
        "You are an emergency dispatcher assistant. "
        "Return ONLY valid JSON output for triage response."
    )

    for ex in examples:
        user = (ex.get("user") or "").strip()
        if not user:
            continue
        total += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=512,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(raw)
            parse_ok += 1
            if REQUIRED_KEYS.issubset(set(parsed.keys())):
                schema_ok += 1
        except Exception:
            continue

    if total == 0:
        print("No valid user examples found.", file=sys.stderr)
        sys.exit(1)

    parse_rate = (parse_ok / total) * 100.0
    schema_rate = (schema_ok / total) * 100.0
    print(f"Model: {model}")
    print(f"Total samples: {total}")
    print(f"JSON parse success: {parse_ok}/{total} ({parse_rate:.1f}%)")
    print(f"Required schema success: {schema_ok}/{total} ({schema_rate:.1f}%)")


if __name__ == "__main__":
    main()
