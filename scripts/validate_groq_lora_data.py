#!/usr/bin/env python3
"""
Validate chat-format JSONL data before LoRA fine-tuning.

Checks each line for:
  - valid JSON object
  - messages array with system/user/assistant roles
  - assistant content that is valid JSON
  - required emergency triage output keys

Usage:
  python scripts/validate_groq_lora_data.py data/llm_groq_lora_train_split.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_ASSISTANT_KEYS = {
    "response_text",
    "extracted_slots",
    "triage_level",
    "category",
    "is_complete",
    "red_flags",
}
VALID_ROLES = {"system", "user", "assistant"}
VALID_TRIAGE = {"CRITICAL", "URGENT", "NON_URGENT"}
VALID_CATEGORY = {"medical", "fire", "crime", "other"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Groq/Together LoRA JSONL data.")
    parser.add_argument("path", help="JSONL file to validate.")
    parser.add_argument("--max-errors", type=int, default=20, help="Stop after this many examples.")
    return parser.parse_args()


def _error(errors: list[str], line_no: int, message: str, max_errors: int) -> None:
    if len(errors) < max_errors:
        errors.append(f"line {line_no}: {message}")


def _validate_line(
    raw: str,
    line_no: int,
    errors: list[str],
    stats: Counter[str],
    max_errors: int,
) -> None:
    try:
        item = json.loads(raw)
    except json.JSONDecodeError as exc:
        _error(errors, line_no, f"invalid JSONL item: {exc}", max_errors)
        return

    messages = item.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        _error(errors, line_no, "messages must be a list with at least 3 items", max_errors)
        return

    roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
    if len(roles) != len(messages) or any(role not in VALID_ROLES for role in roles):
        _error(errors, line_no, f"invalid roles: {roles}", max_errors)
        return
    if "user" not in roles or "assistant" not in roles:
        _error(errors, line_no, f"must include user and assistant roles: {roles}", max_errors)
        return

    for index, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str) or not content.strip():
            _error(errors, line_no, f"message {index} has empty content", max_errors)
            return

    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    try:
        assistant_payload: dict[str, Any] = json.loads(assistant_messages[-1]["content"])
    except json.JSONDecodeError as exc:
        _error(errors, line_no, f"assistant content is not valid JSON: {exc}", max_errors)
        return

    missing = REQUIRED_ASSISTANT_KEYS - set(assistant_payload)
    if missing:
        _error(errors, line_no, f"assistant JSON missing keys: {sorted(missing)}", max_errors)
        return

    triage = assistant_payload.get("triage_level")
    category = assistant_payload.get("category")
    if triage not in VALID_TRIAGE:
        _error(errors, line_no, f"invalid triage_level: {triage}", max_errors)
    if category not in VALID_CATEGORY:
        _error(errors, line_no, f"invalid category: {category}", max_errors)
    if not isinstance(assistant_payload.get("extracted_slots"), dict):
        _error(errors, line_no, "extracted_slots must be an object", max_errors)
    if not isinstance(assistant_payload.get("red_flags"), list):
        _error(errors, line_no, "red_flags must be a list", max_errors)

    stats[f"triage:{triage}"] += 1
    stats[f"category:{category}"] += 1


def main() -> int:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    errors: list[str] = []
    stats: Counter[str] = Counter()
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            _validate_line(raw, line_no, errors, stats, args.max_errors)

    print(f"file={path}")
    print(f"examples={total}")
    print("triage_distribution:")
    for key, value in sorted((k, v) for k, v in stats.items() if k.startswith("triage:")):
        print(f"  {key.removeprefix('triage:')}: {value}")
    print("category_distribution:")
    for key, value in sorted((k, v) for k, v in stats.items() if k.startswith("category:")):
        print(f"  {key.removeprefix('category:')}: {value}")

    if errors:
        print("errors:")
        for err in errors:
            print(f"  - {err}")
        if len(errors) >= args.max_errors:
            print(f"  ... stopped displaying after {args.max_errors} errors")
        return 1

    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
