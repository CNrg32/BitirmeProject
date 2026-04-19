#!/usr/bin/env python3
"""
Split Groq/Together LoRA JSONL data into train/validation files.

Usage:
  python scripts/split_groq_lora_data.py \
    --input data/llm_groq_lora_train.jsonl \
    --train-output data/llm_groq_lora_train_split.jsonl \
    --val-output data/llm_groq_lora_val.jsonl \
    --val-ratio 0.1 \
    --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split JSONL into train/validation files.")
    parser.add_argument("--input", default="data/llm_groq_lora_train.jsonl", help="Input JSONL file path.")
    parser.add_argument(
        "--train-output",
        default="data/llm_groq_lora_train_split.jsonl",
        help="Train JSONL output path.",
    )
    parser.add_argument(
        "--val-output",
        default="data/llm_groq_lora_val.jsonl",
        help="Validation JSONL output path.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio in [0, 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.val_ratio < 1):
        raise ValueError("--val-ratio must be between 0 and 1.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 10:
        raise ValueError("Dataset too small to split reliably (need at least 10 lines).")

    rng = random.Random(args.seed)
    rng.shuffle(lines)

    val_count = max(1, int(len(lines) * args.val_ratio))
    val_lines = lines[:val_count]
    train_lines = lines[val_count:]

    train_out = Path(args.train_output)
    val_out = Path(args.val_output)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    train_out.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_out.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(
        "Split completed:",
        f"total={len(lines)}",
        f"train={len(train_lines)}",
        f"val={len(val_lines)}",
        f"train_file={train_out}",
        f"val_file={val_out}",
    )


if __name__ == "__main__":
    main()
