#!/usr/bin/env python3
"""
Fine-tuning veri setini JSONL formatına dönüştürür.

Kullanım:
  PYTHONPATH=src python scripts/export_llm_finetune_data.py

Çıktı: data/llm_finetune_train.jsonl (OpenAI / genel chat fine-tune formatı).
Kendi eklediğiniz data/llm_fine_tune_examples.json örnekleri bu dosyaya dahil edilir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Proje kökü
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXAMPLES_FILE = DATA_DIR / "llm_fine_tune_examples.json"
OUTPUT_JSONL = DATA_DIR / "llm_finetune_train.jsonl"


def load_examples() -> list:
    if not EXAMPLES_FILE.exists():
        print(f"File not found: {EXAMPLES_FILE}", file=sys.stderr)
        return []
    with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("examples", [])


def main() -> None:
    examples = load_examples()
    if not examples:
        print("No examples to export.", file=sys.stderr)
        sys.exit(1)

    # OpenAI chat fine-tune format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    lines = []
    for ex in examples:
        user = ex.get("user", "")
        ast = ex.get("assistant_json") or ex.get("assistant")
        if not user or not isinstance(ast, dict):
            continue
        assistant_content = json.dumps(ast, ensure_ascii=False)
        record = {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Exported {len(lines)} examples to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
