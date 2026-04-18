#!/usr/bin/env python3
"""
Package LoRA adapter files for Groq upload.

Creates a zip containing:
  - adapter_model.safetensors
  - adapter_config.json

Usage:
  python scripts/package_lora_for_groq.py \
    --adapter-dir out_models/lora_adapter_aws \
    --output out_models/lora_adapter_aws/groq_adapter.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REQUIRED_FILES = ["adapter_model.safetensors", "adapter_config.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package LoRA adapter for Groq.")
    parser.add_argument("--adapter-dir", required=True, help="Directory containing LoRA adapter files.")
    parser.add_argument("--output", required=True, help="Output zip path.")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED_FILES if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {adapter_dir}: {', '.join(missing)}"
        )

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in REQUIRED_FILES:
            zf.write(adapter_dir / name, arcname=name)

    print(f"Created Groq adapter package: {output}")


if __name__ == "__main__":
    main()
