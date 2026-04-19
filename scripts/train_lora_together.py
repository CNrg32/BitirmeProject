#!/usr/bin/env python3
"""
Start and optionally monitor a Together AI LoRA fine-tuning job.

This script uploads a chat-format JSONL file and creates a fine-tune job.

Usage:
  export TOGETHER_API_KEY="..."
  PYTHONPATH=src python scripts/train_lora_together.py \
    --train-file data/llm_groq_lora_train.jsonl \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --wait
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from together import Together


def _coerce(value: Any) -> dict[str, Any]:
    """Convert Together SDK responses into plain dict for stable access."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _require_api_key() -> str:
    api_key = os.getenv("TOGETHER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY is required.")
    return api_key


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Together AI LoRA fine-tuning job.")
    parser.add_argument("--train-file", default="data/llm_groq_lora_train.jsonl", help="Training JSONL path.")
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--validation-file",
        default="",
        help="Optional validation JSONL path. If provided, it will be uploaded too.",
    )
    parser.add_argument("--suffix", default="bitirme-emergency-triage", help="Fine-tune job suffix.")
    parser.add_argument("--wait", action="store_true", help="Poll job status until completion.")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval in seconds.")
    parser.add_argument("--lora", action="store_true", help="Request LoRA mode explicitly.")
    return parser.parse_args()


def _upload_file(client: Together, path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    upload_resp = client.files.upload(file=str(path))
    payload = _coerce(upload_resp)
    file_id = payload.get("id")
    if not file_id:
        raise RuntimeError(f"Could not extract file id from upload response: {payload}")
    print(f"Uploaded {path} -> {file_id}")
    return str(file_id)


def _create_job(client: Together, args: argparse.Namespace, train_file_id: str, val_file_id: str | None) -> str:
    req: dict[str, Any] = {
        "training_file": train_file_id,
        "model": args.base_model,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "suffix": args.suffix,
    }
    if args.lora:
        req["lora"] = True
    if val_file_id:
        req["validation_file"] = val_file_id

    job = client.fine_tuning.create(**req)
    payload = _coerce(job)
    job_id = payload.get("id")
    status = payload.get("status", "unknown")
    if not job_id:
        raise RuntimeError(f"Could not extract job id from response: {payload}")
    print(f"Created fine-tune job: {job_id} (status={status})")
    return str(job_id)


def _poll_job(client: Together, job_id: str, poll_seconds: int) -> None:
    done_states = {"succeeded", "failed", "cancelled"}
    while True:
        job = client.fine_tuning.retrieve(job_id)
        payload = _coerce(job)
        status = str(payload.get("status", "unknown")).lower()
        output_name = payload.get("output_name") or payload.get("fine_tuned_model")
        print(f"[{job_id}] status={status} output={output_name}")
        if status in done_states:
            break
        time.sleep(max(5, poll_seconds))


def main() -> None:
    args = _parse_args()
    api_key = _require_api_key()
    client = Together(api_key=api_key)

    train_file_id = _upload_file(client, args.train_file)
    val_file_id = _upload_file(client, args.validation_file) if args.validation_file else None
    job_id = _create_job(client, args, train_file_id, val_file_id)

    if args.wait:
        _poll_job(client, job_id, args.poll_seconds)
    else:
        print("Run with --wait to monitor status continuously.")
        print(f"You can later check status using this job id: {job_id}")


if __name__ == "__main__":
    main()
