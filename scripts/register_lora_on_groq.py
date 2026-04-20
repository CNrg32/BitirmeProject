#!/usr/bin/env python3
"""
Upload a prepared LoRA adapter zip to Groq and register it as a fine-tuned model.

Prerequisites:
  - Groq LoRA access enabled for the organization.
  - GROQ_API_KEY set in the local environment.
  - A zip created by scripts/package_lora_for_groq.py containing exactly:
      adapter_model.safetensors
      adapter_config.json

Usage:
  python scripts/register_lora_on_groq.py \
    --zip out_models/lora_adapter_aws/groq_adapter.zip \
    --base-model llama-3.3-70b-versatile \
    --name bitirme-emergency-triage \
    --output-env .env.groq-finetuned
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


GROQ_FILES_URL = "https://api.groq.com/openai/v1/files"
GROQ_FINE_TUNINGS_URL = "https://api.groq.com/v1/fine_tunings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a LoRA adapter zip on Groq.")
    parser.add_argument("--zip", required=True, help="Groq adapter zip path.")
    parser.add_argument("--base-model", required=True, help="Groq base model used for the LoRA.")
    parser.add_argument("--name", required=True, help="Name for the registered fine-tuned model.")
    parser.add_argument(
        "--output-env",
        default="",
        help="Optional env file path to write GROQ_FINE_TUNED_MODEL after registration.",
    )
    return parser.parse_args()


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is required. Set it in your local shell, not in chat.")
    return key


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc)


def _request_json(req: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = _read_error(exc)
        raise RuntimeError(f"Groq API error {exc.code}: {body}") from exc


def _upload_file(zip_path: Path, api_key: str) -> str:
    boundary = f"----bitirme-groq-{int(time.time() * 1000)}"
    file_bytes = zip_path.read_bytes()
    parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            "fine_tuning\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{zip_path.name}"\r\n'
            "Content-Type: application/zip\r\n\r\n"
        ).encode("utf-8"),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        GROQ_FILES_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    payload = _request_json(req)
    file_id = payload.get("id")
    if not file_id:
        raise RuntimeError(f"Could not read uploaded file id from Groq response: {payload}")
    print(f"Uploaded {zip_path} -> {file_id}")
    return str(file_id)


def _register_lora(api_key: str, file_id: str, base_model: str, name: str) -> dict[str, Any]:
    body = json.dumps(
        {
            "input_file_id": file_id,
            "base_model": base_model,
            "name": name,
            "type": "lora",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        GROQ_FINE_TUNINGS_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    payload = _request_json(req)
    print("Registered LoRA on Groq:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def _extract_model_id(payload: dict[str, Any]) -> str:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    for key in ("fine_tuned_model", "model", "id", "name"):
        value = data.get(key) if isinstance(data, dict) else None
        if value:
            return str(value)
    return ""


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"Adapter zip not found: {zip_path}")

    api_key = _api_key()
    file_id = _upload_file(zip_path, api_key)
    payload = _register_lora(api_key, file_id, args.base_model, args.name)
    model_id = _extract_model_id(payload)

    if args.output_env and model_id:
        out = Path(args.output_env)
        out.write_text(f'GROQ_FINE_TUNED_MODEL="{model_id}"\n', encoding="utf-8")
        print(f"Wrote model env file: {out}")
    elif not model_id:
        print("Could not infer fine-tuned model id automatically; copy it from the response.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
