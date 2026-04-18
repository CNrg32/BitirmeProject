#!/usr/bin/env python3
"""
Train a LoRA adapter on AWS (EC2/SageMaker-compatible).

Input format:
  JSONL with {"messages":[{"role":"system|user|assistant","content":"..."}]}

Default input:
  data/llm_groq_lora_train.jsonl

Example:
  python scripts/train_lora_aws.py \
    --train-file data/llm_groq_lora_train.jsonl \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir out_models/lora_adapter_aws
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    train_file: str
    base_model: str
    output_dir: str
    max_seq_len: int
    lr: float
    epochs: int
    batch_size: int
    grad_accum: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_4bit: bool


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train LoRA adapter for emergency triage chat model.")
    parser.add_argument("--train-file", default="data/llm_groq_lora_train.jsonl")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="out_models/lora_adapter_aws")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit loading with bitsandbytes (recommended on memory-limited GPUs).",
    )
    args = parser.parse_args()
    return TrainConfig(
        train_file=args.train_file,
        base_model=args.base_model,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
    )


def _messages_to_text(messages: List[Dict[str, str]], tokenizer: AutoTokenizer) -> str:
    """
    Convert chat messages into a single training text.
    Uses tokenizer chat template when available; otherwise fallback format.
    """
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        lines = []
        for m in messages:
            role = (m.get("role") or "user").upper()
            content = m.get("content") or ""
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


def _load_jsonl_dataset(train_file: str) -> Dataset:
    path = Path(train_file)
    if not path.exists():
        raise FileNotFoundError(f"Train file not found: {path}")
    ds = load_dataset("json", data_files=str(path), split="train")
    if "messages" not in ds.column_names:
        raise ValueError("Training JSONL must contain a 'messages' field.")
    return ds


def _tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
    def _map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        text = _messages_to_text(example["messages"], tokenizer)
        tok = tokenizer(text, truncation=True, max_length=max_len)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = ds.map(_map_fn, remove_columns=ds.column_names)
    return tokenized


def _build_model_and_tokenizer(cfg: TrainConfig):
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if cfg.use_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    model.config.use_cache = False

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    model, tokenizer = _build_model_and_tokenizer(cfg)
    ds = _load_jsonl_dataset(cfg.train_file)
    tokenized_ds = _tokenize_dataset(ds, tokenizer, cfg.max_seq_len)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        gradient_checkpointing=True,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metadata_path = Path(cfg.output_dir) / "training_metadata.json"
    metadata = {
        "base_model": cfg.base_model,
        "train_file": cfg.train_file,
        "max_seq_len": cfg.max_seq_len,
        "epochs": cfg.epochs,
        "learning_rate": cfg.lr,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "use_4bit": cfg.use_4bit,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"LoRA training completed. Adapter saved to: {cfg.output_dir}")
    print("For Groq upload, zip adapter_model.safetensors + adapter_config.json")


if __name__ == "__main__":
    main()
