from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


PROMPT_HEADER = (
    "You are a professional emergency dispatcher assistant. "
    "Return ONLY a valid JSON object with keys: "
    "response_text, extracted_slots, triage_level, category, is_complete, red_flags."
)


def _normalize_history(history: Any) -> List[Dict[str, str]]:
    if not isinstance(history, list):
        return []
    normalized: List[Dict[str, str]] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "user")).strip().lower()
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        normalized.append({"role": role, "text": text})
    return normalized


def _format_prompt(example: Dict[str, Any]) -> str:
    language = str(example.get("language", "en")).strip() or "en"
    turns = _normalize_history(example.get("history", []))

    convo_lines: List[str] = []
    for t in turns:
        speaker = "USER" if t["role"] == "user" else "ASSISTANT"
        convo_lines.append(f"{speaker}: {t['text']}")

    convo_text = "\n".join(convo_lines).strip()
    return (
        f"{PROMPT_HEADER}\n"
        f"Language: {language}\n"
        "Conversation:\n"
        f"{convo_text}\n\n"
        "Generate the next assistant JSON output:"
    )


def _format_target(example: Dict[str, Any]) -> str:
    target = example.get("target", {})
    if isinstance(target, str):
        return target.strip()
    if not isinstance(target, dict):
        target = {}

    payload = {
        "response_text": str(target.get("response_text", "")),
        "extracted_slots": target.get("extracted_slots", {}) if isinstance(target.get("extracted_slots", {}), dict) else {},
        "triage_level": str(target.get("triage_level", "URGENT")),
        "category": str(target.get("category", "other")),
        "is_complete": bool(target.get("is_complete", False)),
        "red_flags": target.get("red_flags", []) if isinstance(target.get("red_flags", []), list) else [],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _build_tokenize_fn(tokenizer: AutoTokenizer, max_source_length: int, max_target_length: int):
    def _tokenize(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        prompts = [_format_prompt({k: batch[k][i] for k in batch}) for i in range(len(batch["history"]))]
        targets = [_format_target({k: batch[k][i] for k in batch}) for i in range(len(batch["history"]))]

        model_inputs = tokenizer(
            prompts,
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _tokenize


def _load_splits(train_file: Path, val_file: Path | None, val_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    train_data = load_dataset("json", data_files={"train": str(train_file)})["train"]

    if val_file and val_file.exists():
        val_data = load_dataset("json", data_files={"validation": str(val_file)})["validation"]
        return train_data, val_data

    split = train_data.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a seq2seq model for emergency chatbot JSON outputs.")
    parser.add_argument("--train-file", type=Path, required=True, help="Path to JSONL train file.")
    parser.add_argument("--val-file", type=Path, default=None, help="Optional JSONL validation file.")
    parser.add_argument("--output-dir", type=Path, default=Path("out_models/chatbot_finetuned"))
    parser.add_argument("--model-name", type=str, default="google/flan-t5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_file.exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = _load_splits(
        train_file=args.train_file,
        val_file=args.val_file,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    logger.info("Train samples: %d | Validation samples: %d", len(train_ds), len(val_ds))
    logger.info("Loading model/tokenizer: %s", args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenize_fn = _build_tokenize_fn(
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    use_fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        fp16=use_fp16,
        report_to="none",
        seed=args.seed,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=collator,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    logger.info("Saving model artifacts...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metadata = {
        "base_model": args.model_name,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file) if args.val_file else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
    }
    (args.output_dir / "finetune_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Done. Fine-tuned model saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
