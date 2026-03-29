# Bitirme Project – build and test
.PHONY: test test-cov install-test build-chatbot-data finetune-chatbot

install-test:
	pip install -r requirements-test.txt -q

test:
	PYTHONPATH=src pytest tests/ -v --tb=short

test-cov:
	PYTHONPATH=src pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html --cov-config=.coveragerc --cov-fail-under=70

test-quick:
	PYTHONPATH=src pytest tests/ -q --tb=line

report-docx:
	python scripts/md_to_docx.py

finetune-chatbot: build-chatbot-data
	PYTHONPATH=src python scripts/train_chatbot_finetune.py --train-file data/labels/chatbot_finetune_train.jsonl --val-file data/labels/chatbot_finetune_val.jsonl --output-dir out_models/chatbot_finetuned

build-chatbot-data:
	PYTHONPATH=src python scripts/build_chatbot_finetune_jsonl.py --output-train data/labels/chatbot_finetune_train.jsonl --output-val data/labels/chatbot_finetune_val.jsonl
