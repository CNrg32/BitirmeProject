# Bitirme Project – build and test
.PHONY: test test-cov install-test

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
