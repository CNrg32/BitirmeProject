# Emergency Triage Assistant (Bitirme Project)

Multilingual emergency triage API and Flutter client: text/voice/image input, triage (CRITICAL/URGENT/NON_URGENT), category (medical/crime/fire/other), slot extraction, and report generation.

## How to run tests

From the **project root**:

```bash
# Install test dependencies (once)
pip install -r requirements-test.txt

# Run all tests
make test
# or
PYTHONPATH=src pytest tests/ -v --tb=short

# Run tests with coverage report (target 70%; see .coveragerc)
make test-cov
# or
PYTHONPATH=src pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-config=.coveragerc --cov-fail-under=70
```

- **Unit tests:** `tests/unit/`  
- **Integration tests:** `tests/integration/`  
- **System / E2E:** `tests/system/`  
- **Performance:** `tests/performance/`  
- **User scenarios:** `tests/user_scenario/`

External services (LLM, TTS, ASR, translation, image model) are mocked so tests are deterministic and do not require API keys or GPU.

Test plan report: `reports/test_plan_report.md`.

## Running the backend

```bash
pip install -r requirements.txt
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

## Running the mobile (Flutter) client

```bash
cd mobile && flutter pub get && flutter run -d chrome
```
