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
pip install -r requirements.txt   # edge-tts dahil (insanı TTS sesi için gerekli)
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

**Not:** İnsanı/doğal ses (Edge TTS) için `edge-tts` paketi yüklü olmalı. Backend’i çalıştırdığınız aynı Python ortamında `pip install edge-tts` veya `pip install -r requirements.txt` çalıştırın.

## Running the mobile (Flutter) client

```bash
cd mobile && flutter pub get && flutter run -d chrome
```

## LLM davranışını özelleştirme (few-shot / fine-tuning)

Chatbot cevaplarını kendi istediğiniz forma getirmek için:

1. **Few-shot örnekleri**  
   `data/llm_fine_tune_examples.json` dosyasına örnek diyaloglar ekleyin: her biri `user` (kullanıcı mesajı) ve `assistant_json` (beklenen JSON cevap) içermeli. LLM bu örnekleri görerek benzer formatta cevap verir.

2. **Ek talimatlar**  
   `.env` içinde `LLM_CUSTOM_INSTRUCTIONS="..."` ile sistem prompt’a ek talimatlar ekleyebilirsiniz (ör. “Cevaplar hep 2 cümleyi geçmesin.”).

3. **Fine-tuning veri seti export**  
   Örnekleri OpenAI/chat fine-tune formatında JSONL’e dönüştürmek için:
   ```bash
   PYTHONPATH=src python scripts/export_llm_finetune_data.py
   ```
   Çıktı: `data/llm_finetune_train.jsonl`. İleride gerçek model fine-tuning (OpenAI API veya LoRA) yapmak için bu dosyayı kullanabilirsiniz.
