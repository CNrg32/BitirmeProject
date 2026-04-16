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

4. **Groq LoRA/SFT için veri export (önerilen)**  
   Groq üzerinde kullanacağınız adapter/fine-tuned model için chat-format JSONL üretmek:
   ```bash
   PYTHONPATH=src python scripts/export_groq_lora_data.py
   ```
   Çıktı: `data/llm_groq_lora_train.jsonl`

## Groq ile model özelleştirme akışı

Bu repo Groq'u inference için kullanır. Groq tarafında doğrudan eğitim yerine,
LoRA/SFT adapter'ı dışarıda eğitip Groq'a model kimliği (model id) olarak tanımlarsınız.

1. Eğitim verisini üret:
```bash
python scripts/generate_llm_finetune_examples.py --count 180
PYTHONPATH=src python scripts/export_groq_lora_data.py
```

2. Dış pipeline'da (PEFT/Unsloth vb.) adapter eğit ve Groq hesabına yükle.

3. Uygulamayı Groq custom model ile çalıştır:
```bash
export GROQ_API_KEY="gsk_..."
export GROQ_FINE_TUNED_MODEL="ft:your-groq-model-id"
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Not: `GROQ_FINE_TUNED_MODEL` set edilmezse sırasıyla `GROQ_MODEL`, sonra varsayılan model kullanılır.

4. Modeli JSON uyumu açısından hızlı test et:
```bash
GROQ_API_KEY="gsk_..." GROQ_FINE_TUNED_MODEL="ft:your-groq-model-id" \
PYTHONPATH=src python scripts/eval_groq_json_mode.py
```

## Legacy local fine-tuning notes

The section below is a previous local fine-tuning flow and may not exist in every branch/version of this repository.

Expected JSONL sample format (`data/labels/chatbot_finetune_template.jsonl`):

```json
{"language":"en","history":[{"role":"user","text":"My father is not breathing."}],"target":{"response_text":"Is he conscious right now?","extracted_slots":{"chief_complaint":"not breathing"},"triage_level":"CRITICAL","category":"medical","is_complete":true,"red_flags":["not breathing"]}}
```

Run fine-tuning:

```bash
make finetune-chatbot
```

Build training data automatically from existing CSV labels:

```bash
make build-chatbot-data
```

Then train on generated files:

```bash
PYTHONPATH=src python scripts/train_chatbot_finetune.py \
	--train-file data/labels/chatbot_finetune_train.jsonl \
	--val-file data/labels/chatbot_finetune_val.jsonl \
	--output-dir out_models/chatbot_finetuned
```

Custom run (different dataset/model/output):

```bash
PYTHONPATH=src python scripts/train_chatbot_finetune.py \
	--train-file data/labels/your_chatbot_train.jsonl \
	--val-file data/labels/your_chatbot_val.jsonl \
	--model-name google/flan-t5-small \
	--output-dir out_models/chatbot_finetuned_v2
```

Use the fine-tuned model in backend (offline/local provider):

```bash
# PowerShell
$env:LOCAL_CHATBOT_MODEL_DIR="out_models/chatbot_finetuned"
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Provider order is now:
1. `LOCAL_CHATBOT_MODEL_DIR` (local fine-tuned model)
2. `GROQ_API_KEY`
3. `GEMINI_API_KEY` / `GOOGLE_API_KEY`
