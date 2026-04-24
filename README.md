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

## Local triage model (XLM-R)

Triaj sınıflandırması artık OpenAI fine-tuned modeli yerine yerel XLM-RoBERTa multi-task modeli ile çalışabilir. Model dialog history'sini `[USER] ... [ASSISTANT] ... [USER] ...` formatında alır ve `triage_level` + `category` + `red_flag_present` tahmin eder. `src/services/triage_local_service.py` conservative karar kurallarını (güven eşiği, minimum turn sayısı) uygular.

### Eğitim verisini hazırla

```bash
# 1) Yeni hafif/şüpheli şikayet sentetik veri setlerini üret
python scripts/generate_minor_complaints.py

# 2) Tüm veri kaynaklarını birleştirip train/val/test parquet'lerini yenile
python scripts/merge_train_dataset.py

# 3) History-aware prefix örneklerini + JSONL dialog örneklerini üret
python scripts/prepare_triage_history_dataset.py
# -> output/out_dataset/triage_history_{train,val,test}.parquet
```

### Modeli eğit

```bash
# Varsayılan: xlm-roberta-base, max_len=384, epochs=4, lr=2e-5
python scripts/train_triage_xlmr.py
# -> out_models/triage_xlmr/ (model.pt + config.json + tokenizer/)
```

Mac'te MPS otomatik seçilir; CUDA yoksa CPU'ya düşer. Küçük RAM için `--batch-size 4 --max-len 256`.

### Modeli değerlendir

```bash
python scripts/eval_triage_local.py            # genel test + stress test
python scripts/eval_triage_local.py --show-raw # conservative rule öncesi ham çıktı
```

### Backend'i yerel modelle çalıştır

`.env` dosyasına ekle:

```env
TRIAGE_BACKEND=local                # local | openai (varsayılan: model varsa local)
TRIAGE_CRITICAL_THRESHOLD=0.70      # ham prob eşiği, altında URGENT'a düşer
TRIAGE_URGENT_THRESHOLD=0.55        # altında NON_URGENT'a düşer
TRIAGE_MIN_TURNS_CRITICAL=2         # ilk user turlarında red flag yoksa CRITICAL engeli
TRIAGE_REDFLAG_THRESHOLD=0.5        # binary red-flag head karar sınırı
```

`src/mvp_regex_dictionary.json` + `src/mvp_rules.py` safety-net'i aktif kalır; gerçek red-flag ifadeleri (örn. "nefes alamıyor") her durumda CRITICAL'a zorlar.

**Not:** İnsanı/doğal ses (Edge TTS) için `edge-tts` paketi yüklü olmalı. Backend’i çalıştırdığınız aynı Python ortamında `pip install edge-tts` veya `pip install -r requirements.txt` çalıştırın.

### ASR / TTS / çeviri ortam değişkenleri (performans ve tutarlılık)

| Değişken | Açıklama |
|----------|----------|
| `ASR_MODEL_SIZE` | Whisper modeli: `tiny`, `base`, `small`, `medium`, `large-v3`, … (varsayılan **`small`**, zayıf CPU için `base`) |
| `ASR_DEVICE` | `cpu`, `cuda`, `mps` (boş = CPU) |
| `ASR_COMPUTE_TYPE` | `int8`, `float16` (GPU’da genelde `float16`) |
| `ASR_BEAM_SIZE` | Çözümleme ışını; daha yüksek = daha tutarlı, daha yavaş (varsayılan `5`) |
| `ASR_CONDITION_ON_PREVIOUS_TEXT` | `true`/`false` — segmentler arası bağlam (varsayılan `false`) |
| `ASR_VAD_FILTER` | `true`/`false` — sessiz bölgeleri atla (varsayılan `true`) |
| `ASR_PREPROCESS` | `true`/`false` — `ffmpeg` varsa 16 kHz mono WAV’a çevir (varsayılan `true`) |
| `TRANSLATION_BACKEND` | `deep_translator` (varsayılan), `deepl`, `google`, `local` (Marian tr↔en) |
| `DEEPL_API_KEY` | `TRANSLATION_BACKEND=deepl` için; isteğe `DEEPL_USE_FREE_API=1` |
| `GOOGLE_TRANSLATE_API_KEY` | `TRANSLATION_BACKEND=google` için (veya `TRANSLATE_GOOGLE_API_KEY`) |
| `TTS_CACHE_MAX` | LRU önbellekte tutulacak maksimum farklı (metin, dil) sayısı (`0` = kapalı) |
| `TTS_EDGE_MAX_RETRIES` | Edge TTS ağ hatalarında yeniden deneme (varsayılan `3`) |
| `TTS_EDGE_RETRY_BASE_S` | Üstel geri çekilme tabanı saniye (varsayılan `0.35`) |
| `TTS_EDGE_VOICE_<LANG>` | Edge ses adını geçersiz kıl; örn. `TTS_EDGE_VOICE_EN=en-US-GuyNeural`, `TTS_EDGE_VOICE_ZH_CN=...` |
| `TTS_EDGE_RATE` / `TTS_EDGE_PITCH` / `TTS_EDGE_VOLUME` | Edge prosodi (örn. `TTS_EDGE_RATE=-6%` daha doğal tempo; `+0%` ile sıfırla) |
| `TTS_PROVIDER` | `auto` (Google anahtarı varsa önce Google), `edge` (yalnız Edge), `google` (Google sonra Edge yedeği) |
| `GOOGLE_TTS_API_KEY` | [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech) API anahtarı — TR/EN için Neural2/Wavenet, Edge’den genelde daha tutarlı |
| `TTS_GOOGLE_VOICE_EN` / `TTS_GOOGLE_VOICE_TR` | Örn. `en-US-Neural2-F`, `tr-TR-Neural2-A` |
| `TTS_GOOGLE_SPEAKING_RATE` / `TTS_GOOGLE_PITCH` | Google TTS konuşma hızı (1.0) ve perde (0.0) |
| `USE_OPENAI_FINAL_REPORT` | `true` ise oturum **son raporunu** yalnızca bu adımda OpenAI (GPT) üretir; diyalog hâlâ `GROQ_API_KEY` ile Groq’ta kalır. `OPENAI_API_KEY` gerekir. |
| `OPENAI_FINAL_REPORT_MODEL` | İsteğe bağlı; boşsa `gpt-4.1-mini-2025-04-14` kullanılır (diyalog modeli `OPENAI_MODEL` ile karıştırılmaz). |
| `OPENAI_FINAL_REPORT_MAX_TOKENS` | Son rapor üst sınırı (varsayılan `1200`). |

Daha doğal **TTS** için: `GOOGLE_TTS_API_KEY` ile `TTS_PROVIDER=auto` (veya `google`) kullanın; sadece Edge kullanacaksanız varsayılan sesler `en-US-AriaNeural` / `tr-TR-EmelNeural` ve hafif yavaşlatma (`TTS_EDGE_RATE=-6%`) uygulanır. **Çeviri** kalitesi için üretimde `TRANSLATION_BACKEND=deepl` veya `google` + resmi API anahtarı önerilir.

`ffmpeg` sistem PATH’inde değilse ön-işleme atlanır; üretimde mobil/webm kayıtları için kurulması önerilir.

## Running the mobile (Flutter) client

```bash
cd mobile && flutter pub get && flutter run -d chrome
```

## LLM (Groq diyalog + OpenAI fine-tune triage)

Tam **LLM** modu için birlikte gerekir: **`GROQ_API_KEY`** (kullanıcıya gösterilen yanıtlar Groq’ta), **`OPENAI_API_KEY`** ve **`OPENAI_FINE_TUNED_MODEL`** (**triage**: her turda güncel konuşma ile kategori / `triage_level` / `red_flags`; rapor ve dispatch mantığı bu çıktıya dayanır). İkisi eksikse orchestrator kural tabanlı akışa geçer.

İsteğe bağlı: `GROQ_MODEL`, `GROQ_FINE_TUNED_MODEL`, `OPENAI_TRIAGE_MAX_TOKENS`, `OPENAI_TRIAGE_MAX_HISTORY_TURNS`, `OPENAI_FINE_TUNED_FAST`.

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

## Together AI ile model özelleştirme akışı

Bu repoda veri üretimi lokal yapılır, fine-tune işlemi Together AI API üzerinden tetiklenir.
Groq entegrasyonu kullanacaksanız model uyumluluğunu ayrıca doğrulamanız gerekir.

1. Ortamı kur:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-together-finetune.txt
```

2. Eğitim verisini üret:
```bash
python scripts/generate_llm_finetune_examples.py --count 1200
PYTHONPATH=src python scripts/export_groq_lora_data.py
```

3. Together LoRA fine-tune job oluştur:
```bash
export TOGETHER_API_KEY="..."
PYTHONPATH=src python scripts/train_lora_together.py \
  --train-file data/llm_groq_lora_train.jsonl \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora \
  --wait
```

4. Job tamamlandıktan sonra Together model adını/ID'sini alın.
   - Output adını Together dashboard veya API job detayından görebilirsiniz.

5. JSON uyumu kontrolü için mevcut script ile hızlı test yapın (kendi model id'nizi verin):
```bash
GROQ_API_KEY="gsk_..." GROQ_FINE_TUNED_MODEL="ft:your-model-id" \
PYTHONPATH=src python scripts/eval_groq_json_mode.py
```

Not:
- Bu repodaki backend diyalog için Groq, triage kararı için OpenAI fine-tune (`OPENAI_FINE_TUNED_MODEL`) kullanır.
- Together'da fine-tune edilen modeli doğrudan backend'e bağlamak isterseniz ek provider entegrasyonu gerekir.
- Eski AWS tabanlı yerel LoRA akışı için `scripts/train_lora_aws.py` scripti repoda tutulmuştur.

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

Production LLM stack (when keys are set): Groq for dialog; OpenAI fine-tuned model for triage only. Legacy local provider notes may still refer to `LOCAL_CHATBOT_MODEL_DIR` in older scripts.
