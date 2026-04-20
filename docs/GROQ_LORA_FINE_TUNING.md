# Groq LoRA Fine-Tuning Flow

Groq should be treated as the inference/deployment target for LoRA adapters.
The adapter is trained outside Groq, then uploaded to Groq if the organization
has LoRA access enabled.

## 1. Validate Dataset

```powershell
py scripts\validate_groq_lora_data.py data\llm_groq_lora_train_split.jsonl
py scripts\validate_groq_lora_data.py data\llm_groq_lora_val.jsonl
```

Current checked split:

```text
train examples: 1080
validation examples: 120
required JSON schema: ok
```

## 2. Train LoRA Externally

Together AI is the current external trainer in this repo.

Do not paste API keys into chat or commit them to git.

```powershell
$env:TOGETHER_API_KEY="..."

py -m pip install -r requirements-together-finetune.txt
```

Or install fine-tune-only dependencies into an isolated local folder:

```powershell
py -m pip install --target .finetune_vendor -r requirements-together-finetune.txt
$env:PYTHONPATH="$PWD\.finetune_vendor;$PWD\src"
```

```powershell
py scripts\train_lora_together.py `
  --train-file data\llm_groq_lora_train_split.jsonl `
  --validation-file data\llm_groq_lora_val.jsonl `
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference `
  --epochs 3 `
  --batch-size 8 `
  --learning-rate 1e-5 `
  --suffix bitirme-emergency-triage `
  --lora `
  --lora-r 16 `
  --lora-alpha 32 `
  --wait
```

The LoRA must be trained against the exact base model version that Groq will
serve. If Groq uses a different base model, retrain the adapter for that model.
Use rank 8 or 16 for Groq transfer. Higher ranks can create adapter archives
that are too large for Groq's upload path.

## 3. Package Adapter For Groq

Groq expects a zip containing exactly:

```text
adapter_model.safetensors
adapter_config.json
```

Package the adapter:

```powershell
py scripts\package_lora_for_groq.py `
  --adapter-dir out_models\lora_adapter_aws `
  --output out_models\lora_adapter_aws\groq_adapter.zip
```

Prefer LoRA rank 8 or 16 for lower cold-start latency.

## 4. Register Adapter On Groq

This requires Groq LoRA access for the organization.

```powershell
$env:GROQ_API_KEY="..."

py scripts\register_lora_on_groq.py `
  --zip out_models\lora_adapter_aws\groq_adapter.zip `
  --base-model llama-3.1-8b-instant `
  --name bitirme-emergency-triage `
  --output-env .env.groq-finetuned
```

If registration succeeds, copy the produced `GROQ_FINE_TUNED_MODEL` value into
your local `.env` file.

Groq requires the adapter to be trained against the exact supported base model.
At the time this flow was prepared, Groq LoRA support lists `llama-3.1-8b-instant`
as the supported public-cloud base model.

## 5. Use In Backend

The backend already reads:

```text
GROQ_API_KEY
GROQ_FINE_TUNED_MODEL
GROQ_MODEL
```

Priority is:

```text
GROQ_FINE_TUNED_MODEL -> GROQ_MODEL -> llama-3.3-70b-versatile
```

Restart backend after changing `.env`.

## 6. Evaluate JSON Compliance

```powershell
$env:GROQ_API_KEY="..."
$env:GROQ_FINE_TUNED_MODEL="..."

py scripts\eval_groq_json_mode.py
```

The first pass target is:

```text
JSON parse success: >= 98%
required schema success: >= 95%
```
