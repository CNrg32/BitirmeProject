#!/usr/bin/env bash
# aws_bootstrap.sh
# ================
# EC2 (Deep Learning OSS Nvidia AMI, Ubuntu 22.04) uzerinde XLM-R triage
# modelini uctan uca eger: clone + venv + data prep + train + S3 upload.
#
# Spot instance icin:
#   - SIGTERM (2 dk warning) geldiginde log + partial model S3'e yuklenir.
#   - IMDSv2 /spot/instance-action endpoint'i arkaplanda izlenir.
#   - Tekrar calistirildiginda idempotent: mevcut venv / clone kullanilir.
#
# Kullanim (ornekler):
#   # 1) Repoyu henuz clone etmediysen, bu scripti wget ile al ve calistir:
#   wget https://raw.githubusercontent.com/<USER>/<REPO>/main/scripts/aws_bootstrap.sh
#   chmod +x aws_bootstrap.sh
#   export REPO_URL="https://github.com/<USER>/<REPO>.git"
#   export S3_BUCKET="s3://<SENIN-BUCKETIN>/triage_xlmr"
#   ./aws_bootstrap.sh
#
#   # 2) Repoyu zaten clone ettiysen (ornek: git clone sonrasi):
#   export S3_BUCKET="s3://<SENIN-BUCKETIN>/triage_xlmr"
#   bash scripts/aws_bootstrap.sh
#
# Env degiskenleri:
#   REPO_URL      (opsiyonel) Git clone URL. Script repo disinda calistiysa gerekir.
#   REPO_BRANCH   (opsiyonel, default: main)
#   PROJECT_DIR   (opsiyonel, default: $HOME/BitirmeProject)
#   S3_BUCKET     (zorunlu)  Model + log cikti yolu. Or: s3://my-bucket/triage_xlmr
#   BATCH_SIZE    (default: 16)
#   MAX_LEN       (default: 384)
#   EPOCHS        (default: 4)
#   LR            (default: 2e-5)
#   PATIENCE      (default: 2)

set -euo pipefail

# --------------------- Parametreler ---------------------
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-main}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/BitirmeProject}"
S3_BUCKET="${S3_BUCKET:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LEN="${MAX_LEN:-384}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-2e-5}"
PATIENCE="${PATIENCE:-2}"

LOG_FILE="${PROJECT_DIR}/training_log.txt"
BOOT_LOG="/tmp/aws_bootstrap.log"

# --------------------- Yardimci ---------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$BOOT_LOG"
}

fail() {
    log "FAIL: $*"
    exit 1
}

upload_artifacts() {
    local tag="${1:-final}"
    if [[ -z "$S3_BUCKET" ]]; then
        log "S3_BUCKET bos, upload atlandi."
        return 0
    fi
    log "S3'e yukleniyor (tag=$tag) -> $S3_BUCKET/"
    aws s3 cp "$BOOT_LOG" "$S3_BUCKET/logs/bootstrap_${tag}.log" || true
    if [[ -f "$LOG_FILE" ]]; then
        aws s3 cp "$LOG_FILE" "$S3_BUCKET/logs/training_${tag}.log" || true
    fi
    local model_dir="${PROJECT_DIR}/out_models/triage_xlmr"
    if [[ -d "$model_dir" ]]; then
        aws s3 cp --recursive "$model_dir" "$S3_BUCKET/model_${tag}/" || true
    fi
    log "S3 upload tamamlandi (tag=$tag)."
}

on_term() {
    log "SIGTERM alindi (muhtemelen spot interruption). Partial artifact S3'e gidiyor..."
    upload_artifacts "interrupted_$(date +%s)"
    exit 130
}
trap on_term SIGTERM SIGINT

# Spot interruption notice'i arkaplanda izle (IMDSv2)
monitor_spot() {
    local token
    while true; do
        token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
                -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || echo "")
        if [[ -n "$token" ]]; then
            local code
            code=$(curl -sS -o /dev/null -w "%{http_code}" \
                   -H "X-aws-ec2-metadata-token: $token" \
                   "http://169.254.169.254/latest/meta-data/spot/instance-action" 2>/dev/null || echo "000")
            if [[ "$code" == "200" ]]; then
                log "Spot interruption notice algilandi -> parent'a SIGTERM gonderiyorum."
                kill -TERM "$PARENT_PID" 2>/dev/null || true
                break
            fi
        fi
        sleep 5
    done
}

# --------------------- GPU Kontrol ---------------------
log "===== AWS bootstrap baslangic ====="
log "PROJECT_DIR=$PROJECT_DIR  S3_BUCKET=${S3_BUCKET:-<bos>}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi yok. Yanlis AMI? (Deep Learning OSS Nvidia AMI secili mi?)"
fi
log "GPU bilgileri:"
nvidia-smi | tee -a "$BOOT_LOG"

if ! command -v aws >/dev/null 2>&1; then
    log "aws CLI bulunamadi, yukleniyor..."
    sudo apt-get update -y >/dev/null
    sudo apt-get install -y unzip curl >/dev/null
    curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    (cd /tmp && unzip -q awscliv2.zip && sudo ./aws/install --update) || true
fi

# --------------------- Repo ---------------------
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
    [[ -z "$REPO_URL" ]] && fail "PROJECT_DIR'de repo yok ve REPO_URL bos."
    log "Repo clone: $REPO_URL (branch=$REPO_BRANCH)"
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$PROJECT_DIR"
else
    log "Repo zaten mevcut, guncelleniyor..."
    (cd "$PROJECT_DIR" && git fetch --depth 1 origin "$REPO_BRANCH" && git reset --hard "origin/$REPO_BRANCH") || true
fi

cd "$PROJECT_DIR"

# --------------------- Venv + deps ---------------------
if [[ ! -d ".venv" ]]; then
    log "Venv olusturuluyor..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

log "Pip guncelleniyor ve requirements yukleniyor..."
pip install --upgrade pip >> "$BOOT_LOG" 2>&1
pip install -r requirements.txt >> "$BOOT_LOG" 2>&1

# Torch CUDA dogrulamasi
python - <<'PY' | tee -a "$BOOT_LOG"
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

# --------------------- Data prep ---------------------
log "Merge dataset..."
python scripts/merge_train_dataset.py >> "$BOOT_LOG" 2>&1

log "History dataset hazirlik..."
python scripts/prepare_triage_history_dataset.py >> "$BOOT_LOG" 2>&1

# --------------------- Training ---------------------
PARENT_PID=$$
monitor_spot &
MONITOR_PID=$!

log "Training basliyor: epochs=$EPOCHS batch=$BATCH_SIZE max_len=$MAX_LEN lr=$LR patience=$PATIENCE"
set +e
python scripts/train_triage_xlmr.py \
    --batch-size "$BATCH_SIZE" \
    --max-len "$MAX_LEN" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --patience "$PATIENCE" \
    2>&1 | tee "$LOG_FILE"
TRAIN_RC=${PIPESTATUS[0]}
set -e

kill "$MONITOR_PID" 2>/dev/null || true

if [[ $TRAIN_RC -ne 0 ]]; then
    log "Training basarisiz oldu (rc=$TRAIN_RC). Artifact yine de yukleniyor..."
    upload_artifacts "failed_rc${TRAIN_RC}"
    exit "$TRAIN_RC"
fi

log "Training basarili, final upload..."
upload_artifacts "final"
log "===== AWS bootstrap tamam ====="
