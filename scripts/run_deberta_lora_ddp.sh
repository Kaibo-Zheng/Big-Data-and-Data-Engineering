#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MAX_LENGTH=128
MAX_CHARS=4000
EPOCHS=1
LR=1e-4
HEAD_LR=8e-4
WEIGHT_DECAY=0.01
HEAD_HIDDEN_SIZE=0
HEAD_DROPOUT=0.2
MULTI_SAMPLE_DROPOUT=5
GRAD_ACCUM=16
BATCH_SIZE=1
EVAL_BATCH_SIZE=2
NUM_WORKERS=0
PREFETCH_FACTOR=2
VALID_SIZE=0.1
SAMPLE=0
PREDICT_LIMIT=0
GRADIENT_CHECKPOINTING=1
LORA_R=8
LORA_ALPHA=16
WANDB=0
WANDB_PROJECT="ccks2026"
WANDB_RUN_NAME=""
MODEL_DIR="model"
OUTPUT_DIR="ckpt"
SUBMISSION_DIR="result/submissions"
LOG_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --max-chars) MAX_CHARS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --head-lr) HEAD_LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --head-hidden-size) HEAD_HIDDEN_SIZE="$2"; shift 2 ;;
    --head-dropout) HEAD_DROPOUT="$2"; shift 2 ;;
    --multi-sample-dropout) MULTI_SAMPLE_DROPOUT="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --prefetch-factor) PREFETCH_FACTOR="$2"; shift 2 ;;
    --valid-size) VALID_SIZE="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --predict-limit) PREDICT_LIMIT="$2"; shift 2 ;;
    --no-gradient-checkpointing) GRADIENT_CHECKPOINTING=0; shift ;;
    --lora-r) LORA_R="$2"; shift 2 ;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
    --wandb) WANDB=1; shift ;;
    --wandb_project|--wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_run_name|--wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --submission-dir) SUBMISSION_DIR="$2"; shift 2 ;;
    --log-path) LOG_PATH="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/run_deberta_lora_ddp.sh [options]

Single-node multi-GPU DeBERTa LoRA workflow. Training uses PyTorch DDP;
prediction runs once on a single process to avoid duplicate submission writes.

Options:
  --nproc-per-node N     Number of GPU processes. Defaults to nvidia-smi GPU count.
  --max-length N         Token length, default 128
  --max-chars N          Text trim length, default 4000
  --epochs N             Epochs, default 1
  --lr FLOAT             LoRA learning rate, default 1e-4
  --head-lr FLOAT        Classifier/pooler learning rate, default 8e-4
  --weight-decay FLOAT   Weight decay, default 0.01
  --head-hidden-size N   MLP hidden size, default backbone hidden size
  --head-dropout FLOAT   Head dropout, default 0.2
  --multi-sample-dropout N
                         Number of dropout samples during training, default 5
  --grad-accum N         Gradient accumulation per process, default 16
  --batch-size N         Per-GPU train batch size, default 1
  --eval-batch-size N    Eval batch size, default 2
  --num-workers N        DataLoader workers per process, default 0
  --prefetch-factor N    DataLoader prefetch factor when workers > 0, default 2
  --valid-size FLOAT     Validation split, default 0.1
  --sample N             Optional train sample size for smoke tests
  --predict-limit N      Optional test row limit for smoke tests
  --no-gradient-checkpointing
                         Disable checkpointing to trade memory for speed
  --lora-r N             LoRA rank, default 8
  --lora-alpha N         LoRA alpha, default 16
  --wandb                Enable Weights & Biases logging
  --wandb_project NAME   W&B project, default ccks2026
  --wandb_run_name NAME  W&B run name
  --model-dir PATH       Base model directory
  --output-dir PATH      Adapter output directory
  --submission-dir PATH  Submission output directory
  --log-path PATH        Log file path
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$NPROC_PER_NODE" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
fi

if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NPROC_PER_NODE" -lt 2 ]]; then
  echo "DDP requires --nproc-per-node >= 2. Use scripts/run_deberta_lora.sh for single-GPU training." >&2
  exit 2
fi

mkdir -p logs "$OUTPUT_DIR" "$SUBMISSION_DIR"
if [[ -z "$LOG_PATH" ]]; then
  LOG_PATH="logs/deberta_lora_ddp_$(date +%Y%m%d_%H%M%S).log"
fi

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "$$" > logs/deberta_lora.pid
echo "$LOG_PATH" > logs/deberta_lora_latest.logpath
trap 'status=$?; rm -f logs/deberta_lora.pid; exit $status' EXIT

exec > >(tee -a "$LOG_PATH") 2>&1

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

torchrun_cmd=()
torchrun_label=""
if [[ -n "$TORCHRUN_BIN" ]]; then
  if ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
    echo "[setup] TORCHRUN_BIN not found: $TORCHRUN_BIN" >&2
    exit 127
  fi
  torchrun_cmd=("$TORCHRUN_BIN")
  torchrun_label="$TORCHRUN_BIN"
elif "$PYTHON_BIN" -c "import torch.distributed.run" >/dev/null 2>&1; then
  torchrun_cmd=("$PYTHON_BIN" -m torch.distributed.run)
  torchrun_label="$PYTHON_BIN -m torch.distributed.run"
else
  echo "[setup] torch.distributed.run is unavailable in the selected Python environment." >&2
  echo "[setup] install PyTorch in the active conda environment, then rerun this script." >&2
  exit 127
fi

"$PYTHON_BIN" - <<'PY'
missing = []
for module_name in ("torch", "peft", "transformers", "sklearn", "pandas", "numpy", "sentencepiece", "tiktoken", "tqdm"):
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        missing.append(module_name)
if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + "\nInstall project dependencies with: python -m pip install -r requirements.txt "
        + "-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    )
PY
if [[ "$WANDB" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
try:
    import wandb  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit("Missing Python package: wandb\nInstall with: python -m pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn") from exc
PY
fi

train_options=(
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --max-length "$MAX_LENGTH"
  --max-chars "$MAX_CHARS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --head-lr "$HEAD_LR"
  --weight-decay "$WEIGHT_DECAY"
  --head-hidden-size "$HEAD_HIDDEN_SIZE"
  --head-dropout "$HEAD_DROPOUT"
  --multi-sample-dropout "$MULTI_SAMPLE_DROPOUT"
  --grad-accum "$GRAD_ACCUM"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --valid-size "$VALID_SIZE"
)
if [[ "$WANDB" == "1" ]]; then
  train_options+=(--wandb --wandb_project "$WANDB_PROJECT")
  if [[ -n "$WANDB_RUN_NAME" ]]; then
    train_options+=(--wandb_run_name "$WANDB_RUN_NAME")
  fi
fi
if [[ "$GRADIENT_CHECKPOINTING" == "0" ]]; then
  train_options+=(--no-gradient-checkpointing)
fi
if [[ "$SAMPLE" != "0" ]]; then
  train_options+=(--sample "$SAMPLE")
fi

predict_options=(
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --submission-dir "$SUBMISSION_DIR"
  --max-length "$MAX_LENGTH"
  --max-chars "$MAX_CHARS"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
)
if [[ "$PREDICT_LIMIT" != "0" ]]; then
  predict_options+=(--predict-limit "$PREDICT_LIMIT")
fi

log "repo=$REPO_ROOT"
log "python=$($PYTHON_BIN -c 'import sys; print(sys.executable)')"
log "launcher=$torchrun_label nproc_per_node=$NPROC_PER_NODE"
log "max_length=$MAX_LENGTH max_chars=$MAX_CHARS epochs=$EPOCHS batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM eval_batch_size=$EVAL_BATCH_SIZE lr=$LR head_lr=$HEAD_LR head_dropout=$HEAD_DROPOUT multi_sample_dropout=$MULTI_SAMPLE_DROPOUT lora_r=$LORA_R lora_alpha=$LORA_ALPHA num_workers=$NUM_WORKERS gradient_checkpointing=$GRADIENT_CHECKPOINTING wandb=$WANDB wandb_project=$WANDB_PROJECT wandb_run_name=$WANDB_RUN_NAME"

log "START distributed train-label"
"${torchrun_cmd[@]}" --standalone --nproc_per_node "$NPROC_PER_NODE" \
  train/train_deberta_lora_ddp.py train-label "${train_options[@]}"
log "END distributed train-label"

log "START distributed train-family"
"${torchrun_cmd[@]}" --standalone --nproc_per_node "$NPROC_PER_NODE" \
  train/train_deberta_lora_ddp.py train-family "${train_options[@]}"
log "END distributed train-family"

log "START predict"
"$PYTHON_BIN" train/train_deberta_lora_single.py predict "${predict_options[@]}"
log "END predict"

log "DONE submission=$SUBMISSION_DIR/submit_deberta_lora.jsonl"
