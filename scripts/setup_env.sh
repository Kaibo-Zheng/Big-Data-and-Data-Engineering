#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-conda}}"
CONDA_ENV="${CONDA_ENV:-ccks2026}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_CHANNEL="${CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
CONDA_EXTRA_CHANNELS="${CONDA_EXTRA_CHANNELS:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirrors.aliyun.com/pytorch-wheels/${TORCH_CUDA}}"
TORCH_TRUSTED_HOST="${TORCH_TRUSTED_HOST:-mirrors.aliyun.com}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_TORCHVISION="${INSTALL_TORCHVISION:-0}"
INSTALL_TORCHAUDIO="${INSTALL_TORCHAUDIO:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
PIP_TIMEOUT="${PIP_TIMEOUT:-120}"
PIP_RETRIES="${PIP_RETRIES:-10}"
PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-20}"

echo "[setup] repo=$REPO_ROOT"
echo "[setup] conda_env=$CONDA_ENV"
echo "[setup] python_version=$PYTHON_VERSION"
echo "[setup] conda_channel=$CONDA_CHANNEL"
if [[ -n "$CONDA_EXTRA_CHANNELS" ]]; then
  echo "[setup] conda_extra_channels=$CONDA_EXTRA_CHANNELS"
fi
echo "[setup] pypi=$PIP_INDEX_URL"
echo "[setup] torch_index=$TORCH_INDEX_URL"
echo "[setup] torch=torch==$TORCH_VERSION"

if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
  echo "[setup] conda not found: $CONDA_BIN" >&2
  echo "[setup] install Miniconda/Anaconda first, or set CONDA_BIN=/path/to/conda" >&2
  exit 127
fi

conda_channels=(--override-channels -c "$CONDA_CHANNEL")
for channel in $CONDA_EXTRA_CHANNELS; do
  conda_channels+=(-c "$channel")
done

if "$CONDA_BIN" run -n "$CONDA_ENV" python --version >/dev/null 2>&1; then
  echo "[setup] conda env exists: $CONDA_ENV"
else
  "$CONDA_BIN" create -y -n "$CONDA_ENV" "${conda_channels[@]}" "python=$PYTHON_VERSION" pip
fi

conda_python=("$CONDA_BIN" run -n "$CONDA_ENV" python)
pip_common=(
  --timeout "$PIP_TIMEOUT"
  --retries "$PIP_RETRIES"
)
if "${conda_python[@]}" -m pip install --help | grep -q -- "--resume-retries"; then
  pip_common+=(--resume-retries "$PIP_RESUME_RETRIES")
fi

"${conda_python[@]}" -m pip install --upgrade pip setuptools wheel \
  -i "$PIP_INDEX_URL" \
  --trusted-host "$PIP_TRUSTED_HOST" \
  "${pip_common[@]}"

if [[ "$INSTALL_TORCH" == "1" ]]; then
  torch_packages=("torch==${TORCH_VERSION}+${TORCH_CUDA}")
  if [[ "$INSTALL_TORCHVISION" == "1" ]]; then
    torch_packages+=("torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}")
  fi
  if [[ "$INSTALL_TORCHAUDIO" == "1" ]]; then
    torch_packages+=("torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA}")
  fi
  "${conda_python[@]}" -m pip install "${torch_packages[@]}" \
    --index-url "$TORCH_INDEX_URL" \
    --extra-index-url "$PIP_INDEX_URL" \
    --trusted-host "$TORCH_TRUSTED_HOST" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    "${pip_common[@]}"
fi

"${conda_python[@]}" -m pip install -r requirements.txt \
  -i "$PIP_INDEX_URL" \
  --trusted-host "$PIP_TRUSTED_HOST" \
  "${pip_common[@]}"

"${conda_python[@]}" - <<PY
import importlib.util
import sys

print("[verify] python ok")
if importlib.util.find_spec("torch") is None:
    print("[verify] torch not installed")
    if "$INSTALL_TORCH" == "1":
        raise SystemExit(1)
else:
    import torch

    print(f"[verify] torch={torch.__version__}")
    print(f"[verify] torch_cuda={torch.version.cuda}")
    print(f"[verify] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[verify] gpu_count={torch.cuda.device_count()}")
        print(f"[verify] gpu={torch.cuda.get_device_name(0)}")
    elif "$REQUIRE_CUDA" == "1":
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false")

    try:
        import torch.distributed.run  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"torch.distributed.run unavailable: {exc}") from exc
    print("[verify] torch.distributed.run ok")
PY

echo "[setup] done"
echo "[setup] activate with: conda activate ${CONDA_ENV}"
