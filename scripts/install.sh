#!/usr/bin/env bash
# Install dedicated vLLM (>=0.25) and download the chosen NVFP4 checkpoint.
set -euo pipefail

need_uv() {
  command -v "${UV_BIN:-uv}" >/dev/null 2>&1 && return 0
  echo "install.sh needs uv (fast python env tool)." >&2
  echo "  install: curl -LsSf https://astral.sh/uv/install.sh | sh   (then restart your shell)" >&2
  echo "  or set UV_BIN=/path/to/uv" >&2
  exit 1
}
UV_BIN="${UV_BIN:-uv}"
need_uv

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/setup.env"

VENV="${QWEN38_VENV:-$ROOT/.venv}"
MODEL="${QWEN38_MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"

# Download/install must not run before a precheck exists. Allow occupied GPU
# (STRICT_FREE_VRAM=0) because weights do not need the device.
echo "=== precheck (install-only; GPU occupant allowed) ==="
STRICT_FREE_VRAM=0 "$ROOT/scripts/precheck.sh"

if ! command -v uv >/dev/null; then
  echo "CRIT: uv not found" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
mkdir -p "$UV_CACHE_DIR"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "=== creating venv $VENV ==="
  uv venv "$VENV" --python 3.13
fi

echo "=== installing vLLM stack into $VENV ==="
# Official Unsloth NVFP4 recipe: vllm>=0.25, flashinfer, cutlass-dsl.
# Isolated venv — do not touch the host miniconda vLLM 0.17.0.
uv pip install --python "$VENV/bin/python" \
  "vllm>=0.25.0" \
  "flashinfer-python>=0.6.13" \
  "nvidia-cutlass-dsl>=4.5.2" \
  "transformers>=5.8.0" \
  --torch-backend=auto

"$VENV/bin/python" "$ROOT/scripts/patch_flashinfer.py" || true

echo "=== engine versions ==="
"$VENV/bin/python" - <<'PY'
import transformers, vllm
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
try:
    import flashinfer
    print("flashinfer", getattr(flashinfer, "__version__", "?"))
except Exception as e:
    print("flashinfer", e)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

echo "=== downloading $MODEL into HF_HOME=$HF_HOME ==="
# huggingface_hub from the venv; snapshot is the real checkpoint, not a stub.
"$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="$MODEL")
print("snapshot", path)
PY

echo "INSTALL_OK model=$MODEL venv=$VENV"
