#!/usr/bin/env bash
# Fail-fast GPU/disk/engine precheck. Exit 1 if this host cannot serve Qwen3.8-27B.
set -euo pipefail
# SKIP_GPU_PRECHECK=1 bypasses the GPU-class gate (this kit targets RTX 5090 32GB;
# other GPUs: see CONTRIBUTING.md — run the benches against any OpenAI server).

WEIGHT_TARGET="${WEIGHT_TARGET:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-20000}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-40}"
STRICT_FREE_VRAM="${STRICT_FREE_VRAM:-1}"

echo "=== precheck $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

if ! command -v nvidia-smi >/dev/null; then
  echo "CRIT: nvidia-smi not found"
  exit 1
fi

nvidia-smi
echo
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver_version,compute_cap --format=csv

mapfile -t GPU_ROW < <(nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader,nounits)
IFS=',' read -r GPU_NAME GPU_TOTAL GPU_FREE DRIVER CC <<<"${GPU_ROW[0]}"
GPU_NAME="$(echo "$GPU_NAME" | xargs)"
GPU_TOTAL="$(echo "$GPU_TOTAL" | xargs)"
GPU_FREE="$(echo "$GPU_FREE" | xargs)"
DRIVER="$(echo "$DRIVER" | xargs)"
CC="$(echo "$CC" | xargs)"

echo
echo "parsed name='$GPU_NAME' total_mib=$GPU_TOTAL free_mib=$GPU_FREE driver=$DRIVER cc=$CC"

if [[ "${SKIP_GPU_PRECHECK:-0}" == "1" ]]; then
  echo "SKIP_GPU_PRECHECK=1: skipping 5090-class/VRAM gates"
else
  case "$GPU_NAME" in
    *5090*|*Blackwell*) ;;
    *)
      echo "CRIT: expected RTX 5090-class Blackwell GPU, got '$GPU_NAME' (SKIP_GPU_PRECHECK=1 to bypass)"
      exit 1
      ;;
  esac
  if (( GPU_TOTAL < 30000 )); then
    echo "CRIT: GPU total VRAM ${GPU_TOTAL} MiB < 30000 (need ~32 GB class) (SKIP_GPU_PRECHECK=1 to bypass)"
    exit 1
  fi
fi

echo
echo "=== CUDA ==="
if command -v nvcc >/dev/null; then
  nvcc --version
else
  echo "WARN: nvcc not on PATH"
fi

echo
echo "=== disk on weight target $WEIGHT_TARGET ==="
df -h "$WEIGHT_TARGET" / 2>/dev/null || df -h /
mkdir -p "$WEIGHT_TARGET" 2>/dev/null || true
AVAIL_GB=$(df -BG --output=avail "$WEIGHT_TARGET" 2>/dev/null | tail -1 | tr -dc '0-9')
AVAIL_GB="${AVAIL_GB:-$(df -BG --output=avail / | tail -1 | tr -dc '0-9')}"
echo "avail_gb=$AVAIL_GB"
if (( AVAIL_GB < MIN_FREE_DISK_GB )); then
  echo "CRIT: only ${AVAIL_GB} GB free on $WEIGHT_TARGET (need ${MIN_FREE_DISK_GB})"
  exit 1
fi

echo
echo "=== engines (advisory; do not execute serve binaries) ==="
if command -v vllm >/dev/null; then
  echo "vllm: $(command -v vllm)"
  python3 -c "import vllm; print('vllm_import', vllm.__version__)" 2>/dev/null || echo "vllm_import: failed"
else
  echo "vllm: not on PATH"
fi
command -v sglang >/dev/null && echo "sglang: $(command -v sglang)" || echo "sglang: not on PATH"
command -v ollama >/dev/null && ollama --version || echo "ollama: not on PATH"
command -v llama-server >/dev/null && echo "llama-server: $(command -v llama-server)" || echo "llama-server: not on PATH"

echo
echo "=== justification ==="
echo "BF16 rejected: official Qwen3.8-27B BF16 is ~56 GB (usedStorage 55575816096) and cannot fit ${GPU_TOTAL} MiB."
echo "Official FP8 (~28 GB weights) plus KV is tight on a 32 GB card; not the default."
echo "Default: NVFP4 + current vLLM + FP8 KV. Measured free VRAM now: ${GPU_FREE} MiB of ${GPU_TOTAL} MiB."

if (( GPU_FREE < MIN_FREE_VRAM_MIB )); then
  echo "WARN: free VRAM ${GPU_FREE} MiB < ${MIN_FREE_VRAM_MIB} (occupant must be cleared before serve)."
  if [[ "$STRICT_FREE_VRAM" == "1" ]]; then
    echo "CRIT: refusing to proceed while the 5090 is occupied (STRICT_FREE_VRAM=1)."
    echo "Set STRICT_FREE_VRAM=0 to allow download/install-only after this report."
    exit 2
  fi
fi

echo "PRECHECK_OK name=$GPU_NAME total_mib=$GPU_TOTAL free_mib=$GPU_FREE driver=$DRIVER"
exit 0
