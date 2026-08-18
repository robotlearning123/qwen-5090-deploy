#!/usr/bin/env bash
# Single serve entry for Qwen3.8-27B NVFP4 on the local 5090.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/setup.env"

VENV="${QWEN38_VENV:-$ROOT/.venv}"
MODEL="${QWEN38_MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
HOST="${QWEN38_HOST:-127.0.0.1}"
PORT="${QWEN38_PORT:-8000}"
MAX_LEN="${QWEN38_MAX_MODEL_LEN:-16384}"
GPU_UTIL="${QWEN38_GPU_UTIL:-0.90}"
KV_DTYPE="${QWEN38_KV_CACHE_DTYPE:-fp8}"
PRINT_ONLY=0
# Thinking mode: 1 = model default (thinking on); 0 = force off via repo-local
# template variant (better latency for agentic/cc workloads).
ENABLE_THINKING="${QWEN38_ENABLE_THINKING:-1}"
# Official vLLM Qwen3.8-27B recipe for a single 32GB RTX 5090 at 32k context:
# --enforce-eager is REQUIRED (CUDA-graph capture OOMs otherwise).
ENFORCE_EAGER="${QWEN38_ENFORCE_EAGER:-auto}"

usage() {
  cat <<EOF
Usage: $0 [--print] [--port N] [--max-model-len N]
  Serves $MODEL via the dedicated venv vLLM.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print) PRINT_ONLY=1; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --max-model-len) MAX_LEN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

VLLM_BIN="$VENV/bin/vllm"
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "CRIT: $VLLM_BIN missing. Run scripts/install.sh first." >&2
  exit 1
fi
"$VENV/bin/python" "$ROOT/scripts/patch_flashinfer.py" || true

# Modest context: 32 GB 5090 cannot hold native 262144 + weights.
CMD=(
  "$VLLM_BIN" serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size 1
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --kv-cache-dtype "$KV_DTYPE"
  --max-num-seqs "${QWEN38_MAX_NUM_SEQS:-4}"
  --max-num-batched-tokens "${QWEN38_MAX_BATCHED_TOKENS:-8192}"
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --trust-remote-code
  --served-model-name qwen3.8-27b
  ${QWEN38_LINEAR_BACKEND:+--linear-backend "$QWEN38_LINEAR_BACKEND"}
  ${QWEN38_SPEC_CONFIG:+--speculative-config "$QWEN38_SPEC_CONFIG"}
  ${QWEN38_LANGUAGE_MODEL_ONLY:+--language-model-only}
)

# auto: eager is required when context exceeds what CUDA-graph capture fits
# (official recipe: 32k on a 32GB 5090 needs --enforce-eager; 16k fits graphs).
if [[ "$ENFORCE_EAGER" == "1" || ( "$ENFORCE_EAGER" == "auto" && "$MAX_LEN" -gt 16384 ) ]]; then
  CMD+=(--enforce-eager)
fi
if [[ "$ENABLE_THINKING" == "0" ]]; then
  CMD+=(--chat-template "$ROOT/scripts/chat-template-nothink.jinja")
fi

if [[ "$PRINT_ONLY" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
# Host nvcc is 12.8; FlashInfer SM120 NVFP4 JIT needs CUDA >= 12.9.
# The venv ships CUDA 13.3 nvcc + libs.
CU13="$VENV/lib/python3.13/site-packages/nvidia/cu13"
if [[ -x "$CU13/bin/nvcc" ]]; then
  export CUDA_HOME="$CU13"
  export PATH="$CU13/bin:$PATH"
  export LD_LIBRARY_PATH="${CU13}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LIBRARY_PATH="${CU13}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
exec "${CMD[@]}"
