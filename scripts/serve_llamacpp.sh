#!/usr/bin/env bash
# llama.cpp + DSpark speculative lane for Qwen3.8-27B on the 5090.
# OpenAI-compatible endpoint on port 8010 (bench with bench_speed.py).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF_DIR="${QWEN38_GGUF_DIR:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/gguf}"
# Prerequisites (not auto-downloaded):
#   target : unsloth/Qwen3.8-27B-GGUF :: Qwen3.8-27B-UD-Q4_K_XL.gguf (17.9 GB)
#            huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --local-dir "$GGUF_DIR"
#   drafter: magnitudedev/Qwen3.8-27B-DSpark-GGUF :: Qwen3.8-27B-DSpark-Q8_0.gguf (1.4 GB,
#            sha256 b007a76b2ce57c1a3ecf36046543aebf60a763861624647d653f16d336c781d2)
#   engine : llama.cpp with --spec-type support (upstream >= v0.15 has draft-dspark);
#            reference build: git clone https://github.com/ggml-org/llama.cpp &&
#            cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j
#            -> set QWEN38_LLAMA_BIN to llama.cpp/build/bin/llama-server if not at tools/llama.cpp/
BIN="${QWEN38_LLAMA_BIN:-$ROOT/tools/llama.cpp/build/bin/llama-server}"
TARGET="${QWEN38_LLAMA_TARGET:-$GGUF_DIR/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
DRAFT="${QWEN38_LLAMA_DRAFT:-$GGUF_DIR/Qwen3.8-27B-DSpark-Q8_0.gguf}"
for f in "$TARGET" "$DRAFT"; do
  [[ -f "$f" ]] || { echo "missing $f — see download recipe in this script's header" >&2; exit 1; }
done
[[ -x "$BIN" ]] || { echo "llama-server not at $BIN — build recipe in header / set QWEN38_LLAMA_BIN" >&2; exit 1; }
PORT="${QWEN38_LLAMA_PORT:-8010}"
CTX="${QWEN38_LLAMA_CTX:-32768}"
SPEC="${QWEN38_LLAMA_SPEC:-draft-dspark}"   # or 'none' for baseline runs

if [[ "$SPEC" == "none" ]]; then
  exec "$BIN" \
    -m "$TARGET" --port "$PORT" -ngl 99 -c "$CTX" \
    --jinja --host 127.0.0.1
fi
exec "$BIN" \
  -m "$TARGET" -md "$DRAFT" --spec-type "$SPEC" --spec-draft-n-max "${QWEN38_LLAMA_NMAX:-3}" \
  --port "$PORT" -ngl 99 -ngld 99 -c "$CTX" \
  --jinja --host 127.0.0.1
