#!/usr/bin/env bash
# Sweep serve configs for Qwen3.8-27B on the 5090 and bench each one.
# Usage: scripts/bench_matrix.sh <tag1=ENV1> <tag2=ENV2> ...
#   e.g. scripts/bench_matrix.sh \
#     dspark16='QWEN38_SPEC_CONFIG={"method":"dspark","model":"Doopeworld/Qwen3.8-27B-DSpark-vLLM"}' \
#     seqs16='QWEN38_MAX_NUM_SEQS=16'
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts/bench-2026-08-17"
PORT="${QWEN38_PORT:-8000}"

wait_ready() {
  for _ in $(seq 1 60); do
    curl -sf -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    grep -qE 'ValueError|EngineDead|OutOfMemoryError|CRIT' "$1" 2>/dev/null && return 1
    sleep 5
  done
  return 1
}

kill_engine() {
  pkill -f '[v]llm serve' 2>/dev/null; pkill -f 'VLLM::' 2>/dev/null
  for _ in $(seq 1 24); do
    pgrep -f '[v]llm' >/dev/null 2>&1 || break
    sleep 5
  done
  pkill -9 -f '[v]llm' 2>/dev/null; pkill -9 -f 'VLLM::' 2>/dev/null
  # wait until the GPU is actually released (spec-decode engines can hang
  # holding all of VRAM after an init OOM and poison the next config)
  for _ in $(seq 1 24); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 5
  done
  echo "WARN: GPU not freed (used=${used}MiB)"; return 1
}

run_one() {
  local tag="$1" envs="$2"
  local log="$OUT/serve-${tag}.log"
  echo "=== [$tag] env: $envs"
  kill_engine
  nohup env $envs "$ROOT/scripts/serve.sh" >"$log" 2>&1 &
  if ! wait_ready "$log"; then
    echo "[$tag] FAILED to start"; tr '\r' '\n' <"$log" | grep -E 'ValueError|OutOfMemory|CRIT' | tail -2
    echo "$tag FAILED" >> "$OUT/matrix-results.txt"; return
  fi
  local kv; kv=$(tr '\r' '\n' <"$log" | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1)
  "$ROOT/.venv/bin/python" "$ROOT/scripts/bench_speed.py" \
    --out "$OUT/speed-${tag}.json" --concurrency 4 >/dev/null 2>&1
  local summ; summ=$("$ROOT/.venv/bin/python" - "$OUT/speed-${tag}.json" <<'PY'
import json, sys
rs = json.load(open(sys.argv[1]))
d = {r['label']: r for r in rs}
dn, dt, c4 = d.get('decode_nonthink'), d.get('decode_think'), d.get('concurrent_4')
print(f"{dn['decode_tok_per_s'] if dn else -1:.1f}/{dt['decode_tok_per_s'] if dt else -1:.1f}/{c4['aggregate_tok_per_s'] if c4 else -1:.1f}")
PY
)
  echo "[$tag] KV: $kv | decode/agg: $summ"
  echo "$tag kv=$kv tok/s(nonthink/think/agg4)=$summ" >> "$OUT/matrix-results.txt"
}

mkdir -p "$OUT"; : > "$OUT/matrix-results.txt"
for spec in "$@"; do
  tag="${spec%%=*}"; envs="${spec#*=}"
  run_one "$tag" "$envs"
done
pkill -f '[v]llm serve' 2>/dev/null
echo "=== MATRIX DONE"; cat "$OUT/matrix-results.txt"
