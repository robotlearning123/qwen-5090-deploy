#!/usr/bin/env bash
# Run Claude Code (cc) against the local Qwen3.8-27B vLLM server.
# vLLM >= 0.17 natively serves the Anthropic /v1/messages endpoint, so no
# proxy (ccr/litellm) is needed. Requires scripts/serve.sh to be running.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${QWEN38_PORT:-8000}"
BRIDGE_PORT="${QWEN38_CC_BRIDGE_PORT:-8001}"

# Optional buffered bridge (QWEN38_CC_BRIDGE=1) for debugging what cc sends:
# dumps every /v1/messages body to /tmp/cc-bridge-dumps. Direct works fine.
if [[ "${QWEN38_CC_BRIDGE:-0}" == "1" ]]; then
  if ! curl -sf -m 2 "http://127.0.0.1:${BRIDGE_PORT}/healthz" >/dev/null 2>&1; then
    nohup "$ROOT/.venv/bin/python" "$ROOT/scripts/cc_bridge.py" \
      "$BRIDGE_PORT" "http://127.0.0.1:${PORT}" "${QWEN38_CC_BRIDGE_DUMPS:-/tmp/cc-bridge-dumps}" \
      >/dev/null 2>&1 &
    for _ in $(seq 1 20); do
      curl -sf -m 1 "http://127.0.0.1:${BRIDGE_PORT}/healthz" >/dev/null 2>&1 && break
      sleep 0.5
    done
  fi
  export ANTHROPIC_BASE_URL="http://127.0.0.1:${BRIDGE_PORT}"
else
  export ANTHROPIC_BASE_URL="http://127.0.0.1:${PORT}"
fi
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-dummy}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-dummy}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${QWEN38_CC_MODEL:-qwen3.8-27b}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${QWEN38_CC_MODEL:-qwen3.8-27b}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${QWEN38_CC_MODEL:-qwen3.8-27b}"
# Host env pins these to the remote GLM relay; they override the tier vars
# above, so they must be re-pointed too.
export ANTHROPIC_MODEL="${QWEN38_CC_MODEL:-qwen3.8-27b}"
export ANTHROPIC_SMALL_FAST_MODEL="${QWEN38_CC_MODEL:-qwen3.8-27b}"
# cc asks for 32k output tokens by default; our 16k context serves a 5090, so
# cap the request or vLLM rejects it with 500 max_completion_tokens>max_model_len.
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${QWEN38_CC_MAX_OUTPUT_TOKENS:-4096}"
# cc assumes a 200k window and won't auto-compact before the local server's
# smaller limit kills the request mid-loop. Point compaction at the real
# window (leave headroom for one in-flight reply).
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="${QWEN38_CC_AUTO_COMPACT_WINDOW:-58000}"

if ! command -v claude >/dev/null 2>&1; then
  echo "cc.sh needs the Claude Code CLI (npm i -g @anthropic-ai/claude-code)" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  cat <<EOF
Usage: $0 <claude args...>          # a leading '--' is accepted and stripped
  e.g. $0 -p 'What is 2+2? Reply with just the number.'
       $0                           # interactive session on the local model
Best with: QWEN38_MAX_MODEL_LEN=98304 scripts/serve.sh running (cc's system
prompt + tool schemas reach ~30k+ tokens; 16k/32k contexts 500 mid-task).
Env: QWEN38_PORT=${PORT} ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}
     QWEN38_CC_BRIDGE=1 -> route via debug bridge with request dumps
EOF
  exit 0
fi
# Accept an optional leading `--` separator (it is NOT passed through:
# claude would treat the next token as the prompt text).
[[ "${1:-}" == "--" ]] && shift
# Minimal dedicated config dir (repo cc-profile/): strips user skills/MCP/
# memory so cc's first request drops from 28.6k+ to ~20k tokens (160 -> 24
# tools; fib E2E 92 s -> 44 s measured 2026-08-17). Set CLAUDE_CONFIG_DIR
# yourself to override; QWEN38_CC_FULL_CONFIG=1 uses your normal ~/.claude.
if [[ -z "${CLAUDE_CONFIG_DIR:-}" && "${QWEN38_CC_FULL_CONFIG:-0}" != "1" ]]; then
  export CLAUDE_CONFIG_DIR="$ROOT/cc-profile"
fi

# -p runs kept stalling on the sandbox permission classifier (6 denied
# retries measured 2026-08-17); skip it (common headless practice). Local model,
# run in a scratch dir. QWEN38_CC_SKIP_PERMS=0 to restore prompting.
if [[ "${QWEN38_CC_SKIP_PERMS:-1}" == "1" ]]; then
  exec claude --dangerously-skip-permissions "$@"
fi
exec claude "$@"
