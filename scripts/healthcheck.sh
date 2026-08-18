#!/usr/bin/env bash
# One-command sanity check for a running deployment. Exit 0 = all green.
# Usage: scripts/healthcheck.sh [port]     (default 8000)
set -uo pipefail
PORT="${1:-8000}"
BASE="http://127.0.0.1:${PORT}"
fail=0

say() { printf '%s %s\n' "$1" "$2"; }
[[ "${1:-}" == "-h" ]] && { echo "usage: $0 [port]"; exit 0; }

# 1. server up?
if body=$(curl -sf -m 3 "$BASE/v1/models" 2>/dev/null); then
  say OK "server reachable on :$PORT"
else
  say FAIL "server not reachable on :$PORT (start: scripts/serve.sh &)"; exit 1
fi

# 2. model id
mid=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -n "$mid" ] && say OK "model served: $mid" || { say FAIL "no model id in /v1/models"; exit 1; }

# 3. one completion
out=$(python3 "$(dirname "$0")/chat.py" --base-url "$BASE/v1" --model "$mid" \
      --out /tmp/healthcheck-chat.json --prompt 'Reply with exactly: OK' --no-enable-thinking 2>/dev/null)
if grep -q '"content"' /tmp/healthcheck-chat.json 2>/dev/null; then
  say OK "chat completion works ($(python3 -c 'import json;d=json.load(open("/tmp/healthcheck-chat.json"));print((d.get("response", d).get("choices") or [{}])[0].get("message",{}).get("content","")[:40])' 2>/dev/null))"
else
  say FAIL "chat completion failed"; fail=1
fi

# 4. GPU headroom
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | while read -r l; do say INFO "gpu: $l"; done
fi

[ "$fail" = 0 ] && say PASS "healthcheck complete" || say FAIL "issues above"
exit $fail
