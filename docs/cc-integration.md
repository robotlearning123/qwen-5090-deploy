# Claude Code on a local Qwen (96k lane)

vLLM ≥0.23 natively serves the Anthropic /v1/messages protocol — no proxy
needed. `scripts/cc.sh` wraps the required environment.

The four things that matter:
1. **Context floor ~96k.** Claude-Code's system prompt + tool schemas are
   ~28.6k tokens on the first request and grow (24 → 160 tools observed with
   MCP fleets). 16k/32k/48k/64k all died mid-loop; 96k works.
2. **`CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096`** — the default 32k request
   exceeds the server context.
3. **Minimal config dir** (repo `cc-profile/`): 160 → 24 tools, first request
   28.6k → ~20k tokens. Agent task 92 s → 44 s.
4. **`--dangerously-skip-permissions` for headless runs** — the sandbox
   classifier stalls -p runs (six denied retries measured); with it, plus
   no-think template: 206 s → 29 s on a create-and-run task (7.1×), output
   independently verified.

For thinking control there is no /v1/messages path (see dead-ends #5); use
the server-level `QWEN38_ENABLE_THINKING=0` template swap.
