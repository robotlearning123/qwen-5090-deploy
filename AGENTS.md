# AGENTS.md — operating map for AI agents (and fast humans)

One goal per command. Everything below is verified working on the reference
box (RTX 5090 32 GB, driver ≥ 580, Python 3.13); all paths repo-relative.

## The gate
After ANY change run `scripts/verify.sh`. Exit 0 = safe to commit. It checks:
python/shell syntax, every entrypoint's --help, unit tests, manifest JSON
parsing, the internal-reference leak grep (public hygiene), and that every
doc path reference resolves. CI runs the identical script.

## State discovery (first 60 seconds in this repo)
1. `cat repo.json` — what this repo is, where things live (machine manifest)
2. `cat scripts/profiles.json` — the serving lanes and their measured numbers
3. `scripts/serve.sh --print` — exact flags any lane will use (no side effects)
4. `scripts/healthcheck.sh` — is a deployment already running?

## Repo map
- `scripts/` — every executable. No code outside scripts/ and tests/.
- `tests/` — pytest, CPU-safe (mock servers; no GPU needed).
- `benchmarks/<model>/` — machine-readable receipts (JSON/TSV). Read-only.
- `docs/` — results.md (numbers), dead-ends.md (what fails and why),
  cc-integration.md (agent-on-local-model guide).
- `profiles.md` — the 4 serving lanes and when to use each.

## Command contracts
| intent | command | exit 0 means |
|---|---|---|
| prerequisites | `scripts/precheck.sh` | GPU/driver/disk OK (safe to run first, always) |
| install | `scripts/install.sh` | .venv ready + checkpoint downloaded (idempotent) |
| serve (pick lane first) | `scripts/serve.sh` | server on 127.0.0.1:8000 (blocks; run in background) |
| print serve cmd only | `scripts/serve.sh --print` | no side effects — use to inspect lane flags |
| health | `curl -sf 127.0.0.1:8000/v1/models` | server reachable; JSON lists served model id |
| one chat smoke | `scripts/chat.py --out /tmp/c.json --prompt 'say OK'` | 200 + non-empty content |
| speed bench | `scripts/bench_speed.py --base-url http://127.0.0.1:8000/v1 --model <id> --out s.json` | JSON written |
| quality bench | `scripts/bench_quality.py --base-url ... --model <id> --out q.json` | 16-task JSON with pass flags |
| run Claude Code on it | `scripts/cc.sh -p '<task>'` (needs 96k lane) | task output; independently verify file effects |
| stop server | `pkill -f 'vllm serve'` | — |

Gotcha (burned us 5×): never put the literal server binary name in the same
shell command as a pkill of it — bracket the pattern (`pkill -f '[v]llm'`).

## Lane env-var presets
`QWEN38_MAX_MODEL_LEN` context (16384 default; 98304 for agent workloads;
eager auto-enables >16k). `QWEN38_SPEC_CONFIG` JSON for speculative decoding
(MTP recipe inside). `QWEN38_ENABLE_THINKING=0` no-think template.
`QWEN38_LLAMA_*` for the llama.cpp lane. Full table: profiles.md.

## Agent rules for changing this repo
1. Numbers in docs must link to a receipt in benchmarks/ (no unreceipted
   claims — auditors check).
2. Benchmark grading is executable-oracle only (see bench_quality.py);
   never eyeball-grade.
3. Speculative-decoding changes need a before/after quality run showing
   per-task identity (lossless check).
4. New hardware results = new dir `benchmarks/<model>/<gpu>/` with the same
   JSON keys as siblings + the exact serve command in a `cmd.txt`.
