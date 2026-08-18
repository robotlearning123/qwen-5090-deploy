# qwen-5090-deploy

**The Qwen-on-RTX-5090 series — best use of every local Qwen on one 32 GB card.**

> Automating this repo? `AGENTS.md` lists the command contracts and the
> verification gate (`scripts/verify.sh`, exit 0 = safe to commit).
>
> **New here?** The 4-command quickstart is below; prerequisites and a
> troubleshooting table follow. A slow first start is the one-time NVFP4
> JIT compile.

Every generation gets the same treatment: deployment profiles that actually
fit 32 GB, benchmarks with executable oracles (no eyeball grading), and
written receipts for every dead end we hit.

| model | what you get | headline (this repo, single RTX 5090 32 GB) |
|---|---|---|
| **Qwen3.8-27B (dense, NVFP4)** | 4 serving lanes + full bench + Claude-Code integration | 64 → **134 tok/s** thinking decode (MTP, lossless); HumanEval **83.5%**; cc agent task 206 s → **29 s**; DSpark fits **only** via llama.cpp |
| **Qwen3.6-35B-A3B (MoE, NVFP4/AWQ)** | the full 2026-05 measurement campaign: vLLM/SGLang/llama.cpp/Ollama + A100 remote reference | **1,785 tok/s** aggregate @64 req (the MoE record on this card) |
| **Qwen3.6-27B (dense)** | the failure record | could NOT run on 32 GB in 2026-05 (OOM / multimodal init) — the wall that Qwen3.8-27B-NVFP4 broke through |

## Prerequisites
- Reference box: RTX 5090 32 GB, driver ≥ 580 (Blackwell); the install path enforces 5090-class (`SKIP_GPU_PRECHECK=1` to override — other GPUs: run the benches against any OpenAI-compatible server, see CONTRIBUTING.md)
- Python 3.11+, ~25 GB disk for the NVFP4 checkpoint, internet for one model download
- First `vllm serve` JIT-compiles NVFP4 kernels: expect 8–25 min ONCE (cached afterwards; `scripts/patch_flashinfer.py` fixes the known cu13 linker issue automatically at serve time)

## Quickstart (Qwen3.8-27B)

```bash
scripts/install.sh                                  # dedicated .venv + NVFP4 snapshot
QWEN38_MAX_MODEL_LEN=98304 scripts/serve.sh         # long-context lane (agents/cc)
scripts/chat.py --out /tmp/chat.json --prompt 'hi'  # smoke
# speed lane: QWEN38_SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}' scripts/serve.sh
# Claude Code on it: scripts/cc.sh -p 'fix the failing test in tests/'
```

See `profiles.md` for all four lanes (bench / MTP-speed / cc-96k /
llama.cpp-DSpark) and when to use each.

## Why this exists

Most "runs great on my GPU" repos show one happy path. Real single-GPU work
is choosing between trade-offs and knowing which walls are physical. This
series records both sides:

- `docs/results.md` — the numbers, same harness across engines
- `docs/dead-ends.md` — DSpark's 8-config OOM matrix on 32 GB, MTP@96k,
  prefix-cache-on-hybrid, reasoning-effort injection, and why each fails
- `docs/cc-integration.md` — running Claude Code against a local Qwen:
  context-budget math, the 206 s → 29 s optimization ladder
- `benchmarks/` — machine-readable JSON + the curated prior-generation kits

## Repo layout

```
scripts/     serve/chat/bench/oracle tooling (paths are $ROOT-relative)
tests/       pytest suite (CPU-safe)
benchmarks/  qwen3.8-27b-nvfp4/ · qwen3.6-35b-a3b-moe/ · qwen3.6-27b/
docs/        results · dead-ends · cc-integration
```

## Troubleshooting
| symptom | cause → fix |
|---|---|
| first serve dies in `ninja`/`ld: cannot find -lcudart` | FlashInfer JIT link paths — fixed automatically by `scripts/patch_flashinfer.py`; it runs at serve start, rerun `scripts/serve.sh` |
| `Engine core initialization failed` right after weights load | another engine still holds VRAM (spec-decode engines can hang): `pkill -9 -f '[v]llm'`, wait for `nvidia-smi` to drain, retry |
| cc/agent clients 500 with "maximum context length" | you're on the 16k lane; agent harnesses need `QWEN38_MAX_MODEL_LEN=98304 scripts/serve.sh` |
| 96k lane slow per-token | that's eager mode (required ≥32k on 32 GB); use the MTP speed lane for single-stream work |
| quality result differs run-to-run | borderline tasks rotate with sampling (see docs/results.md fail-pair note); rerun before concluding |

## Tooling reference
| file / command | purpose |
|---|---|
| `AGENTS.md` | repo map, command contracts, verification rules |
| `llms.txt` | short overview with entry points |
| `repo.json` / `scripts/profiles.json` | machine-readable manifests |
| `scripts/healthcheck.sh [port]` | deployment sanity check |
| `scripts/verify.sh` | full check suite — run after any change (CI runs the same) |

## Contributions welcome

More Qwen generations, more GPUs (4090/3090/48 GB — same JSON schema, PR the
results), more dead ends. See `CONTRIBUTING.md`.

## Credits & licenses

- Our code: Apache-2.0. `scripts/chat-template-nothink.jinja` derives from
  `unsloth/Qwen3.8-27B-NVFP4` (Apache-2.0).
- All benchmark artifacts are the authors' own measurements (see each
  `benchmarks/*/PROVENANCE.md`); no third-party content is redistributed.
- Model weights belong to their publishers; nothing here redistributes
  weights.
