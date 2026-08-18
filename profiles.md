# Serving lanes — Qwen3.8-27B on 32 GB RTX 5090

All measured 2026-08-17/18 (raw JSON in benchmarks/qwen3.8-27b-nvfp4/).

| lane | command | think decode | quality | use for |
|---|---|---|---|---|
| bench | `scripts/serve.sh` (16k, CUDA graphs) | 64.4 tok/s | 14/16 | max batch: ~225 tok/s aggregate is the card's compute wall |
| **speed** | `QWEN38_SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}' scripts/serve.sh` | **133.8 tok/s** | 14/16 identical (lossless) | interactive / API single-stream |
| **agents** | `QWEN38_MAX_MODEL_LEN=98304 scripts/serve.sh` (+ `scripts/cc.sh`) | 64 tok/s | cc E2E verified | Claude-Code-style agents; 96k floor = agent system+tools ≈ 29k+ tokens |
| llama-DSpark | `scripts/serve_llamacpp.sh` (GGUF Q4_K_XL + DSpark drafter, n_max=3) | 90.2 tok/s | 14/16 | the ONLY way DSpark fits 32 GB; upgrade path as acceptance improves |

Notes: eager is auto-enabled >16k context (CUDA-graph capture OOMs, per the
official vLLM recipe); `QWEN38_ENABLE_THINKING=0` swaps in a no-think chat
template (~27% faster routine agent turns (44 s → 32 s, receipt cc-e2e-fib.json), lossless).
