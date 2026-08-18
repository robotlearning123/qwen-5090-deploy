# Qwen3.6-35B-A3B campaign summary (2026-05-12..14, RTX 5090 32 GB)

Curated overview; every number links to a machine receipt in this directory.
Raw working notes are retained privately — only data receipts ship here.

## Serving throughput (vLLM 0.19.1, RedHatAI NVFP4, 8k ctx, 64 req × 512 tok)
- **1,785.4 tok/s aggregate** — `vllm_nvfp4_8k_p64_mbt8192_nolog_multilen_result_2026-05-13.json`
- official-bench re-run 1,663.3 out tok/s, mean TTFT 6.28 s, mean TPOT 12.58 ms
  — `official_random_in20_out512_p64_temp0_2026-05-13.json`
- long-context (10k in / 1.5k out): single 168.9 tok/s, ×10 concurrent 768.2
  — `official_aa_shape_random_in10000_out1500_c1_vllm0202_gmem088_2026-05-13.json`
- engine ladder: vLLM 1,785 > SGLang 832 (8k ctx) > Ollama ~171
  — `sglang_qwen3.6_runtime_probe_2026-05-13.json`, `qwen3.6_35b_5090_leaderboard_2026-05-13.json`
- backend one-by-one smoke (llama.cpp 502.7 / vLLM 460.4 / SGLang 69.5 tok/s,
  8×128 shape) — `backend-smoke/*.json`
- negative result: MTP speculative decoding NOT competitive on this MoE
  (92.0 tok/s single-stream) — superseded in the 3.8-27B campaign where MTP
  gave +57/+108%: see ../qwen3.8-27b-nvfp4/REPORT.md

## Remote reference (A100-40GB, AWQ-Marlin, vLLM 0.20.2) — remote-a100/
- 1,406 tok/s @64 concurrent, TTFT p50 338 ms, TPOT p50 22.1 ms
  — `remote-a100/awq_marlin_a100.json`
- SWE-bench Lite n=300: 260/300 = 86.7% produced valid patches
  — `remote-a100/swe-lite-summary.json` (+ raw .jsonl.gz)

Hardware context and cross-generation comparison: repo-root docs/results.md.
