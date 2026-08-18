# Dead ends on 32 GB — measured, with receipts

Each item was reproduced at least once; raw logs retained in the private lab
monorepo, summaries + configs here.

1. **vLLM `method=dspark` with the NVFP4 target — does not fit.** 8 configs
   (util 0.85-0.93 × graphs/eager × mbt 2048-8192 × spec 4/7 × ±LM-only): all
   die on a fixed ~2.54 GB speculator workspace allocation, or hang silently
   in init after weight load. `--language-model-only` provably frees ~2.6 GB
   (KV pool 46k → 123k tokens with MTP) but does not unblock the speculator.
   The same drafter runs fine in llama.cpp at 19.4 GB total (see profiles).
2. **MTP + 96k context — no.** KV fits only at util ≥0.94 where real ~29k
   prompts OOM activations (167 MB alloc, 210 MB free → engine death).
3. **Raising max_num_seqs does nothing for throughput.** seqs 4/8/16 →
   225.7/221.7/223.0 tok/s: compute-bound, and per-seq KV shrinks 126k→80k.
4. **Prefix caching never hits for agent traffic** on this hybrid
   (Gated-DeltaNet + attention) model — hits=0 across multi-turn runs.
5. **`reasoning_effort` cannot be injected** via /v1/messages
   (chat_template_kwargs ignored; identical output across xhigh/low/medium).
6. **max_num_batched_tokens=16384 at 96k** — 192 MB OOM on first request and
   KV pool drops 126k→98k.
7. **Qwen3.6-27B could not run at all** on this card in 2026-05 (OOM at
   weight-load / multimodal init errors) — see
   benchmarks/qwen3.6-27b/README.md. Qwen3.8-27B-NVFP4 (~21.3 GB)
   is the first 27B-class model that fits.
