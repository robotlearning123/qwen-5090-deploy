# Remote A100 reference lane

Qwen3.6-35B-A3B AWQ-Marlin (checkpoint ~22 GiB per publisher) on an ACCESS-allocated A100-40GB VM,
vLLM 0.20.2, 2026-05-14. Serving numbers: `awq_marlin_a100.json`
(1,406 tok/s @64c, TTFT p50 338 ms, TPOT p50 22.1 ms). Quality:
SWE-bench Lite n=300 → 86.7% valid patches (`swe-lite-summary.json`,
raw rows in the .jsonl.gz). The FP8 variant OOM'd on 40 GB.
Full local-vs-remote table: ../../CAMPAIGN.md and repo docs/results.md.
