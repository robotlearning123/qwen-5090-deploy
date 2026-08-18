# Contributing

Three ways this series grows:

1. **New hardware result** — run `scripts/bench_speed.py` +
   `scripts/bench_quality.py` against your deployment, PR the JSONs under
   `benchmarks/<model>/<your-gpu>/` (schema: see existing files). Include the
   exact serve command; results without a command will be asked for it.
2. **New Qwen generation** — add a `benchmarks/<model>/` lane using the same
   harness. Keep the executable-oracle rule: no eyeball grading.
3. **New dead end** — the most welcome PRs. Reproduce, then document config +
   failure evidence under `docs/dead-ends.md`.

Rules: every number links to a raw JSON/log; speculative-decoding claims need
a lossless check (same suite before/after); no weight redistribution.
