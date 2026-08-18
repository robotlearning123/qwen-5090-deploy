# Qwen3.6-27B on 32 GB — the wall (2026-05)

The dense Qwen3.6-27B could not be served on this card in May 2026: every
attempt failed before inference — vLLM died at weight-load/OOM (bf16/FP8
footprints ≥ 54 GB / ≥ 28 GB + activations on 32 GB) and the era's toolchain
hit multimodal-processor init errors. Raw probe log retained privately.

Three months later Qwen3.8-27B-NVFP4 (~21.3 GB weights) became the first
27B-class model to fit this GPU — see ../qwen3.8-27b-nvfp4/. The lesson this
directory exists to record: **quantization format availability, not model
size, was the gating factor for 27B-class on 32 GB.**
