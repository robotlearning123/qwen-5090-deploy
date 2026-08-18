#!/usr/bin/env python3
"""Streaming speed bench for the local Qwen3.8-27B endpoint. Stdlib only.

Measures TTFT, decode tokens/s, prefill speed (long-prompt variant) and
aggregate throughput under concurrency. Writes raw metrics to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat import build_payload  # noqa: E402 — shipped payload builder

FILLER_PARA = (
    "The printed circuit board of the 1980s was a landscape of through-hole "
    "components, hand-soldered waves, and silkscreened reference designators. "
    "Engineers traced signals with oscilloscopes the size of suitcases and "
    "documented revisions in three-ring binders that smelled of flux and "
    "coffee. Nobody talked about impedance budgets in standups, because there "
    "were no standups, only schedules pinned above drafting tables."
)


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = min(int(q * len(s)), len(s) - 1)
    return round(1000 * s[k], 2)


def stream_one(base_url: str, payload: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
    """One streamed completion; returns latency/token metrics."""
    p = dict(payload)
    p["stream"] = True
    p["stream_options"] = {"include_usage": True}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    last_mark: float | None = None
    itls: list[float] = []
    usage: dict[str, Any] | None = None
    content_chars = 0
    reason_chars = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[len("data: "):]
            if body == "[DONE]":
                break
            try:
                d = json.loads(body)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            delta = ((d.get("choices") or [{}])[0] or {}).get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            if content or reasoning:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                elif last_mark is not None:
                    itls.append(now - last_mark)
                last_mark = now
                content_chars += len(content)
                reason_chars += len(reasoning)
    total = time.perf_counter() - t0
    ctok = (usage or {}).get("completion_tokens")
    ptok = (usage or {}).get("prompt_tokens")
    n_itl = max(len(itls), 1)
    p50 = percentile(itls, 0.5)
    return {
        "ok": ttft is not None,
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_s": round(total, 3),
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "content_chars": content_chars,
        "reasoning_chars": reason_chars,
        "decode_tok_per_s": round(ctok / total, 2) if ctok else None,
        "itl_ms_mean": round(1000 * sum(itls) / n_itl, 2) if itls else None,
        "itl_ms_p50": p50,
    }


def run_case(base_url: str, model: str, label: str, prompt: str, *,
             thinking: bool, max_tokens: int, temperature: float = 0.7) -> dict[str, Any]:
    payload = build_payload(
        model=model,
        prompt=prompt,
        enable_thinking=thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.8,
        top_k=20,
    )
    m = stream_one(base_url, payload)
    m["label"] = label
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 4, 8])
    args = ap.parse_args()

    results: list[dict[str, Any]] = []

    # 1. decode speed, non-thinking (steady-state generation)
    print("[speed] decode non-thinking ...", flush=True)
    results.append(run_case(
        args.base_url, args.model, "decode_nonthink",
        "Write a clear, factual 450-word explanation of how the HTTP protocol "
        "works, covering requests, responses, and status codes.",
        thinking=False, max_tokens=700,
    ))

    # 2. decode speed, thinking mode (reasoning + answer)
    print("[speed] decode thinking ...", flush=True)
    results.append(run_case(
        args.base_url, args.model, "decode_think",
        "A rectangle's length is twice its width. Its perimeter is 36 cm. "
        "What is its area? Reason carefully, then give the final answer.",
        thinking=True, max_tokens=2048, temperature=1.0,
    ))

    # 3. prefill speed: ~8k-token prompt, 1-token answer, non-thinking
    print("[speed] prefill 8k ...", flush=True)
    long_prompt = (FILLER_PARA + "\n\n") * 150 + \
        "Reply with exactly the single word OK and nothing else."
    results.append(run_case(
        args.base_url, args.model, "prefill_8k",
        long_prompt, thinking=False, max_tokens=8, temperature=0.0,
    ))

    # 4. concurrency: identical short-generation requests in parallel
    for n in args.concurrency:
        print(f"[speed] concurrency n={n} ...", flush=True)
        prompt = ("Write a 250-word technical summary of how a CPU branch "
                  "predictor works.")
        payload = build_payload(
            model=args.model, prompt=prompt, enable_thinking=False,
            max_tokens=400, temperature=0.7, top_p=0.8, top_k=20,
        )
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            ms = list(ex.map(lambda _: stream_one(args.base_url, payload), range(n)))
        wall = time.perf_counter() - t0
        toks = sum(m.get("completion_tokens") or 0 for m in ms)
        results.append({
            "label": f"concurrent_{n}",
            "wall_s": round(wall, 3),
            "requests": n,
            "total_completion_tokens": toks,
            "aggregate_tok_per_s": round(toks / wall, 2) if wall else None,
            "per_request": [
                {k: m[k] for k in ("ttft_s", "total_s", "completion_tokens", "decode_tok_per_s")}
                for m in ms
            ],
        })

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"BENCH_SPEED_OK -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
