#!/usr/bin/env python3
"""Canonical HumanEval (164 problems) against a local OpenAI-compatible server.

Feeds the official prompt, extracts the completion, runs prompt+completion
against the official test() in a subprocess. pass@1, thinking mode.
Dataset: openai/openai_humaneval (public). Run with --fetch-data once.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_quality import ask, first_code_block  # noqa: E402

DATA_URL = ("https://huggingface.co/datasets/openai/openai_humaneval/resolve/"
            "main/openai_humaneval/test-00000-of-00001.parquet")


def fetch_data(dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading HumanEval -> {dest}")
    urllib.request.urlretrieve(DATA_URL, dest)


def load_rows(data: Path) -> list[dict]:
    if data.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(data).to_dict("records")
    return json.load(open(data))


def solve_one(args, row):
    base_url, model = args
    instr = ("Complete this Python function. Output ONLY one fenced Python "
             "code block containing the full function (keep the signature):\n\n"
             + row["prompt"])
    r = ask(base_url, model, instr, thinking=True, max_tokens=6000, temperature=0.2)
    comp = first_code_block(r["content"]) or ""
    if re.search(rf"def {row['entry_point']}\(", comp):
        code = comp
    else:
        code = row["prompt"] + comp
    harness = (code + "\n\n" + row["test"]
               + f"\ncheck({row['entry_point']})\nprint('HE_PASS')\n")
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.py"
        f.write_text(harness)
        try:
            out = subprocess.run([sys.executable, str(f)],
                                 capture_output=True, text=True, timeout=30)
            ok = "HE_PASS" in out.stdout
        except subprocess.TimeoutExpired:
            ok = False
    return {"task_id": row["task_id"], "pass": bool(ok),
            "ctok": r["completion_tokens"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--data", default=None,
                    help="humaneval json/parquet (default artifacts/bench-dataset/humaneval.parquet)")
    ap.add_argument("--out", default="humaneval-results.json")
    ap.add_argument("--fetch-data", action="store_true",
                    help="download the public dataset, then exit")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = Path(a.data) if a.data else root / "artifacts" / "bench-dataset" / "humaneval.parquet"
    if a.fetch_data:
        fetch_data(data)
        return
    if not data.exists():
        raise SystemExit(f"dataset missing: {data}\nrun once: {Path(__file__).name} --fetch-data")
    rows = load_rows(data)
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda r: solve_one((a.base_url, a.model), r), rows))
    Path(a.out).write_text(json.dumps(results, indent=1))
    n = sum(r["pass"] for r in results)
    print(f"HumanEval pass@1: {n}/{len(results)} = {n / len(results) * 100:.1f}%")


if __name__ == "__main__":
    main()
