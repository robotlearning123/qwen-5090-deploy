#!/usr/bin/env python3
"""AIME (Maxwell-Jia/AIME_2024, 30 problems) against a local OpenAI-compatible
server. Thinking mode, official-card sampling, \\boxed/last-integer exact match.
Dataset is public; run with --fetch-data once.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_quality import ask  # noqa: E402

DATA_URL = ("https://datasets-server.huggingface.co/rows?dataset=Maxwell-Jia%2FAIME_2024"
            "&config=default&split=train&offset=0&length=100")


def fetch_data(dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as r:
        d = json.load(r)
    rows = [x["row"] for x in d.get("rows", [])]
    if not rows:
        raise SystemExit("dataset server returned no rows")
    dest.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"saved {len(rows)} problems -> {dest}")


def extract_answer(text: str):
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    m2 = re.findall(r"-?\d+", tail)
    return m2[-1] if m2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--data", default=None,
                    help="aime json (default artifacts/bench-dataset/aime2024.json)")
    ap.add_argument("--out", default="aime24-results.json")
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--fetch-data", action="store_true",
                    help="download the public dataset, then exit")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = Path(a.data) if a.data else root / "artifacts" / "bench-dataset" / "aime2024.json"
    if a.fetch_data:
        fetch_data(data)
        return
    if not data.exists():
        raise SystemExit(f"dataset missing: {data}\nrun once: {Path(__file__).name} --fetch-data")
    rows = json.load(open(data))
    out, npass = [], 0
    for i, r in enumerate(rows):
        prompt = r["Problem"] + ("\n\nReason step by step, then give the final "
                                 "integer answer (0-999) inside \\boxed{}.")
        resp = ask(a.base_url, a.model, prompt, thinking=True, max_tokens=a.max_tokens)
        got = extract_answer(resp["content"])
        want = str(r["Answer"]).strip()
        ok = got is not None and (got.lstrip("0") == want.lstrip("0") or got == want)
        npass += bool(ok)
        out.append({"id": r.get("ID", i), "want": want, "got": got, "pass": bool(ok),
                    "ctok": resp["completion_tokens"], "latency_s": resp["latency_s"]})
        print(f"[{i+1}/{len(rows)}] {'PASS' if ok else 'FAIL'} want={want} got={got} "
              f"({resp['latency_s']}s)", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"AIME24: {npass}/{len(rows)} = {npass/len(rows)*100:.1f}%")


if __name__ == "__main__":
    main()
