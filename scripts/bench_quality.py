#!/usr/bin/env python3
"""Quality bench for the local Qwen3.8-27B endpoint: coding (executable
oracles), math (known answers), instruction following, long-context needle.

Coding answers are extracted, written to disk and *run* under the venv python
against real assertions — no eyeball grading. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat import build_payload, post_chat  # noqa: E402 — shipped client

FILLER_PARA = (
    "The printed circuit board of the 1980s was a landscape of through-hole "
    "components, hand-soldered waves, and silkscreened reference designators. "
    "Engineers traced signals with oscilloscopes the size of suitcases and "
    "documented revisions in three-ring binders that smelled of flux and "
    "coffee. Nobody talked about impedance budgets in standups, because there "
    "were no standups, only schedules pinned above drafting tables."
)


def ask(base_url: str, model: str, prompt: str, *, thinking: bool,
        max_tokens: int, temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    # Official Qwen3.8 card sampling: thinking 1.0/0.95/20, instruct
    # 0.7/0.80/20 + presence_penalty 1.5 (anti-repetition).
    if temperature is None:
        temperature = 1.0 if thinking else 0.7
    top_p = 0.95 if thinking else 0.80
    payload = build_payload(
        model=model, prompt=prompt, enable_thinking=thinking,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=20,
    )
    if not thinking:
        payload["presence_penalty"] = 1.5
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    t0 = time.perf_counter()
    status, body = post_chat(base_url + "/chat/completions", payload, 900)
    dt = time.perf_counter() - t0
    usage = body.get("usage") or {}
    msg = ((body.get("choices") or [{}])[0]).get("message") or {}
    return {
        "status": status,
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": [
            {"name": (tc.get("function") or {}).get("name"),
             "args": (tc.get("function") or {}).get("arguments")}
            for tc in (msg.get("tool_calls") or [])
        ],
        "finish_reason": ((body.get("choices") or [{}])[0]).get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_s": round(dt, 2),
    }


def first_code_block(text: str) -> str | None:
    m = re.search(r"```(?:[Pp]ython|[Pp]y)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    if re.search(r"^\s*(def |class |import |from )", text, re.MULTILINE):
        return text  # bare code, no fences
    return None


def run_code_oracle(code: str, tests: str, timeout: int = 30) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qwen38-oracle-") as td:
        sol = Path(td) / "sol.py"
        tst = Path(td) / "test_sol.py"
        sol.write_text(code)
        tst.write_text("from sol import *\n" + tests + "\nprint('ORACLE_PASS')\n")
        try:
            r = subprocess.run(
                [sys.executable, str(tst)], capture_output=True, text=True,
                timeout=timeout,
            )
            return {
                "pass": "ORACLE_PASS" in r.stdout,
                "stdout_tail": r.stdout[-500:],
                "stderr_tail": r.stderr[-500:],
                "returncode": r.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"pass": False, "stdout_tail": "", "stderr_tail": "TIMEOUT", "returncode": -1}


# ---------------------------------------------------------------- coding tasks
CODING_TASKS: list[dict[str, Any]] = [
    {
        "id": "code_two_sum",
        "prompt": (
            "Write a Python function `two_sum(nums, target)` that returns the "
            "indices (as a list) of the two numbers adding to target, or an "
            "empty list if none. Output ONLY one fenced Python code block, no "
            "explanation."
        ),
        "tests": (
            "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n"
            "assert two_sum([3, 2, 4], 6) == [1, 2]\n"
            "assert two_sum([3, 3], 6) == [0, 1]\n"
            "assert two_sum([1, 2, 3], 99) == []\n"
            "assert two_sum([], 5) == []\n"
        ),
        "max_tokens": 6000,
    },
    {
        "id": "code_lru_cache",
        "prompt": (
            "Implement class `LRUCache(capacity)` with `get(key)` returning "
            "value or -1, and `put(key, value)` evicting the least recently "
            "used when full. O(1) operations. Output ONLY one fenced Python "
            "code block, no explanation."
        ),
        "tests": (
            "c = LRUCache(2)\n"
            "c.put(1, 1); c.put(2, 2)\n"
            "assert c.get(1) == 1\n"
            "c.put(3, 3)  # evicts key 2\n"
            "assert c.get(2) == -1\n"
            "c.put(4, 4)  # evicts key 1\n"
            "assert c.get(1) == -1\n"
            "assert c.get(3) == 3\n"
            "assert c.get(4) == 4\n"
            "c2 = LRUCache(1); c2.put('a', 10); c2.put('b', 20)\n"
            "assert c2.get('a') == -1 and c2.get('b') == 20\n"
        ),
        "max_tokens": 3000,
    },
    {
        "id": "code_debug_bisect",
        "prompt": (
            "This binary search has bugs:\n"
            "```python\n"
            "def bsearch(a, x):\n"
            "    lo, hi = 0, len(a)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if a[mid] < x:\n"
        "            lo = mid\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return lo if a and lo < len(a) and a[lo] == x else -1\n"
        "```\n"
            "Fix it so it returns the index of x in sorted list a, or -1. "
            "Output ONLY one fenced Python code block with the fixed function "
            "`bsearch(a, x)`, no explanation."
        ),
        "tests": (
            "import random\n"
            "assert bsearch([1, 3, 5, 7, 9, 11], 7) == 3\n"
            "assert bsearch([1, 3, 5, 7, 9, 11], 1) == 0\n"
            "assert bsearch([1, 3, 5, 7, 9, 11], 11) == 5\n"
            "assert bsearch([1, 3, 5, 7, 9, 11], 4) == -1\n"
            "assert bsearch([], 3) == -1\n"
            "assert bsearch([42], 42) == 0\n"
            "a = sorted(random.Random(7).sample(range(1000), 200))\n"
            "for i, v in enumerate(a):\n"
            "    assert bsearch(a, v) == i, (i, v)\n"
            "assert bsearch(a, -1) == -1 and bsearch(a, 1001) == -1\n"
        ),
        "max_tokens": 3000,
    },
    {
        "id": "code_flatten_json",
        "prompt": (
            "Write `flatten(d, sep='.')` that flattens nested dicts into one "
            "dict with compound keys, preserving lists as values (lists are "
            "NOT flattened). Output ONLY one fenced Python code block, no "
            "explanation."
        ),
        "tests": (
            "assert flatten({'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}) == "
            "{'a.b': 1, 'a.c.d': 2, 'e': 3}\n"
            "assert flatten({'x': [1, 2, {'y': 3}]}) == {'x': [1, 2, {'y': 3}]}\n"
            "assert flatten({}) == {}\n"
            "assert flatten({'a': {'b': {'c': {'d': 4}}}}) == {'a.b.c.d': 4}\n"
        ),
        "max_tokens": 6000,
    },
]

MATH_TASKS: list[dict[str, Any]] = [
    {
        "id": "math_boxes", "answer": "53",
        "prompt": "Tom has 3 boxes, each holding 12 pencils. He gives away 7 pencils and buys 2 more boxes of 12. How many pencils does he have now? End your final answer with the number alone on the last line.",
    },
    {
        "id": "math_discount", "answer": "54",
        "prompt": "A jacket costs $80. A 25% discount is applied, then an extra 10% off the discounted price at checkout. What is the final price in dollars? End with the number alone on the last line.",
    },
    {
        "id": "math_trains", "answer": "2",
        "prompt": "Two trains start 300 km apart and drive toward each other at 70 km/h and 80 km/h. How many hours until they meet? End with the number alone on the last line.",
    },
    {
        "id": "math_arith", "answer": "395",
        "prompt": "Compute 17*24 - 39/3. End with just the final number on the last line.",
    },
    {
        "id": "math_prime", "answer": "151",
        "prompt": "What is the smallest prime number strictly greater than 150? End with just the number on the last line.",
    },
]

INSTRUCTION_TASKS: list[dict[str, Any]] = [
    {
        "id": "inst_three_words",
        "prompt": "Reply with exactly three words, nothing else — no punctuation.",
        "check": lambda c: len(c.strip().split()) == 3,
        "note": "exactly 3 words",
    },
    {
        "id": "inst_chinese",
        "prompt": "请用不超过20个字直接回答：法国的首都是哪座城市？",
        "check": lambda c: "巴黎" in c and len(c.strip()) <= 30,
        "note": "contains 巴黎, <=30 chars",
    },
    {
        "id": "inst_json",
        "prompt": "Return ONLY a valid JSON object (no markdown, no prose) with keys: "
                 "\"a\" = integer 5, \"b\" = string \"hi\", \"c\" = list [1,2,3].",
        "check": lambda c: _json_ok(c),
        "note": "parses as JSON with exact values",
    },
]


def _json_ok(c: str) -> bool:
    try:
        d = json.loads(c.strip())
        return d == {"a": 5, "b": "hi", "c": [1, 2, 3]}
    except json.JSONDecodeError:
        return False


def needle_task(depth_frac: float, passphrase: str, filler_reps: int) -> dict[str, Any]:
    paras = [FILLER_PARA] * filler_reps
    pos = int(depth_frac * filler_reps)
    paras.insert(pos, f"Note for the archivist: the magic passphrase for the archive is {passphrase}.")
    prompt = "\n\n".join(paras) + (
        "\n\nWhat is the magic passphrase for the archive? Reply with only the passphrase."
    )
    return {"id": f"needle_depth{int(depth_frac*100)}", "prompt": prompt, "answer": passphrase}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results: list[dict[str, Any]] = []

    for t in CODING_TASKS:
        print(f"[quality] {t['id']} ...", flush=True)
        r = ask(args.base_url, args.model, t["prompt"], thinking=True,
                max_tokens=t["max_tokens"])
        code = first_code_block(r["content"])
        if code:
            oracle = run_code_oracle(code, t["tests"])
        else:
            oracle = {"pass": False, "stderr_tail": "no code block extracted"}
        results.append({"id": t["id"], "pass": oracle["pass"], "oracle": oracle,
                        "resp": r, "code": code})
        print(f"  -> {'PASS' if oracle['pass'] else 'FAIL'} ({r['completion_tokens']} ctok, {r['latency_s']}s)", flush=True)

    for t in MATH_TASKS:
        print(f"[quality] {t['id']} ...", flush=True)
        r = ask(args.base_url, args.model, t["prompt"], thinking=True,
                max_tokens=1500)
        last_line = r["content"].strip().splitlines()[-1] if r["content"].strip() else ""
        lastnum = re.findall(r"-?\d+(?:\.\d+)?", last_line)
        got = lastnum[-1] if lastnum else None
        try:
            ok = got is not None and abs(float(got) - float(t["answer"])) < 1e-6
        except ValueError:
            ok = False
        results.append({"id": t["id"], "pass": bool(ok), "expected": t["answer"],
                        "got": got, "resp": r})
        print(f"  -> {'PASS' if ok else 'FAIL'} expected={t['answer']} got={got}", flush=True)

    for t in INSTRUCTION_TASKS:
        print(f"[quality] {t['id']} ...", flush=True)
        r = ask(args.base_url, args.model, t["prompt"], thinking=False,
                max_tokens=300)
        ok = r["status"] == 200 and t["check"](r["content"])
        results.append({"id": t["id"], "pass": bool(ok), "note": t["note"], "resp": r})
        print(f"  -> {'PASS' if ok else 'FAIL'} content={r['content'][:80]!r}", flush=True)

    # long-context needle: 180 paras × ~82 tok ≈ 14.7k-token context (16321 was 1 over 16384 at 200)
    for depth in (0.1, 0.5, 0.9):
        t = needle_task(depth, "ZEBRA-4912", 180)
        print(f"[quality] {t['id']} ...", flush=True)
        r = ask(args.base_url, args.model, t["prompt"], thinking=False,
                max_tokens=64, temperature=0.0)
        ok = "ZEBRA-4912" in r["content"]
        results.append({"id": t["id"], "pass": bool(ok),
                        "prompt_tokens": r["prompt_tokens"], "expected": t["answer"],
                        "got": r["content"].strip()[:60], "resp": r})
        print(f"  -> {'PASS' if ok else 'FAIL'} ptok={r['prompt_tokens']} got={r['content'].strip()[:40]!r}", flush=True)

    # tool calling through the qwen3_coder parser
    print("[quality] tool_call ...", flush=True)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    r = ask(args.base_url, args.model,
            "What's the weather in Tokyo right now? Use the tool.",
            thinking=False, max_tokens=300, tools=tools)
    tc = r["tool_calls"]
    ok = bool(tc) and tc[0]["name"] == "get_weather" and "Tokyo" in (tc[0]["args"] or "")
    results.append({"id": "tool_call", "pass": bool(ok), "tool_calls": tc, "resp": r})
    print(f"  -> {'PASS' if ok else 'FAIL'} tool_calls={tc}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    npass = sum(1 for r in results if r["pass"])
    print(f"\nQUALITY SUMMARY: {npass}/{len(results)} PASS -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
