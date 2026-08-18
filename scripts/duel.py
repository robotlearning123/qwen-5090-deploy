#!/usr/bin/env python3
"""Cross-harness duel: run the same coding tasks through different agent CLIs
(any local or remote lane) and grade with the same executable
oracles used for the local qwen3.8 arm (bench_quality.py tasks).

Usage: duel.py --arm "glm-5.3|<cli> -p '{prompt}'" --arm "grok-4.5|<cli> -p '{prompt}'" --out X.json
  {prompt} in the command template is replaced per task.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_quality import CODING_TASKS, run_code_oracle  # noqa: E402


def last_code_block(text: str):
    """Last fenced python block — skips blocks echoed inside the prompt."""
    import re
    blocks = re.findall(r"```[Pp]ython[\w.]*\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None

FIB_PROMPT = (
    "Create fib.py with a function fib(n) returning the nth Fibonacci number "
    "(fib(0)=0, fib(1)=1), plus a __main__ block printing fib(10). Then run it "
    "with python3 and show the output."
)


def run_cli(cmd_template: str, prompt: str, timeout: int, cwd: str) -> tuple[str, float]:
    cmd = cmd_template.replace("{prompt}", prompt.replace("'", "'\\''"))
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
        out = r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired as e:
        so = e.stdout
        if isinstance(so, bytes):
            so = so.decode("utf-8", "replace")
        out = (so or "") + "\nTIMEOUT"
    return out, time.perf_counter() - t0


def fib_oracle(td: str) -> bool:
    p = Path(td) / "fib.py"
    if not p.exists():
        return False
    r = subprocess.run(
        [sys.executable, str(p)], capture_output=True, text=True, timeout=30
    )
    return r.stdout.strip().endswith("55")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name|command-template-with-{prompt}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    results = []
    for arm in args.arm:
        name, tmpl = arm.split("|", 1)
        for t in CODING_TASKS:
            print(f"[duel:{name}] {t['id']}", flush=True)
            with tempfile.TemporaryDirectory() as td:
                out, dt = run_cli(tmpl, t["prompt"], args.timeout, td)
                code = last_code_block(out)
                oracle = (run_code_oracle(code, t["tests"])
                          if code else {"pass": False, "stderr_tail": "no code block"})
            results.append({"arm": name, "task": t["id"], "pass": oracle["pass"],
                            "latency_s": round(dt, 1), "stderr": oracle.get("stderr_tail", "")[-150:]})
            print(f"  -> {'PASS' if oracle['pass'] else 'FAIL'} in {dt:.0f}s", flush=True)
        print(f"[duel:{name}] fib_agentic", flush=True)
        with tempfile.TemporaryDirectory() as td:
            out, dt = run_cli(tmpl, FIB_PROMPT, args.timeout, td)
            ok = fib_oracle(td)
            ran = "55" in out
        results.append({"arm": name, "task": "fib_agentic",
                        "pass": bool(ok and ran), "file_ok": ok, "ran_ok": ran,
                        "latency_s": round(dt, 1)})
        print(f"  -> {'PASS' if (ok and ran) else 'FAIL'} in {dt:.0f}s", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    for arm in [a.split("|", 1)[0] for a in args.arm]:
        rs = [r for r in results if r["arm"] == arm]
        print(f"{arm}: {sum(r['pass'] for r in rs)}/{len(rs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
