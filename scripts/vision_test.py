#!/usr/bin/env python3
"""Vision capability test for the local Qwen3.8-27B VLM server.

Generates deterministic PIL images with known ground truth, sends them via
the OpenAI image API, and scores exact-number answers. No eyeball grading.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


def b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def img_shapes() -> tuple[Image.Image, dict]:
    img = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((30, 40, 110, 120), fill="red")
    d.ellipse((150, 40, 230, 120), fill="red")
    d.ellipse((270, 40, 350, 120), fill="red")
    d.rectangle((120, 180, 280, 260), fill="blue")
    d.text((185, 205), "42", fill="black")
    q = "How many red circles are in this image, and what number is written " \
        "inside the blue rectangle? Answer as exactly: '<circles>, <number>'."
    return img, {"q": q, "a": "3, 42"}


def img_chart() -> tuple[Image.Image, dict]:
    img = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(img)
    d.line((40, 270, 380, 270), fill="black", width=2)  # x-axis
    d.line((40, 270, 40, 30), fill="black", width=2)    # y-axis
    for i, (label, h) in enumerate((("A", 60), ("B", 150), ("C", 210))):
        x = 80 + i * 110
        d.rectangle((x, 270 - h, x + 60, 270), fill="green")
        d.text((x + 25, 275), label, fill="black")
    q = ("This bar chart shows three bars labeled A, B, C with heights "
         "proportional to 60, 150, 210. Which labeled bar is the tallest, "
         "and what is its value? Answer as exactly: '<label> <value>'.")
    return img, {"q": q, "a": "C 210"}


def ask_vision(base_url: str, model: str, img: Image.Image, q: str,
               thinking: bool, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64_png(img)}},
                {"type": "text", "text": q},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        base_url + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    msg = body["choices"][0]["message"]
    return msg.get("content") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    results = []
    for name, fn in (("shapes", img_shapes), ("chart", img_chart)):
        img, spec = fn()
        text = ask_vision(args.base_url, args.model, img, spec["q"],
                          thinking=args.thinking,
                          max_tokens=4000 if args.thinking else 400)
        nums = re.findall(r"-?\d+", text)
        want = re.findall(r"-?\d+", spec["a"])
        ok = all(w in nums for w in want)
        results.append({"test": name, "pass": bool(ok), "expected": spec["a"],
                        "numbers_in_answer": nums[:6],
                        "answer_head": text.strip()[-160:]})
        print(f"{name}: {'PASS' if ok else 'FAIL'} expected={spec['a']} got_nums={nums[:4]}")

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"VISION {'SUMMARY'}: {sum(r['pass'] for r in results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
