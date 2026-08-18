#!/usr/bin/env python3
"""Hit a live OpenAI-compatible Qwen3.8-27B endpoint and write raw JSON."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_PROMPT = "List three prime numbers greater than 100. Reply with just the three numbers."


def build_payload(
    *,
    model: str,
    prompt: str,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    return payload


def post_chat(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            parsed = json.loads(body.decode("utf-8"))
            return int(resp.status), parsed
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": raw}
        return int(e.code), parsed
    except urllib.error.URLError as e:
        print(f"endpoint unreachable: {url} ({e.reason})\n"
              "start the server first: scripts/serve.sh (or pass --base-url)",
              file=sys.stderr)
        raise SystemExit(1)


def extract_assistant(body: dict[str, Any]) -> tuple[str, str]:
    """Return (content, reasoning) from a chat-completions body."""
    choices = body.get("choices") or []
    if not choices:
        return "", ""
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content), str(reasoning)


def completion_ok(status: int, body: dict[str, Any]) -> bool:
    if status != 200:
        return False
    content, reasoning = extract_assistant(body)
    text = (content + reasoning).strip()
    if not text:
        return False
    lowered = text.lower()
    if "error" in body and not content and not reasoning:
        return False
    # Reject a 200 whose only payload is an engine error string.
    if lowered.startswith("error:") or lowered.startswith("internal server error"):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--out", required=True, help="Path to write the raw JSON body")
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    args = p.parse_args(argv)

    if args.enable_thinking:
        temperature = 1.0 if args.temperature is None else args.temperature
        top_p = 0.95 if args.top_p is None else args.top_p
        top_k = 20
    else:
        temperature = 0.7 if args.temperature is None else args.temperature
        top_p = 0.80 if args.top_p is None else args.top_p
        top_k = 20

    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = build_payload(
        model=args.model,
        prompt=args.prompt,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    status, body = post_chat(url, payload, timeout=args.timeout)
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"http_status": status, "request": payload, "response": body}, f, indent=2)
        f.write("\n")

    content, reasoning = extract_assistant(body)
    print(f"http_status={status} out={out_path}")
    print(f"content_len={len(content)} reasoning_len={len(reasoning)}")
    if content:
        print("--- content ---")
        print(content[:2000])
    if reasoning:
        print("--- reasoning ---")
        print(reasoning[:2000])

    if not completion_ok(status, body):
        print("FAIL: empty or error completion", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
