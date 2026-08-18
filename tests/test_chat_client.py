"""Drive the shipped chat client against a real HTTP OpenAI-shaped server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import chat  # noqa: E402  — shipped entry, not a reimplementation


class _Handler(BaseHTTPRequestHandler):
    last_body: bytes = b""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _Handler.last_body = self.rfile.read(length)
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "101, 103, 107",
                        "reasoning_content": "three primes above 100",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return


def _serve() -> HTTPServer:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def test_build_payload_thinking_defaults() -> None:
    payload = chat.build_payload(
        model="qwen3.8-27b",
        prompt="hi",
        enable_thinking=True,
        max_tokens=32,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
    )
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["messages"][0]["content"] == "hi"


def test_build_payload_thinking_off() -> None:
    payload = chat.build_payload(
        model="qwen3.8-27b",
        prompt="hi",
        enable_thinking=False,
        max_tokens=32,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_completion_ok_rejects_empty_and_error() -> None:
    assert not chat.completion_ok(500, {"choices": []})
    assert not chat.completion_ok(200, {"choices": [{"message": {"content": ""}}]})
    assert not chat.completion_ok(
        200, {"choices": [{"message": {"content": "Error: boom"}}]}
    )
    assert chat.completion_ok(
        200, {"choices": [{"message": {"content": "101, 103, 107"}}]}
    )
    assert chat.completion_ok(
        200, {"choices": [{"message": {"content": "", "reasoning_content": "think"}}]}
    )


def test_live_client_hits_openai_path(tmp_path: Path) -> None:
    httpd = _serve()
    port = httpd.server_address[1]
    out = tmp_path / "chat.json"
    rc = chat.main(
        [
            "--base-url",
            f"http://127.0.0.1:{port}/v1",
            "--out",
            str(out),
            "--prompt",
            "List three primes above 100.",
            "--enable-thinking",
            "--timeout",
            "5",
        ]
    )
    httpd.shutdown()
    assert rc == 0
    saved = json.loads(out.read_text())
    assert saved["http_status"] == 200
    assert saved["request"]["messages"][0]["content"] == "List three primes above 100."
    assert saved["response"]["choices"][0]["message"]["content"] == "101, 103, 107"
    posted = json.loads(_Handler.last_body.decode("utf-8"))
    assert posted["model"] == "qwen3.8-27b"
    assert posted["chat_template_kwargs"]["enable_thinking"] is True
