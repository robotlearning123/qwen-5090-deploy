#!/usr/bin/env python3
"""Bridge for cc <-> vLLM /v1/messages.

Necessary, not just debugging: cc's direct requests lose the user
turn by the time vLLM's anthropic endpoint parses them (reproduced
2026-08-17: direct -> model sees only "-p"; via this buffered bridge ->
full task text arrives). The bridge re-serializes with an exact
content-length, which is what fixes it.

Forwards Anthropic-protocol requests to the vLLM server and dumps each
request body to a scratch dir so you can see exactly what cc sent.
Usage: python cc_bridge.py [listen_port] [upstream_base] [dump_dir]
"""

from __future__ import annotations

import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
UPSTREAM = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
DUMP = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/cc-proxy-dumps")
DUMP.mkdir(parents=True, exist_ok=True)
HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}


class Handler(BaseHTTPRequestHandler):
    n = 0

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("content-length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self._proxy("GET")

    def _proxy(self, method: str) -> None:
        Handler.n += 1
        n = Handler.n
        body = b""
        if length := int(self.headers.get("content-length") or 0):
            body = self.rfile.read(length)
        (DUMP / f"{n:03d}-{method}.json").write_bytes(body)
        req = urllib.request.Request(
            UPSTREAM + self.path, data=body if body else None, method=method,
            headers={k: v for k, v in self.headers.items() if k.lower() not in HOP},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            (DUMP / f"{n:03d}-ERROR-{e.code}.json").write_bytes(data)
            self.send_response(e.code)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        self._proxy("POST")

    def log_message(self, *a) -> None:  # silence
        pass


if __name__ == "__main__":
    print(f"proxy :{LISTEN_PORT} -> {UPSTREAM}, dumps in {DUMP}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
