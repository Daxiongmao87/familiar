"""JEV lab server: static form plus same-origin /score proxy.

The openjev-serve endpoint has no CORS headers, so browsers cannot call it
cross-origin. This server (stdlib only) serves index.html and forwards
POST /api/score and GET /api/health to the local openjev endpoint.

Usage: python3 server.py [port]  (default 8093, binds 0.0.0.0)
"""

from __future__ import annotations

import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OPENJEV = "http://127.0.0.1:8199"
HERE = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    server_version = "jevlab/1"

    def log_message(self, fmt: str, *args) -> None:  # quieter than default
        sys.stderr.write("jevlab: %s\n" % (fmt % args))

    def _send_json(self, status: int, obj: object) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        if self.path == "/api/health":
            try:
                with urllib.request.urlopen(f"{OPENJEV}/health", timeout=10) as r:
                    self._send_json(200, json.loads(r.read().decode()))
            except Exception as exc:
                self._send_json(502, {"error": f"openjev unreachable: {exc}"})
            return
        if self.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        if self.path != "/api/score":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            row = json.loads(raw.decode())
        except ValueError:
            self._send_json(400, {"error": "invalid JSON"})
            return
        req = urllib.request.Request(
            f"{OPENJEV}/score",
            data=json.dumps(row).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                self._send_json(200, json.loads(r.read().decode()))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:500]
            self._send_json(exc.code, {"error": f"openjev HTTP {exc.code}: {detail}"})
        except Exception as exc:
            self._send_json(502, {"error": f"openjev unreachable: {exc}"})


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8093
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"jevlab on 0.0.0.0:{port} -> {OPENJEV}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
