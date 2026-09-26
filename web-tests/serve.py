"""Serve web/ locally with the same security headers as vercel.json (port 4173)."""

import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {h["key"]: h["value"] for h in json.loads((ROOT / "vercel.json").read_text())["headers"][0]["headers"]}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript"}

    def end_headers(self):
        for k, v in HEADERS.items():
            self.send_header(k, v)
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4173
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT / "web"))).serve_forever()
