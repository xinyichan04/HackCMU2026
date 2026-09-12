#!/usr/bin/env python3
"""Voice bridge: the thin seam between the LE SSERAFIMIFY page and the
live_voice RVC engine (live/stream_convert.py).

The page polls /status and shows one of three states:
  offline  - engine not present on this machine (no external/rvc checkout,
             no trained weights, or no python env that imports torch)
  ready    - everything found; /start will launch live mic->voice conversion
  LIVE     - stream_convert.py is running

Stdlib only, binds 127.0.0.1. CORS is open because the demo page is served
from a different local port. The bridge never fabricates readiness: it
re-checks the real files on every /status call.

Usage:
    python3 poc/web/voicebridge.py            # port 8903
    RVC_PYTHON=/path/to/env/bin/python3 ...   # python that has torch+rvc deps
"""
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RVC_DIR = REPO_ROOT / "external" / "rvc"
CONFIG = REPO_ROOT / "training" / "config.yaml"
PORT = int(os.environ.get("PORT", "8903"))
RVC_PYTHON = os.environ.get("RVC_PYTHON", sys.executable)

proc = None  # the running stream_convert.py, if any


def model_name() -> str:
    # training/config.yaml names the experiment; the usable weight is
    # <experiment_name>.pth (same rule as stream_convert.default_model_name,
    # re-implemented here so the bridge needs no yaml dependency).
    if CONFIG.is_file():
        for line in CONFIG.read_text().splitlines():
            if line.strip().startswith("experiment_name:"):
                return line.split(":", 1)[1].split("#")[0].strip() + ".pth"
    return "my_voice.pth"


def find_weight() -> Path | None:
    name = model_name()
    for base in (RVC_DIR / "assets" / "weights", REPO_ROOT / "models"):
        if (base / name).is_file():
            return base / name
        if base.is_dir():
            hits = sorted(base.glob("*.pth"))
            if hits:
                return hits[0]
    return None


def check() -> dict:
    global proc
    if proc is not None and proc.poll() is not None:
        proc = None  # engine exited on its own
    running = proc is not None
    if not RVC_DIR.is_dir():
        return {"available": False, "running": running, "model": None,
                "detail": "no external/rvc checkout on this machine"}
    weight = find_weight()
    if weight is None:
        return {"available": False, "running": running, "model": None,
                "detail": f"trained weights not found ({model_name()})"}
    try:
        subprocess.run([RVC_PYTHON, "-c", "import torch"], check=True,
                       capture_output=True, timeout=20)
    except Exception:
        return {"available": False, "running": running, "model": weight.name,
                "detail": "no python env with torch (set RVC_PYTHON)"}
    return {"available": True, "running": running, "model": weight.stem,
            "detail": str(weight)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path == "/status":
            self._send(200, check())
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        global proc
        if self.path == "/start":
            st = check()
            if not st["available"]:
                self._send(409, st)
                return
            if proc is None:
                proc = subprocess.Popen(
                    [RVC_PYTHON, str(REPO_ROOT / "live" / "stream_convert.py")],
                    cwd=REPO_ROOT,
                    stdout=open(REPO_ROOT / "voicebridge.log", "ab"),
                    stderr=subprocess.STDOUT,
                )
            self._send(200, check())
        elif self.path == "/stop":
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                proc = None
            self._send(200, check())
        else:
            self._send(404, {"error": "unknown path"})

    def log_message(self, *a):  # keep the terminal quiet
        pass


if __name__ == "__main__":
    print(f"voice bridge on http://127.0.0.1:{PORT}  (engine: {check()['detail']})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
