#!/usr/bin/env bash
# Serve the repo so the browser can load the page, the .glb and the WASM.
# Camera access needs a secure context — http://localhost counts as one, file:// does not.
set -euo pipefail
PORT="${1:-8901}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "serving $ROOT on http://localhost:$PORT"
echo
echo "   open:  http://localhost:$PORT/poc/web/livebody.html"
echo
echo "   options:  ?model=../models3d/vrm-sample.vrm          swap the model (.vrm)"
echo
cd "$ROOT"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
