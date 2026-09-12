#!/usr/bin/env bash
# Serve the repo so the browser can load the page, the .glb and the WASM.
# Camera access needs a secure context — http://localhost counts as one, file:// does not.
set -euo pipefail
PORT="${1:-8901}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "serving $ROOT on http://localhost:$PORT"
echo
echo "   open:  http://localhost:$PORT/poc/web/lesserafimify.html   THE experience (up to 5 bodies)"
echo "          http://localhost:$PORT/poc/web/livebody.html        single-body dev page"
echo
echo "   options:  ?model=../models3d/vrm-sample.vrm          swap the model (.vrm)"
echo "             ?voice=http://127.0.0.1:8903                voice bridge URL (lesserafimify)"
echo
cd "$ROOT"

# Refresh music/index.json so local dev exercises the SAME discovery path as a
# static host (Pages, S3, ...), which serves no directory listing to fall back on.
python3 - <<'PY'
import json, os, re
d = os.path.join('poc', 'web', 'music')
if os.path.isdir(d):
    names = sorted(f for f in os.listdir(d) if re.search(r'\.(mp3|m4a|ogg|wav)$', f, re.I))
    with open(os.path.join(d, 'index.json'), 'w') as fh:
        json.dump(names, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
    print(f'   music: {len(names)} track(s) indexed')
PY
echo

exec python3 -m http.server "$PORT" --bind 127.0.0.1
