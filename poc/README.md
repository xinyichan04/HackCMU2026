# POC — browser 3D head (the picked approach)

Live webcam → MediaPipe Face Landmarker (in-browser) → ARKit-51 blendshapes +
head pose → three.js renders a .glb head over the video. No python, no server
logic — one HTML page and a static file server.

## Run

```bash
poc/web/serve.sh            # serves the repo on http://localhost:8901
# open http://localhost:8901/poc/web/live3d.html
```

Options and hotkeys are printed by `serve.sh` and shown on the page
(`?model=` to swap the .glb, `[` / `]` to resize the head live).

## Layout

- `web/` — `live3d.html` (the app) and `serve.sh`
- `models3d/` — test heads; gitignored, re-fetch with `models3d/fetch.sh`.
  The real character model arrives via `AVATAR3D-HANDOFF.md` (repo root)
- `tools/` — .glb acceptance checks: `glb_inspect.py` (headless/CI) and
  `verify_glb_threejs.html` (browser truth)
- `archive/` — the three earlier approaches (toon 2D avatar, photoreal
  one-shot swap, facepaint) plus the python tracking they shared. Still
  runnable; see `archive/README.md`

## What a model needs

A .glb (or anything convertible to one) whose mesh carries the ARKit-51
morph targets — `tools/glb_inspect.py <model.glb>` grades a candidate in
seconds. A model without morphs still tracks as a rigid head; the face
just won't animate.
