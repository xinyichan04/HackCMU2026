# POC — livebody: full-body tracking → VRM avatar (the product)

Live webcam → MediaPipe (in-browser): Pose Landmarker (33 body points) +
Face Landmarker (head pose + expressions) → Kalidokit → a rigged VRM
character mirrors you, full body, ~30 fps. No python, no server logic —
one HTML page and a static file server.

## Run

```bash
poc/web/serve.sh            # serves the repo on http://localhost:8901
# open http://localhost:8901/poc/web/livebody.html
```

Hotkeys on the page: `h` hide avatar · `v` hide video · `m` mirror/direct ·
`d` debug · scroll wheel zooms. `?model=` swaps in any .vrm.

## What a model needs

A **rigged VRM** (VRM 0 or 1): humanoid skeleton for the body, ideally the
standard expression set (blink/aa/happy/surprised) for the face. An unrigged
mesh cannot be body-tracked — the pipeline for turning a generated statue
into a VRM is: regenerate in A-pose → Mixamo auto-rig → Blender VRM export.

## Layout

- `web/` — `livebody.html` (the app) and `serve.sh`
- `models3d/` — test models; gitignored, re-fetch with `models3d/fetch.sh`
- `tools/` — .glb inspection: `glb_inspect.py`, `verify_glb_threejs.html`
- `archive/` — everything superseded, still runnable (see `archive/README.md`):
  the three pre-3D approaches (toon avatar, photoreal swap, facepaint) with
  their python tracking, and **`live3d.html`** — the face-only rigid-head
  page (dropped 2026-09-12 when livebody became the product; it remains the
  only page that drives *unrigged* .glb heads with ARKit-51 morphs, so it
  comes back with one `git mv` if that capability is ever needed)
