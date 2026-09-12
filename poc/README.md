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

## One go: photo → avatar in livebody (`poc/web/orchestrator.py`)

```bash
export MESHY_API_KEY=msy_...      # and/or TRIPO_API_KEY; omit both to run the manual path only
export BLENDER=/Applications/Blender.app/Contents/MacOS/Blender   # or --blender
python3 poc/web/orchestrator.py   # http://localhost:8901/  — also serves livebody.html
```

Upload page → job page that walks **generate → rig → convert → done** and ends
with an *open in livebody* link. Stdlib only, no venv. Every job lives in
`poc/models3d/generated/<id>/` (gitignored — inputs are real people).

| stage | API (default: Meshy; Tripo also wired) | fallback (no key / not riggable) |
|---|---|---|
| generate | Meshy `image-to-3d` with `pose_mode: a-pose` (1 photo) or `multi-image-to-3d` (2–4, first = front); Tripo v3 `image-to-model` / `multiview-to-model` (front, left, back, right) | upload a mesh (.glb/.fbx/.obj/.zip) instead of photos |
| rig | Meshy `rigging` (any textured humanoid GLB, ~5 credits) or Tripo `rig-check` → `rig` with `spec: mixamo`, FBX out (~25–30 credits) | **Mixamo by hand**: the page converts the mesh to FBX, pauses with the download link and the Mixamo settings (FBX Binary, T-pose, With Skin), and takes the rigged FBX back |
| convert | `blender --background --python poc/tools/fbx_to_vrm.py -- rigged.fbx avatar.vrm [texture]` | same |

Keys: `MESHY_API_KEY` (meshy.ai → settings → API), `TRIPO_API_KEY` (platform.tripo3d.ai → API Keys).
Generator and rigger are chosen independently, so a Meshy mesh can take a Tripo rig
(guaranteed Mixamo bone names) and an uploaded mesh can take either API rig.
Meshy's bone naming is undocumented; `fbx_to_vrm.py` raises if a required bone is
unmapped, which is the signal to switch rigger. Tripo's v3 generation body is written
from the v2 docs and marked UNVERIFIED in the source; its rig endpoints are confirmed.
Expect 3–8 minutes for an API run; the job page polls and the
server keeps `job.json` so finished avatars survive a restart. Photo advice
is on the upload page: whole body, arms off the torso, plain background —
auto-riggers reject anything else. A rigged body has **no face
blendshapes**, so livebody's blink/mouth stay still on generated characters
until a later step adds VRM expressions.

Before running celebrity photos through it, read the provider's terms on
real-person likeness.

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
