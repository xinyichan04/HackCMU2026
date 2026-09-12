# HackCMU2026 — "Be LE SSERAFIM" (POC)

Live camera → face landmarks → stylized LE SSERAFIM-inspired cartoon avatar, on a MacBook, no server.
Output goes to a preview window, an OBS virtual camera (Zoom / FaceTime / OBS), and/or an MP4.

The POC deliberately renders a **clearly stylized avatar**, not a photorealistic swap onto a real
member's face. See `SPEC.md` (revision note at the top) for why and what changed.

## Run

```bash
# once: Python 3.12 venv (mediapipe has no 3.13 wheels yet)
/Users/felixlin/miniconda3/bin/python3.12 -m venv .venv
./.venv/bin/pip install -r poc/requirements.txt

# live preview (downloads the 3.7 MB landmarker model on first run)
./.venv/bin/python poc/live.py --pack poc/packs/le-sserafim --character chaewon

# publish to Zoom / FaceTime (needs OBS installed once for its virtual camera driver)
./.venv/bin/python poc/live.py --character chaewon --virtual-cam

# record
./.venv/bin/python poc/live.py --character chaewon --record demo.mp4

# offline: run on a clip instead of the camera
./.venv/bin/python poc/live.py --source in.mp4 --record out.mp4 --no-preview

# up to 4 people in frame, each gets the next member in the pack
./.venv/bin/python poc/live.py --max-faces 4
```

Run it from Terminal.app (or iTerm) and grant it camera access when macOS asks. IDE sandboxes usually
cannot open the camera.

Hotkeys in the preview: `n`/`p` next/previous character, `r` start/stop recording, `e` toggle landmark
smoothing, `d` debug mesh, `m` mirror, `q` quit.

## Photo mode (photoreal one-shot face swap)

`--mode photo` swaps a real face from a reference photo onto each tracked person instead of drawing a
cartoon. Nothing is trained: InsightFace `buffalo_l` detects faces and computes a 512-d identity embedding,
`inswapper_128` regenerates the face crop as the reference identity with the live pose/expression, and
pastes it back. One photo per character; adding a character = a JSON entry + a photo.

```bash
./.venv/bin/pip install -r poc/requirements.txt            # adds insightface + onnxruntime (CoreML on macOS)
mkdir -p poc/packs/le-sserafim/chaewon && cp ~/Pictures/chaewon.jpg poc/packs/le-sserafim/chaewon/ref.jpg
./.venv/bin/python poc/live.py --mode photo --character chaewon
./.venv/bin/python poc/live.py --mode photo --ref ~/Pictures/someone.jpg     # one-off reference, no pack edit
./.venv/bin/python poc/live.py --mode photo --det-size 320 --detect-every 2  # faster on M1-M3
./.venv/bin/python poc/live.py --mode photo --source in.mp4 --record out.mp4 --no-preview --enhance
```

First run downloads `inswapper_128.onnx` (554 MB) into `poc/models/` and insightface fetches `buffalo_l`
(~300 MB). Reference photos: frontal, evenly lit, no heavy stage makeup, face ≥ 512 px; keep a few candidates
and pick by eye. They are gitignored (`poc/packs/**/*.jpg|png`) because they are real people's likenesses;
every teammate drops their own copies in locally. Expect roughly 5-10 fps on M1-M3 and 13-20 on M4 with
CoreML at 720p for one face; `e` toggles the GFPGAN enhancer (needs `pip install gfpgan basicsr facexlib`,
pulls torch, halves fps: use it for recordings, not live). `d` shows detector boxes. Output is watermarked
"AI face swap". Licences: inswapper/ArcFace/RetinaFace are non-commercial research models.

macOS camera gotcha: if the preview opens but the camera is black or "cannot open camera" with no permission
prompt, a previous deny is cached; run `tccutil reset Camera com.apple.Terminal` and launch again.

## Layout

```
poc/
  live.py        camera / file → track → render → preview / virtual cam / MP4, hotkeys
  tracker.py     MediaPipe Face Landmarker wrapper (478 landmarks, 52 blendshapes, pose, smoothing)
  avatar.py      data-driven cartoon renderer (hair, bangs, eyes, brows, mouth, blush, accessories)
  packs/le-sserafim/pack.json   five characters, all parameters, no code
  models/        gitignored; face_landmarker.task is auto-downloaded
  tests/         pytest: pack loading, tracking, rendering, speed, headless CLI + recording
```

Add a character by adding an object to `pack.json`:

```json
{ "id": "newbie", "name": "Newbie", "skin": "#F8DFCD", "hair": "#5A3B8C", "hair_style": "bob",
  "bangs": "side", "eye": "#3A2A5A", "eye_shape": 0.6, "lip": "#E27A86", "blush": "#F7A6AC",
  "accessories": ["glasses"] }
```

`hair_style`: long | bob | short | ponytail. `bangs`: full | side | none.
`accessories`: star_clip | ribbon | cat_ears | glasses | heart | hoops.

## Tests

```bash
./.venv/bin/python -m pytest poc/tests -q
```

Tests use a drawn synthetic face (`poc/tests/synth.py`) so they run without a camera or any real photo.
