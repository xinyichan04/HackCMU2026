# 3D avatar handoff — tracker infrastructure, and what the `.glb` path has to build

Written 2026-09-11 for whoever picks up the **rigged-3D-head** route (option (b)).
Everything below was verified by running the code in this repo, not recalled. Claims that are
measured rather than documented are tagged.

---

## 0. The one-paragraph version

This repo already **produces the driving signal** for a 3D avatar: per frame, over UDP or JSONL, it
emits 51 ARKit-named blendshape coefficients plus a head pose, in a stable JSON record, at 30 fps on
an M4. **Nothing in this repo can draw a 3D model.** Your job is the render half: take that stream,
drive a `.glb`/`.vrm` head with it, and put it on screen. The two known gotchas before you start are
**§4 (the head-pose fields are not angles)** and **§6.3 (you may want to skip this repo's live path
entirely and track in the browser)**.

---

## 1. What exists today

Branch `felix` is canonical. Python 3.12 venv at `./.venv` (mediapipe has no 3.13 wheels).

```
poc/tracker.py    MediaPipe Face Landmarker wrapper. 478 landmarks, 52 blendshape
                  categories, geometric head pose, optional temporal smoothing.
poc/track.py      The probe + the data tap. Live OpenCV window; emits the JSON record
                  over --udp-out / --jsonl-out. Also accepts --source udp:PORT as an
                  INPUT, so an iPhone running ARKit can replace the camera later.
poc/bodytrack.py  33-point body pose (multi-person, world coords, segmentation mask)
                  + 21-point hands. Same record, under "bodies" / "hands".
poc/facepaint.py  Option (a): 2D texture warped onto the mesh. Unrelated to your work
                  except as the thing you are trying to beat on hair/silhouette.
poc/assets/       canonical_face_model.obj — MediaPipe's 468-vertex mesh + UV layout.
                  NOT shipped in the mediapipe pip wheel; vendored here.
poc/live.py       Older cartoon + inswapper face-swap modes. Not your path.
poc/tests/        9 pytest tests, camera-free (drawn synthetic face in synth.py).
```

Run the tap:

```bash
./.venv/bin/python poc/track.py --udp-out 127.0.0.1:9001 --no-landmarks-out
./.venv/bin/python poc/track.py --jsonl-out /tmp/session.jsonl      # record for replay
./.venv/bin/python poc/track.py --list-cameras                      # find the iPhone index
```

Must be launched from an app that holds camera permission (Terminal.app / iTerm). An IDE terminal
opens a black window. Silent-deny fix: `tccutil reset Camera com.apple.Terminal`.

---

## 2. The data contract

One JSON object per frame. As a **UDP datagram** (`--udp-out HOST:PORT`) and/or one **line** in a
JSONL file (`--jsonl-out PATH`). Both can be on at once. Real sample, truncated:

```json
{
  "t": 1234.567,
  "frame": 41,
  "faces": [
    {
      "id": 0,
      "source": "mediapipe",
      "bbox": [493, 227, 793, 558],
      "yaw": -0.0016, "pitch": 0.5598, "roll": 0.0026,
      "blendshapes": { "_neutral": 0.0, "browDownLeft": 0.0012, "browInnerUp": 0.3442, "...": 0.0 },
      "landmarks": [[640.6, 439.3], [643.3, 387.2], "... 478 pairs ..."]
    }
  ]
}
```

| field | meaning |
|---|---|
| `t` | `time.monotonic()` seconds, 3 dp. Monotonic — **not** wall clock, not comparable across processes. |
| `frame` | frame counter from process start. |
| `faces[].id` | index within the frame, `0..max_faces-1`. **Re-assigned per frame — not a stable identity.** With multiple faces, expect swaps. |
| `faces[].source` | `"mediapipe"` (camera) or `"arkit-udp"` (iPhone input). |
| `faces[].bbox` | `[x0,y0,x1,y1]` integer pixels, landmark extremes. `null` when source is arkit-udp. |
| `yaw`/`pitch`/`roll` | **See §4 before using these.** |
| `blendshapes` | name → float, 4 dp. See §3. |
| `landmarks` | 478 `[x,y]` pixel pairs, 1 dp. No z. `null` when source is arkit-udp. Omitted entirely under `--no-landmarks-out`. |

`bodies[]` and `hands[]` appear only with `--body` / `--hands`.

### Things that will bite you

- **No packet is sent on a frame with nothing detected.** The emit is gated on having at least one
  face/body/hand. Your renderer must hold the last pose and decide its own timeout — silence means
  "lost tracking", and it is common (look away from the camera and it stops).
- **`landmarks` and `bbox` can be `null`** whenever the source is an iPhone rather than the camera.
  Handle it from day one or the ARKit upgrade path breaks you.
- **The frame is mirrored by default** (`--no-mirror` turns it off), so landmark x and the sign of
  yaw are in selfie space. Decide which convention your scene uses and state it.
- **UDP packet size** [measured]: **~9.1 KB** with landmarks, **~1.4 KB** with `--no-landmarks-out`.
  9 KB is legal (under the 64 KB datagram cap) but well over a typical 1500-byte MTU, so it
  fragments — one lost fragment drops the whole frame. **Use `--no-landmarks-out` for a 3D rig**;
  you need blendshapes and pose, not 478 2D points.
- **Landmarks are smoothed, blendshapes are not.** `tracker.py` applies an EMA (weight 0.35) to
  landmark positions only. Coefficients arrive raw and are visibly jittery. Smooth them on your
  side — a per-coefficient EMA around 0.5–0.7 is the obvious first try [estimate].

---

## 3. The blendshapes — precise ARKit parity

**52 categories are emitted, but that is not ARKit's 52.** [measured — full list dumped from the
model in this repo]

- `_neutral` is MediaPipe's own and is **not a morph target**. Never send it to the rig.
- The other **51 are exactly ARKit's names**, spelled identically.
- **ARKit's `tongueOut` is absent.** ARKit's 52 = these 51 + `tongueOut`. If Chaewon's model has a
  tongue morph, nothing in this pipeline will ever drive it.

So: **51 drivable coefficients**, all `0.0–1.0`.

```
browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight
cheekPuff cheekSquintLeft cheekSquintRight
eyeBlinkLeft eyeBlinkRight eyeLookDownLeft eyeLookDownRight eyeLookInLeft eyeLookInRight
eyeLookOutLeft eyeLookOutRight eyeLookUpLeft eyeLookUpRight eyeSquintLeft eyeSquintRight
eyeWideLeft eyeWideRight
jawForward jawLeft jawOpen jawRight
mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight mouthFunnel
mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight mouthPucker
mouthRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper
mouthSmileLeft mouthSmileRight mouthStretchLeft mouthStretchRight
mouthUpperUpLeft mouthUpperUpRight
noseSneerLeft noseSneerRight
```

**Key by name, never by index.** The order happens to be `_neutral` then alphabetical, but nothing
guarantees it across model versions.

`Left`/`Right` are **the subject's** left and right, as in ARKit — not the viewer's. On a mirrored
preview the subject's left eye appears on the right of the image. Get this wrong and blinks land on
the wrong eye, which is the classic symptom.

---

## 4. Head pose — the current fields are NOT angles

Read this before you write any rotation code.

`tracker.py::_pose()` derives pose **geometrically from 2D landmarks**, as a cheap heuristic for a
2D overlay. What you actually get:

- `roll` — **radians**, real: `atan2` across the eye line.
- `yaw` — a **normalised ratio clipped to `[-1, +1]`**, from how the nose sits between the face-oval
  extremes, scaled by a hand-tuned `2.2`. Not radians, not degrees, no calibrated zero.
- `pitch` — same: a **normalised `[-1, +1]`** value from where the nose falls between eyes and chin,
  through a hand-tuned `(0.42 - t) * 4.0`.

These are fine for tilting a 2D sprite. **They are not good enough to orient a 3D head** — no real
units, no roll/yaw/pitch decomposition you can trust, and no translation at all.

### What to use instead

MediaPipe will hand you a proper **4×4 facial transformation matrix**, and this repo currently has
it switched off. In `poc/tracker.py`:

```python
output_facial_transformation_matrixes=False,   # <- flip to True
```

Then `res.facial_transformation_matrixes[i]` is a 4×4 [measured, confirmed working — I enabled it
and dumped one]: upper-left 3×3 is rotation, the last column is translation. On a face filling a
720p frame the translation came out ≈ `(0.16, -0.58, -32.5)`, consistent with MediaPipe's metric
canonical face model, so **units look like centimetres** [estimate — inferred from magnitude, not
from documentation].

This is a ~5-line change to `tracker.py` plus a field in `track.py::record()`. **It is not done.**
Either do it yourself or ask the session that owns the tracker. Do not try to reconstruct a usable
3D head rotation from the existing `yaw`/`pitch` — it will look wrong and you will waste hours.

---

## 5. What the `.glb` must satisfy

In priority order:

1. **Morph targets named per ARKit's convention** — `jawOpen`, `eyeBlinkLeft`, `mouthSmileRight`…
   Match §3's list and it's a 1:1 hookup with zero mapping code. Different names are survivable
   (write a lookup table). **No morph targets at all = the face cannot emote, only turn.**
   - glTF stores these names in `mesh.extras.targetNames`; three.js surfaces them as
     `mesh.morphTargetDictionary` / `morphTargetInfluences`. **Verify the names survived export** —
     some exporters drop `targetNames` and you are left with anonymous indices.
   - `.vrm` also works (it is glTF underneath) but VRM 1.0 wraps morphs in its own *expression*
     system; you drive expressions, not raw morph names. VRoid Studio exports ARKit-compatible
     shapes. Ready Player Me exports `.glb` with ARKit naming directly.
2. **A head bone, or a clean root**, so head rotation can be applied separately from the face.
3. **glTF conventions**: Y-up, **facing +Z**, textures embedded (that is what `.glb` buys over `.gltf`).

   > **CORRECTED 2026-09-12 — this line previously said "facing −Z" and that was wrong.** It was
   > taken on trust and it propagated into a generated model before anyone checked it.
   >
   > Measured from `poc/models3d/facecap.glb`, a real ARKit asset, by loading it in three.js and
   > taking the world-space centroid of the vertices each morph actually displaces:
   >
   > ```
   > eyeBlink_L   centroid x=+0.397 y=+0.251 z=+0.390
   > eyeBlink_R   centroid x=-0.374 y=+0.277 z=+0.396
   > noseSneer_L  centroid x=+0.226 y=-0.008 z=+0.561
   > mouthSmile_L centroid x=+0.279 y=-0.426 z=+0.500
   > ```
   >
   > Every facial feature sits at **positive z**, so the face looks down **+Z** — which is also what
   > puts it toward three.js's default camera, since that camera looks down −Z. And the subject's
   > left eye is at **+X**.
   >
   > **The two conventions are locked together — you cannot mix them.** Rotate that head 180° about
   > Y so it faces −Z and every x flips: the subject's left eye lands at **−X**. So:
   >
   > | face direction | subject's left is at |
   > |---|---|
   > | **+Z** (correct, matches ARKit assets and three.js) | **+X** |
   > | −Z | −X |
   >
   > Building a head that faces −Z while placing the subject's left at +X produces a **mirrored
   > rig**: `eyeBlinkLeft` closes the model's anatomical *right* eye. Names cannot catch this, and
   > it is the exact "blinks land on the wrong eye" symptom §3 warns about.
4. **Budget**: under ~100k triangles, 1–2 textures.
5. `.obj` is **not usable** — no rig, no morph targets, geometry only. `.fbx` works but needs
   conversion.

---

## 6. What you have to build

### 6.1 The renderer

Nothing here can draw a `.glb`. `track.py` renders with OpenCV, which draws 2D primitives into a
numpy array — it has no camera, lights, shaders or depth buffer.

**Recommended: Three.js in the browser.** It loads GLB and VRM natively, has `GLTFLoader`,
morph-target support and skinning built in, and avoids fighting macOS's deprecated OpenGL.
Python options (pyrender, moderngl) exist but that OpenGL deprecation is a real time sink on this
machine [secondary — reasoning, not benchmarked here].

### 6.2 A transport bridge — if you keep tracking in Python (read §6.3 first)

**A browser page cannot receive UDP datagrams.** There is no API for it. So you need one of:

- a small **WebSocket bridge**: a ~30-line Node or Python process that binds the UDP port
  `track.py` sends to and forwards each datagram as a WS message to the page; or
- teach `track.py` to speak WebSocket directly (a `--ws-out` flag).

The bridge is the lower-risk option — it leaves `track.py` untouched and lets you restart the
renderer without restarting tracking.

### 6.3 The camera image — and why you may not want this bridge at all

**Added after a reader correctly spotted that §6.2 gets the numbers into the browser but not the
picture.** A head that occludes the real head has to composite over the video, so the page needs
frames, not just coordinates.

**Can `track.py` and a browser hold the camera at once? Yes** [measured — two processes opened the
default camera simultaneously and both read frames]. So the obvious design (page calls
`getUserMedia`, bridge unchanged) is *possible*.

**But it is the wrong design, for a reason that is not about permissions.** Two independent capture
streams are **not frame-synchronised**. `track.py` grabs, tracks and sends coefficients for *its*
frame; the page displays a *different* frame it grabbed itself. The avatar is therefore posed from a
moment the viewer never saw, and the offset varies with latency. A head meant to sit exactly over
the real one visibly swims against it — and the error grows the faster the subject moves, which for
a dance demo is precisely the case that matters.

**Strongly consider doing the tracking in the browser instead.** MediaPipe ships a JS/WASM build:

```html
<script type="module">
  import { FaceLandmarker, FilesetResolver } from
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";
</script>
```

[measured — `@mediapipe/tasks-vision` v1.0.1 resolves on the CDN, browser bundle present]. It runs
the same Face Landmarker, returns the same blendshape categories, and exposes
`facialTransformationMatrixes` — **the 4×4 that §4 says you otherwise have to go enable in Python.**

Doing that collapses four problems at once: no UDP, no WebSocket bridge, no camera sharing, and no
synchronisation error, because one process owns the camera and the frame you track is the frame you
draw on.

`track.py` keeps its value as the probe, the recorder (`--jsonl-out`, for deterministic replay while
you build) and the path for genuine ARKit data off an iPhone. It just does not have to be in the
live loop. **Decide this before building the bridge in §6.2** — the bridge is only worth writing if
you have a specific reason to keep tracking in Python.

### 6.4 Frame pacing

The stream is ~30 fps but irregular, and **silent whenever tracking is lost** (§2). Drive your
render loop from `requestAnimationFrame` and treat incoming packets as *state updates*, not as
frame triggers. Interpolate toward the latest values; do not block waiting for a packet.

---

## 7. Developing without a camera

The camera is a shared resource and needs a GUI session, so build against recorded data:

```bash
./.venv/bin/python poc/track.py --jsonl-out /tmp/session.jsonl --no-landmarks-out   # record once
```

Then replay: read the lines, respect the `t` deltas, push them over UDP/WS at ~30 fps. **A replay
script does not exist yet** — it is ~20 lines and worth writing first, because it makes your
renderer work deterministic and testable.

`poc/tests/synth.py` draws a synthetic face that the tracker detects, which is how the existing 9
tests run with no camera and no real photo. Follow that pattern for anything you add.

---

## 8. Settled facts — do not re-litigate

These were established and communicated already:

- **ARKit face tracking is iOS-only.** `ARFaceTrackingConfiguration` has no macOS target. Xcode on
  the Mac builds and deploys it *to the phone*. This is the reason the whole UDP-source path exists.
- **Continuity Camera gives the Mac RGB only.** TrueDepth and LiDAR depth are not exposed over it.
  Real depth requires an iOS app streaming over — same bridge pattern.
- **MediaPipe needs a roughly frontal face.** A hard-profile frame correctly reports 0 faces
  [measured]. That is the RGB ceiling and exactly what TrueDepth would improve.
- **inswapper cannot produce hair** — it is a 128 px face-crop model, structurally inner-face only.
- **Option (a) (`facepaint.py`) paints the face only.** Hair, ears and silhouette live outside the
  mesh. **That is the entire reason option (b) exists** — it is the only route to hair plus live
  control plus full framerate at once. Keep that as the success criterion.
- Performance bar to beat, on an M4 at 720p [measured]: 30 fps with face+body+hands+segmentation
  all on; option (a) adds 14 ms/face and still holds 30.2 fps.

---

## 9. House conventions

- Branch `felix` is canonical. `main` is README-only. Coordinate before force-pushing — there was a
  two-agent branch collision here already.
- Reference photos and generated textures are gitignored (`poc/packs/**/*.jpg|png`, `poc/textures/`)
  because they are real people's likenesses. **Do not commit a Chaewon asset.** The `.glb` is the
  same category — keep it local unless Felix says otherwise.
- Tests run camera-free and are expected to stay that way: `./.venv/bin/python -m pytest poc/tests -q`.
- Source-tag anything you research before relaying it: `[measured]` / `[documented]` / `[secondary]`
  / `[estimate]`. This is a standing practice after a subagent fabricated specifics on this project.
