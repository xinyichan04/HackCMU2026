# HackCMU 2026 — Proof of Concept Spec

> **Revision note (v0.3, 2026-09-11, implemented in `poc/`).**
> v0.2 below specified a photorealistic one-shot face swap (InsightFace `inswapper_128`) from a real
> member's photo onto the player, published to Zoom/FaceTime. That part is **not** implemented: it is a
> live, realistic impersonation of a real person who has not consented, and a watermark does not change
> what the artifact is. The POC instead implements the **stylized-avatar route** that §9 already listed
> as the fallback: MediaPipe Face Landmarker (478 landmarks + 52 blendshapes, CPU, ~10 ms) drives a
> cartoon character rendered from data. Everything else in this spec (pass criteria, CLI, hotkeys,
> pack format, virtual camera, recording, multi-face path in §8) is kept and implemented as written.
>
> Superseded: §3 pipeline/models, §3.1, §3.2, §6 numbers, and the model rows of §9.
> Character packs are still "one JSON entry per character, zero code"; `ref.jpg` is replaced by
> design parameters (hair style/colour, bangs, eye shape, lip/blush colour, accessories).
> Measured on the M4 MacBook, headless at 1280×720: see README.


**Project:** "Be LE SSERAFIM" (working title). Multiplayer track.
**POC goal (Felix, 2026-09-11):** hook up to a MacBook camera; software turns Felix into Chaewon, live, through the camera.
**Status:** Draft v0.2. Assumptions are marked `ASSUMPTION`.

## 1. What the POC proves

One person, one character, one laptop. Felix sits in front of the MacBook, the app shows him as Chaewon in real time, and the result can be recorded or fed to Zoom/OBS as a virtual camera. If this works, multiplayer is "run it per detected face with a lookup table," which is the full project (§8).

Pass criteria:

1. Live: 15+ fps at 720p on an Apple Silicon MacBook, no server.
2. Believable: head turns, blinks, mouth movement carry through; the face reads as Chaewon to someone who knows the group.
3. Character is data: swapping Chaewon for Sakura means changing one reference image, not code.

## 2. Non-goals

- Multi-person, identity assignment, remote players (full project, not POC).
- Training anything. No DeepFaceLab / DFM models (days of GPU per character, Windows-only tooling, archived project).
- Body, hair, outfit. Face region only.
- Mobile.

## 3. Approach: one-shot face swap, all local

Use InsightFace's one-shot swapper. It takes a single reference photo of the character and swaps identity onto any face, no training. Runs in ONNX Runtime with the CoreML execution provider on Apple Silicon.

```
MacBook camera (OpenCV / AVFoundation, 1280x720)
   │
   ▼
Face detection + 106-pt alignment + 512-d embedding     insightface buffalo_l (det_10g + w600k_r50)
   │           (embedding computed once for the Chaewon reference at startup)
   ▼
Swap: inswapper_128.onnx(source_embedding=Chaewon, target_face=Felix crop)
   │
   ▼
Paste back with the model's affine + soft mask feathering, color transfer
   │
   ▼ (optional, every 2nd frame or on demand)
Face enhancer: GFPGAN 1.4 / CodeFormer on the swapped crop only
   │
   ▼
Output: (a) preview window, (b) pyvirtualcam → "OBS Virtual Camera" so Zoom/FaceTime/OBS see Chaewon, (c) MP4 record
```

### 3.1 Two ways to get there, do both in order

**Step 0 — reference run with Deep-Live-Cam (≤ 1 hour).**
Deep-Live-Cam is an open-source wrapper around exactly this pipeline (insightface + inswapper + GFPGAN) with a webcam mode and a macOS Apple Silicon install path. Install it, load one Chaewon photo, click Live. That gives a working demo on day one and calibrates expectations (quality, fps, which reference photos work). It is the floor.

**Step 1 — own pipeline (the actual POC code, ~150 lines of Python).**
Re-implement the loop above ourselves so we own it: we need per-face routing, a character-pack format, and a recording path that Deep-Live-Cam doesn't give us cleanly. Same models, our loop.

`ASSUMPTION: Python 3.11, insightface 0.7.x, onnxruntime-silicon (CoreML EP), opencv-python, pyvirtualcam, gfpgan. inswapper_128.onnx is fetched from the community mirror since InsightFace pulled the official download.`

### 3.2 Reference image rules (matters more than code)

- Frontal, neutral or slight smile, even lighting, no heavy stage makeup, no glasses, ≥ 512 px face.
- Keep 3–5 candidates; the swap looks different with each. Pick empirically.
- Store as `packs/le-sserafim/chaewon/ref.jpg` plus `pack.json` so the multiplayer version reuses it.

## 4. Repo layout

```
HackCMU2026/
  SPEC.md
  README.md
  poc/
    live.py            # camera → detect → swap → enhance → preview / virtual cam / record
    swap.py            # FaceSwapper class: load models, embed reference, swap(frame, face)
    packs/
      le-sserafim/pack.json
      le-sserafim/chaewon/ref.jpg
    requirements.txt
    models/            # gitignored: buffalo_l, inswapper_128.onnx, GFPGANv1.4.pth
```

`pack.json`:

```json
{ "id": "le-sserafim", "characters": [
  { "id": "chaewon", "name": "Kim Chaewon", "ref": "chaewon/ref.jpg" } ] }
```

## 5. CLI for the POC

```
python poc/live.py --pack packs/le-sserafim --character chaewon
python poc/live.py --pack packs/le-sserafim --character chaewon --virtual-cam
python poc/live.py --pack packs/le-sserafim --character chaewon --record out.mp4
python poc/live.py --pack packs/le-sserafim --character chaewon --enhance   # GFPGAN on
```

Hotkeys in the preview window: `e` toggle enhancer, `n` next character in pack, `r` start/stop recording, `q` quit.

## 6. Performance budget (M1/M2 MacBook Air, CoreML EP)

| Stage | Expected |
|---|---|
| Detection (det_10g at 640) | 15–25 ms |
| Swap (inswapper_128) | 20–35 ms |
| Paste + color | 2 ms |
| GFPGAN (when on) | 80–150 ms → run every N frames or on a worker thread |
| **Without enhancer** | **~15–25 fps** |

If below 15 fps: detect every 2nd frame and track the box in between; run detection at 320 for a single face; use `--no-enhance` for the live demo and enhance only the recording offline.

## 7. Milestones

| When | Deliverable |
|---|---|
| Hour 1 | Deep-Live-Cam running with a Chaewon reference; pick the best reference photo |
| Hour 3 | `swap.py` swaps a single still image (Felix photo → Chaewon) |
| Hour 5 | `live.py` live preview at ≥ 15 fps, hotkeys |
| Hour 6 | Virtual camera works in Zoom / Photo Booth |
| Hour 7 | Recording to MP4; pack.json loader; `n` cycles members with more refs added |
| Hour 8 | POC demo clip (15 s) committed; go/no-go on multiplayer |

## 8. From POC to the multiplayer product

Everything below reuses the POC loop unchanged:

- **N faces:** detection already returns every face; loop the swap per face with the character chosen for that person.
- **Who is who:** at lobby time each friend enrolls (2 s in front of the camera → their own insightface embedding). At runtime each detected face is matched to the enrolled embeddings by cosine similarity, then swapped with that person's chosen member. Identity re-checked every ~1 s, tracked by box IoU in between. This is the core of the "multiplayer" claim.
- **Throughput:** 4 faces × 30 ms swap ≈ 8 fps on a MacBook. Options: swap at 512×288 tracking resolution, skip enhancer, or move the swap to a GPU box (RTX / rented A100) over WebSocket for the demo.
- **Web front end (stretch):** browser captures camera and streams frames to the Python server; server returns swapped frames. Keeps the laptop as the demo box and phones as extra cameras.

## 9. Risks

| Risk | Mitigation |
|---|---|
| inswapper model download/licensing (non-commercial research license) | Fine for a hackathon; say so in the README; do not ship it |
| Deepfake of a real idol | Watermark "AI face swap" on output; only use on consenting friends; stylized-avatar fallback (MediaPipe landmarks + VRM head) if judges object |
| CoreML EP fails to build | Fall back to CPU EP (~5–8 fps) or run the same code on a teammate's CUDA laptop |
| Reference photo gives an uncanny result | Keep 5 candidates, pick live; enable enhancer for stills |
| Virtual camera permissions on macOS | Install OBS once for its virtual camera driver; pyvirtualcam uses it |

## 10. Definition of done

- Felix on the MacBook camera, swapped to Chaewon, ≥ 15 fps, no server.
- Swap works in Zoom via virtual camera.
- One recorded clip in the repo.
- Changing `--character` to a second member with a new `ref.jpg` works with zero code changes.
