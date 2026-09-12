# Be LE SSERAFIM — HackCMU 2026 (Multiplayer track)

Point a camera at you and your friends and everyone becomes a LE SSERAFIM member, live. Each person picks a
member, the app finds every face in the frame and replaces it with that person's pick in real time, and the
result can be recorded or sent to Zoom / FaceTime as a virtual camera. LE SSERAFIM is the launch character
pack; the engine is character-agnostic, so any group or set of characters is a data drop, not a code change.

## Why it's interesting

Most face filters are one person, one effect, on a phone. This is a shared scene: several people in one
frame, each mapped to a different character, identities staying put when people move, cross, or leave and
come back. The hard parts are multi-face tracking, per-person identity assignment, and doing it fast enough
on a laptop that it feels like a mirror rather than a video.

## How it works

The camera feed goes through a loop that runs many times a second:

1. **Find faces.** A face detector returns every face in the frame with landmarks or key points.
2. **Decide who is who.** Each detected face is matched to a player (by position now; by a face
   "fingerprint" enrolled at the start in the full multiplayer version), and each player has a character.
3. **Replace the face.** Three interchangeable renderers, each a different tradeoff:
   - **Toon avatar** — landmarks drive a hand-drawn cartoon head (hair, eyes, mouth, accessories) defined
     purely by data. Fully local, ~30 fps, clearly stylized. The default and the judged-demo path.
   - **Photoreal one-shot swap** — a pretrained swapper takes one reference photo per character and
     regenerates the face as that identity with the live expression. No training; one photo is the
     character. Local on Apple Silicon at 5-20 fps depending on chip.
   - **Reenactment** — the reverse: a still photo of the character is animated by the player's live
     expressions (LivePortrait on Apple MLX). Zero training, no NVIDIA GPU.
4. **Composite and output.** Watermarked frame → preview window, OBS virtual camera, or MP4.

No model is trained by us; everything runs from pretrained weights on the laptop. See `SPEC.md` for the
design, `RESEARCH-finetune.md` for why we did not train per-character models, and the per-mode docs below.

## Repository map

| Path | What |
|---|---|
| `poc/` | The live app: `live.py` (camera → track → render → output), `tracker.py`, `avatar.py` (toon), `swap.py` (photoreal), `packs/` (character data), `tests/` |
| `poc/README.md` | How to install and run each mode, hotkeys, adding a character |
| `SETUP.md`, `run-filter.sh`, `sources/` | Reenactment filter setup (FasterLivePortrait-MLX) |
| `SPEC.md` | Proof-of-concept spec and revision notes |
| `RESEARCH-finetune.md` | Survey of face-swap / fine-tuning options, costs and licences |

## Responsible use

Outputs are always watermarked. The characters are real people; the photoreal and reenactment modes exist
for the team's own experiments, and the public demo uses the stylized avatar. Reference photos are never
committed. Pretrained swap models carry non-commercial research licences.

## Team

CMU students, HackCMU 2026. Built in Python on Apple Silicon: MediaPipe, InsightFace, ONNX Runtime
(CoreML), OpenCV, MLX.
