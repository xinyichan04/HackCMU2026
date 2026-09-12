# Source portraits

Drop character source photos here. This folder is gitignored (except this README) so
copyrighted celebrity photos are **not** committed.

## What makes a good source photo

The model auto-detects and crops the face, so you do **not** need to pre-crop — but the
photo must have:

- **One clear, front-facing face** looking roughly at the camera.
- **Neutral or gentle expression**, eyes open, mouth relaxed (this is the "rest pose"
  your expressions drive away from).
- **Even lighting**, no heavy shadows, no sunglasses/hands/hair covering the face.
- Reasonably high resolution (face at least ~256 px tall). Square-ish framing helps.

## Naming convention (used by run-filter.sh)

- `chaewon.jpg`  <- default character
- `sakura.jpg`, `yunjin.jpg`, ...  <- switch by passing the path to run-filter.sh

Provide 3–5 candidates per person if you can; we pick the one that reenacts best.
