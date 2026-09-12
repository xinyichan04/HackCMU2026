"""Live avatar: camera -> face landmarks -> stylized character -> preview / virtual camera / MP4.

    python poc/live.py --pack poc/packs/le-sserafim --character chaewon
    python poc/live.py --pack poc/packs/le-sserafim --character chaewon --virtual-cam
    python poc/live.py --pack poc/packs/le-sserafim --character chaewon --record out.mp4
    python poc/live.py --pack poc/packs/le-sserafim --source clip.mp4 --record out.mp4 --no-preview
    python poc/live.py --mode photo --character chaewon                 # photoreal swap (needs ref photos)
    python poc/live.py --mode photo --ref ~/Pictures/chaewon.jpg        # one-off reference for the current character

Hotkeys: n/p next/prev character, r start/stop recording, e toggle landmark smoothing (toon) / face
         enhancer (photo), d debug overlay, m toggle mirror, q quit.

Modes: toon  = MediaPipe landmarks -> stylized cartoon avatar (default, as shipped in PR #1).
       photo = InsightFace one-shot face swap from a reference photo per character (see swap.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avatar import AvatarRenderer, Pack  # noqa: E402
from tracker import MODEL_URL, FaceTracker  # noqa: E402

HERE = Path(__file__).resolve().parent
WATERMARKS = {"toon": "AI avatar", "photo": "AI face swap"}


def ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading face landmarker model to {path} ...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def open_source(args) -> tuple[cv2.VideoCapture, bool, float]:
    """Returns (capture, is_camera, fps)."""
    if args.source:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            sys.exit(f"cannot open {args.source}")
        return cap, False, cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        sys.exit("cannot open camera. On macOS grant Camera access to your terminal app "
                 "(System Settings > Privacy & Security > Camera) and run from Terminal, not an IDE sandbox.")
    return cap, True, 30.0


def draw_watermark(frame: np.ndarray, text: str = WATERMARKS["toon"]):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x, y = 14, h - 14
    cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, text: str, recording: bool):
    cv2.putText(frame, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 30, 28), 10, (0, 0, 255), -1, cv2.LINE_AA)


def draw_debug(frame: np.ndarray, faces):
    for f in faces:
        for x, y in f.pts[::3]:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
        x0, y0, x1, y1 = f.bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.putText(frame, f"yaw {f.yaw:+.2f} pitch {f.pitch:+.2f} blink {f.shapes.get('eyeBlinkLeft', 0):.2f} "
                    f"jaw {f.shapes.get('jawOpen', 0):.2f}", (x0, max(12, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)


class Recorder:
    def __init__(self):
        self.writer = None
        self.path = None

    @property
    def active(self):
        return self.writer is not None

    def start(self, path: str, size: tuple[int, int], fps: float):
        self.stop()
        self.path = path
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            self.writer = None
            print(f"could not open {path} for writing", flush=True)
        else:
            print(f"recording -> {path}", flush=True)

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            print(f"saved {self.path}", flush=True)
        self.writer = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["toon", "photo"], default="toon",
                    help="toon = cartoon avatar (default); photo = photoreal one-shot face swap")
    ap.add_argument("--pack", default=str(HERE / "packs" / "le-sserafim"))
    ap.add_argument("--character", default=None, help="character id in pack.json (default: first)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", default=None, help="video/image file instead of the camera")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-faces", type=int, default=1, help="faces to track; face i gets character (current+i)")
    ap.add_argument("--virtual-cam", action="store_true", help="publish to OBS Virtual Camera (pyvirtualcam)")
    ap.add_argument("--record", default=None, help="MP4 path; starts recording immediately")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-smoothing", action="store_true")
    ap.add_argument("--model", default=str(HERE / "models" / "face_landmarker.task"))
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (tests)")
    ap.add_argument("--save-frame", default=None, help="write the last rendered frame to this PNG (tests)")
    photo = ap.add_argument_group("photo mode")
    photo.add_argument("--ref", default=None, help="reference photo for the CURRENT character (overrides pack.json)")
    photo.add_argument("--swapper", default=None, help="path to inswapper_128(.fp16).onnx (default: auto-download)")
    photo.add_argument("--provider", choices=["auto", "coreml", "cuda", "cpu"], default="auto")
    photo.add_argument("--det-size", type=int, default=640, help="detector input size; 320 is faster")
    photo.add_argument("--detect-every", type=int, default=1, help="run the detector every N frames (2 = faster)")
    photo.add_argument("--enhance", action="store_true", help="GFPGAN on the swapped face (slow; not for live use)")
    args = ap.parse_args(argv)

    pack = Pack.load(args.pack)
    cur = pack.index(args.character) if args.character else 0
    tracker = renderer = swapper = None
    if args.mode == "photo":
        from swap import FaceSwapper, draw_debug_photo  # noqa: E402  (heavy imports only in photo mode)
        swapper = FaceSwapper(args.swapper, det_size=args.det_size, provider=args.provider, enhancer=args.enhance)
        if swapper.reference_for(pack.characters[cur], pack.path, args.ref) is None:
            sys.exit(f"no reference photo for {pack.characters[cur].id}: add \"ref\" in pack.json and drop the "
                     f"photo in {pack.path}, or pass --ref <photo>")
        last_faces: list = []
    else:
        tracker = FaceTracker(str(ensure_model(Path(args.model))), max_faces=args.max_faces,
                              smoothing=0.0 if args.no_smoothing else 0.35)
        renderer = AvatarRenderer()
    watermark = WATERMARKS[args.mode]
    cap, is_camera, src_fps = open_source(args)
    mirror = is_camera and not args.no_mirror
    debug = False
    rec = Recorder()
    vcam = None
    out = None
    t0 = time.monotonic()
    frame_i = 0
    fps_ema = 0.0
    last = time.perf_counter()
    win = "avatar (q quit, n/p character, r record, e smoothing, d debug, m mirror)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_camera:
                    print("camera read failed", flush=True)
                break
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                if is_camera:
                    frame = cv2.resize(frame, (args.width, args.height))
            if mirror:
                frame = cv2.flip(frame, 1)
            out = frame
            if args.mode == "photo":
                if frame_i % max(1, args.detect_every) == 0 or not last_faces:
                    last_faces = swapper.detect(frame, max_faces=args.max_faces)
                faces = last_faces
                for i, face in enumerate(faces):
                    ch = pack.characters[(cur + i) % len(pack.characters)]
                    ref = swapper.reference_for(ch, pack.path, args.ref if i == 0 else None)
                    if ref is None:
                        continue                      # character without a photo: leave that face alone
                    out = swapper.swap(out, face, ref)
                    if swapper.enhancer_on:
                        out = swapper.enhance(out, face)
            else:
                ts = int((time.monotonic() - t0) * 1000) if is_camera else int(frame_i * 1000 / src_fps)
                faces = tracker.track(frame, ts)
                for i, face in enumerate(faces):
                    out = renderer.render(out, face, pack.characters[(cur + i) % len(pack.characters)])
            draw_watermark(out, watermark)

            if args.record and not rec.active and frame_i == 0:
                rec.start(args.record, (out.shape[1], out.shape[0]), src_fps)
            rec.write(out)

            if args.virtual_cam:
                if vcam is None:
                    import pyvirtualcam
                    vcam = pyvirtualcam.Camera(width=out.shape[1], height=out.shape[0], fps=int(src_fps),
                                               fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False)
                    print(f"virtual camera: {vcam.device}", flush=True)
                vcam.send(out)

            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
            frame_i += 1

            if not args.no_preview:
                view = out.copy()
                if debug:
                    (draw_debug_photo if args.mode == "photo" else draw_debug)(view, faces)
                extra = ("  enhance" if swapper.enhancer_on else "") if args.mode == "photo" \
                    else ("  smooth" if tracker.smoothing else "")
                draw_hud(view, f"{args.mode} {pack.characters[cur].name}  {fps_ema:4.1f} fps  faces {len(faces)}"
                               f"{extra}", rec.active)
                cv2.imshow(win, view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q") or k == 27:
                    break
                elif k == ord("n"):
                    cur = (cur + 1) % len(pack.characters)
                elif k == ord("p"):
                    cur = (cur - 1) % len(pack.characters)
                elif k == ord("r"):
                    if rec.active:
                        rec.stop()
                    else:
                        rec.start(args.record or time.strftime("avatar-%Y%m%d-%H%M%S.mp4"),
                                  (out.shape[1], out.shape[0]), src_fps)
                elif k == ord("e"):
                    if args.mode == "photo":
                        swapper.disable_enhancer() if swapper.enhancer_on else swapper.enable_enhancer()
                    else:
                        tracker.smoothing = 0.0 if tracker.smoothing else 0.35
                elif k == ord("d"):
                    debug = not debug
                elif k == ord("m"):
                    mirror = not mirror
            if args.max_frames and frame_i >= args.max_frames:
                break
    finally:
        rec.stop()
        if vcam is not None:
            vcam.close()
        cap.release()
        if tracker is not None:
            tracker.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
        if args.save_frame and out is not None:
            cv2.imwrite(args.save_frame, out)
            print(f"saved {args.save_frame}", flush=True)
    print(f"{frame_i} frames, {fps_ema:.1f} fps (end-to-end incl. capture)", flush=True)


if __name__ == "__main__":
    main()
