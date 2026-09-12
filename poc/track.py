"""Face-tracking probe: show what the tracker sees, and hand the data to whatever comes next.

    python poc/track.py --list-cameras          # which cameras exist (built-in, Continuity, USB)
    python poc/track.py                         # live window, default camera
    python poc/track.py --camera 1              # iPhone via Continuity Camera
    python poc/track.py --udp-out 127.0.0.1:9001    # stream tracking JSON to another process
    python poc/track.py --jsonl-out track.jsonl     # record tracking data to a file
    python poc/track.py --source udp:9000       # take ARKit data from an iPhone instead of a camera

Hotkeys: m mesh, l landmarks, b blendshape bars, p pose axes, i ids, q quit, s save a PNG.

Two sources, one output format:
  camera  MediaPipe Face Landmarker - multi-face, 478 landmarks, 52 blendshapes inferred from RGB.
  udp     An iPhone running ARKit (Live Link Face, or your own Xcode app) sending the 52 ARKit
          blendshapes. TrueDepth-grade, single face. ARKit is iOS-only - there is no macOS
          face-tracking API - so the phone does the tracking and the Mac consumes it.
Both emit the same record, so anything built on this keeps working when the source changes.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live import ensure_model  # noqa: E402
from tracker import MODEL_URL, FaceTracker  # noqa: E402

HERE = Path(__file__).resolve().parent

# MediaPipe face-mesh contours, as index pairs - enough to read the geometry without the full
# 2000-edge tesselation, which turns into a grey smear at preview size.
CONTOURS = {
    "oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "lips_outer": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    "lips_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
    "eye_r": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "eye_l": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    "brow_r": [70, 63, 105, 66, 107],
    "brow_l": [300, 293, 334, 296, 336],
    "nose": [168, 6, 197, 195, 5, 4, 1],
}
PALETTE = [(80, 220, 255), (120, 255, 140), (255, 160, 90), (230, 120, 255), (255, 230, 100)]

# The blendshapes worth watching live; the full 52 still go out over UDP/JSONL.
WATCH = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight",
         "browInnerUp", "browDownLeft", "browDownRight", "eyeLookOutLeft", "cheekPuff"]


def list_cameras(limit: int = 8) -> list[dict]:
    """Probe camera indices. On macOS a Continuity Camera iPhone shows up as an ordinary index."""
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append({"index": i, "width": w, "height": h,
                              "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 1)})
        cap.release()
    return found


class UdpFaceSource:
    """Receive ARKit blendshapes from an iPhone as JSON datagrams.

    Expected payload (what your Xcode app should send; keys beyond these are ignored):
        {"blendshapes": {"jawOpen": 0.31, ...}, "head": {"yaw": 0.1, "pitch": -0.05, "roll": 0.02}}
    """

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.5)
        self.port = port
        self.packets = 0

    def read(self) -> dict | None:
        try:
            data, _ = self.sock.recvfrom(65535)
        except socket.timeout:
            return None
        try:
            msg = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        self.packets += 1
        head = msg.get("head") or {}
        return {"id": 0, "source": "arkit-udp",
                "blendshapes": msg.get("blendshapes") or {},
                "yaw": float(head.get("yaw", 0.0)), "pitch": float(head.get("pitch", 0.0)),
                "roll": float(head.get("roll", 0.0)), "bbox": None, "landmarks": None}

    def close(self):
        self.sock.close()


def record(face, idx: int) -> dict:
    """One face -> the wire format. Landmarks are rounded to keep packets small."""
    x0, y0, x1, y1 = face.bbox
    return {
        "id": idx,
        "source": "mediapipe",
        "bbox": [x0, y0, x1, y1],
        "yaw": round(face.yaw, 4), "pitch": round(face.pitch, 4), "roll": round(face.roll, 4),
        "blendshapes": {k: round(v, 4) for k, v in face.shapes.items()},
        "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in face.pts],
    }


def draw_face(frame: np.ndarray, face, idx: int, show: dict):
    color = PALETTE[idx % len(PALETTE)]
    p = face.pts
    if show["mesh"]:
        for name, ring in CONTOURS.items():
            closed = not name.startswith(("brow", "nose"))
            pts = p[ring].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], closed, color, 1, cv2.LINE_AA)
    if show["landmarks"]:
        for x, y in p:
            cv2.circle(frame, (int(x), int(y)), 1, color, -1, cv2.LINE_AA)
    x0, y0, x1, y1 = face.bbox
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
    if show["ids"]:
        cv2.putText(frame, f"face {idx}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    if show["pose"]:
        draw_pose(frame, face, color)


def draw_pose(frame: np.ndarray, face, color):
    """Yaw/pitch/roll as a small axis cross at the nose - the quickest way to see tracking jitter."""
    nose = face.pts[1]
    n = int(max(30.0, (face.bbox[2] - face.bbox[0]) * 0.35))
    c, s = math.cos(face.roll), math.sin(face.roll)
    ex, ey = c * n, s * n                      # face-local x axis, rolled into image space
    ux, uy = -s * n, c * n                     # face-local y axis
    o = (int(nose[0]), int(nose[1]))
    cv2.arrowedLine(frame, o, (int(nose[0] + ex * (1 + face.yaw)), int(nose[1] + ey)), color, 2,
                    cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(frame, o, (int(nose[0] + ux), int(nose[1] + uy * (1 - face.pitch))),
                    (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(frame, f"y{face.yaw:+.2f} p{face.pitch:+.2f} r{math.degrees(face.roll):+.0f}",
                (o[0] - 50, o[1] + n + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_shape_bars(frame: np.ndarray, shapes: dict[str, float], color):
    """Live bars for a few blendshapes: proof the 52 ARKit-equivalent coefficients are really there."""
    if not shapes:
        return
    x, y, w = 12, 60, 150
    cv2.rectangle(frame, (x - 8, y - 22), (x + w + 118, y + 18 * len(WATCH)), (0, 0, 0), -1)
    cv2.putText(frame, f"blendshapes ({len(shapes)} total)", (x - 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    for i, name in enumerate(WATCH):
        v = float(shapes.get(name, 0.0))
        yy = y + i * 18
        cv2.rectangle(frame, (x, yy), (x + w, yy + 11), (55, 55, 55), -1)
        cv2.rectangle(frame, (x, yy), (x + int(w * min(v, 1.0)), yy + 11), color, -1)
        cv2.putText(frame, f"{name} {v:.2f}", (x + w + 8, yy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220, 220, 220), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, text: str):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-cameras", action="store_true", help="probe camera indices and exit")
    ap.add_argument("--source", default="camera", help="'camera' (default) or 'udp:<port>' for ARKit input")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-faces", type=int, default=4, help="how many faces to track at once")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-smoothing", action="store_true", help="raw landmarks, no temporal EMA")
    ap.add_argument("--udp-out", default=None, metavar="HOST:PORT", help="stream tracking JSON per frame")
    ap.add_argument("--jsonl-out", default=None, help="append one JSON record per frame to this file")
    ap.add_argument("--no-landmarks-out", action="store_true", help="omit landmark arrays from output")
    ap.add_argument("--model", default=str(HERE / "models" / "face_landmarker.task"))
    ap.add_argument("--no-preview", action="store_true", help="headless; use with --max-frames")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (tests)")
    ap.add_argument("--save-frame", default=None, help="write the last drawn frame to this PNG")
    args = ap.parse_args(argv)

    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("no cameras found (on macOS, grant Camera access to your terminal app)")
            return 1
        print(f"{len(cams)} camera(s):")
        for c in cams:
            print(f"  --camera {c['index']}   {c['width']}x{c['height']} @ {c['fps'] or '?'}fps")
        print("\nA Continuity Camera iPhone appears here as an ordinary index once it is awake"
              "\nand near the Mac; it is usually the highest-resolution entry.")
        return 0

    udp_in = None
    cap = None
    tracker = None
    if args.source.startswith("udp:"):
        udp_in = UdpFaceSource(int(args.source.split(":", 1)[1]))
        print(f"listening for ARKit blendshape packets on UDP {udp_in.port}")
        print("expected JSON: {\"blendshapes\": {...52 ARKit names...}, \"head\": {\"yaw\":..,\"pitch\":..,\"roll\":..}}")
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print(f"cannot open camera {args.camera}. Try --list-cameras. On macOS the terminal app "
                  f"needs Camera permission (System Settings > Privacy & Security > Camera).")
            return 1
        tracker = FaceTracker(str(ensure_model(Path(args.model))), max_faces=args.max_faces,
                              smoothing=0.0 if args.no_smoothing else 0.35)

    sink = None
    if args.udp_out:
        host, port = args.udp_out.rsplit(":", 1)
        sink = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))
        print(f"streaming tracking JSON to udp://{host}:{port}")
    jsonl = open(args.jsonl_out, "a") if args.jsonl_out else None

    show = {"mesh": True, "landmarks": False, "shapes": True, "pose": True, "ids": True}
    fps_ema, t_prev, frame_i, out = 0.0, time.monotonic(), 0, None
    print("tracking. keys: m mesh, l landmarks, b bars, p pose, i ids, s save, q quit")
    try:
        while True:
            faces_out: list[dict] = []
            if udp_in:
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                rec = udp_in.read()
                if rec:
                    faces_out.append(rec)
                    draw_shape_bars(frame, rec["blendshapes"], PALETTE[0])
                    cv2.putText(frame, f"ARKit over UDP - {udp_in.packets} packets",
                                (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (120, 255, 140), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"waiting for packets on UDP {udp_in.port} ...",
                                (12, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (150, 150, 150), 1, cv2.LINE_AA)
                faces = []
            else:
                ok, frame = cap.read()
                if not ok:
                    print("camera read failed")
                    break
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                faces = tracker.track(frame, int(time.monotonic() * 1000))
                for i, f in enumerate(faces):
                    draw_face(frame, f, i, show)
                    faces_out.append(record(f, i))
                if show["shapes"] and faces:
                    draw_shape_bars(frame, faces[0].shapes, PALETTE[0])

            now = time.monotonic()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
            src = f"udp:{udp_in.port}" if udp_in else f"cam {args.camera}"
            draw_hud(frame, f"{src}  |  faces {len(faces_out)}/{args.max_faces}  |  {fps_ema:4.1f} fps"
                            f"  |  {'52 blendshapes' if faces_out and faces_out[0]['blendshapes'] else 'no blendshapes'}")

            if faces_out and (sink or jsonl):
                payload = faces_out
                if args.no_landmarks_out:
                    payload = [{k: v for k, v in f.items() if k != "landmarks"} for f in faces_out]
                line = json.dumps({"t": round(now, 3), "frame": frame_i, "faces": payload})
                if sink:
                    sock, addr = sink
                    try:
                        sock.sendto(line.encode("utf-8"), addr)
                    except OSError as e:
                        print(f"udp send failed: {e}")
                if jsonl:
                    jsonl.write(line + "\n")

            out = frame
            if not args.no_preview:
                cv2.imshow("track.py - face tracking probe (q quits)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("m"):
                    show["mesh"] = not show["mesh"]
                if k == ord("l"):
                    show["landmarks"] = not show["landmarks"]
                if k == ord("b"):
                    show["shapes"] = not show["shapes"]
                if k == ord("p"):
                    show["pose"] = not show["pose"]
                if k == ord("i"):
                    show["ids"] = not show["ids"]
                if k == ord("s"):
                    name = f"track-{int(time.time())}.png"
                    cv2.imwrite(name, frame)
                    print(f"saved {name}")
            frame_i += 1
            if args.max_frames and frame_i >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if args.save_frame and out is not None:
            cv2.imwrite(args.save_frame, out)
            print(f"wrote {args.save_frame}")
        if cap:
            cap.release()
        if tracker:
            tracker.close()
        if udp_in:
            udp_in.close()
        if jsonl:
            jsonl.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
    print(f"{frame_i} frames, {fps_ema:.1f} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
