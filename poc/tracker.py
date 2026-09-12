"""Face tracking: MediaPipe Face Landmarker (478 landmarks + 52 blendshapes), CPU, ~10 ms/frame on Apple Silicon."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


@dataclass
class Face:
    pts: np.ndarray            # (478, 2) float32 pixel coordinates
    shapes: dict[str, float]   # blendshape name -> 0..1 (eyeBlinkLeft, jawOpen, mouthSmileLeft, ...)
    yaw: float                 # -1 (nose toward image left) .. +1 (image right)
    pitch: float               # -1 (looking down) .. +1 (looking up)
    roll: float                # radians, image-space angle of the eye line

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x0, y0 = self.pts.min(axis=0)
        x1, y1 = self.pts.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)


class FaceTracker:
    def __init__(self, model_path: str, max_faces: int = 1, smoothing: float = 0.35):
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self.smoothing = smoothing      # EMA weight of the previous frame; 0 = off
        self._prev: list[np.ndarray] = []
        self._last_ts = -1

    def track(self, frame_bgr: np.ndarray, ts_ms: int) -> list[Face]:
        # VIDEO mode requires strictly increasing timestamps.
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(img, ts_ms)

        faces: list[Face] = []
        if len(res.face_landmarks) != len(self._prev):
            self._prev = []
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
            if self.smoothing > 0 and i < len(self._prev):
                pts = self.smoothing * self._prev[i] + (1 - self.smoothing) * pts
            shapes = {c.category_name: c.score for c in res.face_blendshapes[i]} if res.face_blendshapes else {}
            faces.append(Face(pts=pts, shapes=shapes, **_pose(pts)))
        self._prev = [f.pts for f in faces]
        return faces

    def close(self):
        self._lm.close()


def _pose(p: np.ndarray) -> dict:
    """Geometric head pose from landmarks (no transform matrix needed)."""
    eye_r, eye_l = p[33], p[263]          # subject's right eye (image left), subject's left eye (image right)
    roll = math.atan2(eye_l[1] - eye_r[1], eye_l[0] - eye_r[0])
    # un-roll so yaw/pitch are measured in the face's own frame
    c, s = math.cos(-roll), math.sin(-roll)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    q = (p - p[1]) @ R.T                  # nose tip at origin
    left, right = q[234][0], q[454][0]    # face oval extremes
    yaw = 0.0 if right == left else float(np.clip(-(left + right) / (right - left) * 2.2, -1, 1))
    eyes_y = (q[33][1] + q[263][1]) / 2
    chin_y = q[152][1]
    t = 0.0 if chin_y == eyes_y else -eyes_y / (chin_y - eyes_y)   # nose position between eyes (0) and chin (1)
    pitch = float(np.clip((0.42 - t) * 4.0, -1, 1))
    return {"yaw": yaw, "pitch": pitch, "roll": roll}
