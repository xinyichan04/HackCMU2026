"""Body and hand tracking to sit alongside the face tracker (MediaPipe Pose + Hands, CPU).

Pose gives 33 landmarks per person, multi-person, plus *world* landmarks (roughly metric 3D
inferred from a single RGB frame - positions are usable, absolute depth is not) and an optional
person segmentation mask. Hands give 21 landmarks each, which is where choreography actually
lives; face-only tracking discards it.

Models download once into poc/models/ (lite pose ~3 MB, hands ~7 MB).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
            "float16/1/pose_landmarker_lite.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
            "float16/1/hand_landmarker.task")

# Pose landmark indices worth naming (MediaPipe's 33-point topology).
POSE_NAMES = {0: "nose", 11: "shoulder_l", 12: "shoulder_r", 13: "elbow_l", 14: "elbow_r",
              15: "wrist_l", 16: "wrist_r", 23: "hip_l", 24: "hip_r", 25: "knee_l", 26: "knee_r",
              27: "ankle_l", 28: "ankle_r"}
# Skeleton edges: torso, arms, legs. Kept explicit so the overlay reads clearly at preview size.
POSE_EDGES = [(11, 12), (11, 23), (12, 24), (23, 24),
              (11, 13), (13, 15), (12, 14), (14, 16),
              (23, 25), (25, 27), (24, 26), (26, 28),
              (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
              (27, 29), (27, 31), (28, 30), (28, 32)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]


def ensure(path: Path, url: str) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {path.name} ...", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


@dataclass
class Body:
    pts: np.ndarray                      # (33, 2) pixel coordinates
    world: np.ndarray                    # (33, 3) approximate metres, hip-centred
    visible: np.ndarray                  # (33,) 0..1 per-landmark visibility
    mask: np.ndarray | None = None       # person segmentation, same HxW as the frame

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ok = self.pts[self.visible > 0.5]
        src = ok if len(ok) else self.pts
        x0, y0 = src.min(axis=0)
        x1, y1 = src.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)


@dataclass
class Hand:
    pts: np.ndarray                      # (21, 2) pixel coordinates
    label: str = "?"                     # "Left" / "Right" as seen by the camera
    score: float = 0.0
    world: np.ndarray = field(default_factory=lambda: np.zeros((21, 3), dtype=np.float32))


class BodyTracker:
    def __init__(self, max_bodies: int = 2, segmentation: bool = False):
        opts = vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(ensure(MODELS / "pose_landmarker_lite.task", POSE_URL))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max_bodies,
            output_segmentation_masks=segmentation,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1

    def track(self, frame_bgr: np.ndarray, ts_ms: int) -> list[Body]:
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(img, ts_ms)
        out: list[Body] = []
        for i, lms in enumerate(res.pose_landmarks):
            pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
            vis = np.array([getattr(p, "visibility", 1.0) for p in lms], dtype=np.float32)
            world = np.zeros((len(lms), 3), dtype=np.float32)
            if res.pose_world_landmarks and i < len(res.pose_world_landmarks):
                world = np.array([(p.x, p.y, p.z) for p in res.pose_world_landmarks[i]], dtype=np.float32)
            mask = None
            if res.segmentation_masks and i < len(res.segmentation_masks):
                mask = res.segmentation_masks[i].numpy_view()
            out.append(Body(pts=pts, world=world, visible=vis, mask=mask))
        return out

    def close(self):
        self._lm.close()


class HandTracker:
    def __init__(self, max_hands: int = 4):
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(ensure(MODELS / "hand_landmarker.task", HAND_URL))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def track(self, frame_bgr: np.ndarray, ts_ms: int) -> list[Hand]:
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(img, ts_ms)
        out: list[Hand] = []
        for i, lms in enumerate(res.hand_landmarks):
            pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
            label, score = "?", 0.0
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                label, score = res.handedness[i][0].category_name, res.handedness[i][0].score
            world = np.zeros((len(lms), 3), dtype=np.float32)
            if res.hand_world_landmarks and i < len(res.hand_world_landmarks):
                world = np.array([(p.x, p.y, p.z) for p in res.hand_world_landmarks[i]], dtype=np.float32)
            out.append(Hand(pts=pts, label=label, score=score, world=world))
        return out

    def close(self):
        self._lm.close()


def draw_body(frame: np.ndarray, body: Body, color, show_mask: bool = False, labels: bool = False):
    if show_mask and body.mask is not None:
        tint = np.zeros_like(frame)
        tint[:] = color
        a = np.clip(body.mask, 0, 1)[..., None] * 0.35
        np.copyto(frame, (frame * (1 - a) + tint * a).astype(np.uint8))
    for a, b in POSE_EDGES:
        if body.visible[a] > 0.5 and body.visible[b] > 0.5:
            cv2.line(frame, tuple(body.pts[a].astype(int)), tuple(body.pts[b].astype(int)), color, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(body.pts):
        if body.visible[i] > 0.5:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    if labels:
        for i, name in POSE_NAMES.items():
            if body.visible[i] > 0.5:
                cv2.putText(frame, name, (int(body.pts[i][0]) + 5, int(body.pts[i][1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_hand(frame: np.ndarray, hand: Hand, color):
    for a, b in HAND_EDGES:
        cv2.line(frame, tuple(hand.pts[a].astype(int)), tuple(hand.pts[b].astype(int)), color, 1, cv2.LINE_AA)
    for x, y in hand.pts:
        cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1, cv2.LINE_AA)
    wrist = hand.pts[0]
    cv2.putText(frame, hand.label, (int(wrist[0]) - 12, int(wrist[1]) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def body_record(body: Body, idx: int) -> dict:
    x0, y0, x1, y1 = body.bbox
    return {
        "id": idx,
        "bbox": [x0, y0, x1, y1],
        "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in body.pts],
        "world": [[round(float(v), 4) for v in p] for p in body.world],
        "visible": [round(float(v), 3) for v in body.visible],
        "has_mask": body.mask is not None,
    }


def hand_record(hand: Hand, idx: int) -> dict:
    return {
        "id": idx,
        "label": hand.label,
        "score": round(float(hand.score), 3),
        "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in hand.pts],
    }
