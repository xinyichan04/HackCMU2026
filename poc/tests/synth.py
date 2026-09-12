"""A drawn face that MediaPipe detects; lets the pipeline run without a camera or real photos."""
import cv2
import numpy as np


def synthetic_face(w=1280, h=720, cx=None, cy=None, scale=1.0, mouth_open=False):
    cx = w // 2 if cx is None else cx
    cy = h // 2 if cy is None else cy
    s = scale
    img = np.full((h, w, 3), (200, 190, 180), np.uint8)
    cv2.ellipse(img, (cx, cy), (int(150 * s), int(200 * s)), 0, 0, 360, (170, 200, 235), -1)
    cv2.ellipse(img, (cx, int(cy - 120 * s)), (int(175 * s), int(130 * s)), 0, 180, 360, (30, 25, 20), -1)
    for ex in (int(cx - 60 * s), int(cx + 60 * s)):
        ey = int(cy - 40 * s)
        cv2.ellipse(img, (ex, ey), (int(30 * s), int(16 * s)), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (ex, ey), int(12 * s), (60, 40, 20), -1)
        cv2.circle(img, (ex, ey), int(5 * s), (0, 0, 0), -1)
        cv2.ellipse(img, (ex, int(cy - 75 * s)), (int(40 * s), int(8 * s)), 0, 180, 360, (40, 30, 20), -1)
    cv2.ellipse(img, (cx, int(cy + 20 * s)), (int(14 * s), int(30 * s)), 0, 0, 360, (140, 170, 215), -1)
    if mouth_open:
        cv2.ellipse(img, (cx, int(cy + 95 * s)), (int(45 * s), int(28 * s)), 0, 0, 360, (40, 30, 60), -1)
    else:
        cv2.ellipse(img, (cx, int(cy + 90 * s)), (int(45 * s), int(18 * s)), 0, 0, 180, (90, 90, 200), -1)
    return img
