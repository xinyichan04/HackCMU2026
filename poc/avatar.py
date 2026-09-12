"""Stylized 2D avatar renderer driven by MediaPipe face landmarks.

A character is pure data (see packs/*/pack.json): colors, hair style, bangs, eye shape, accessories.
Every shape is drawn in a face-local frame (origin between the eyes, unit = ear-to-ear width, y up), so
head scale, roll and yaw carry through automatically; blinks and mouth shapes come from the landmarks.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from tracker import Face

# ---- landmark index tables (MediaPipe Face Mesh) --------------------------------------------------
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
BROW_R = [70, 63, 105, 66, 107]          # subject's right brow (image left)
BROW_L = [300, 293, 334, 296, 336]
EYE_R = dict(outer=33, inner=133, top=159, bottom=145, iris=468, blink="eyeBlinkRight")
EYE_L = dict(outer=263, inner=362, top=386, bottom=374, iris=473, blink="eyeBlinkLeft")


def hex_bgr(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return (b, g, r)


def shade(bgr, f: float):
    return tuple(int(max(0, min(255, c * f))) for c in bgr)


@dataclass
class Character:
    id: str
    name: str
    skin: str = "#F7DCC9"
    hair: str = "#2B1B14"
    hair_style: str = "long"       # long | bob | short | ponytail
    bangs: str = "full"            # full | side | none
    eye: str = "#4A2C1A"
    eye_shape: float = 0.62        # eye height / width
    lip: str = "#E27A86"
    blush: str = "#F5A0A8"
    accessories: list[str] = field(default_factory=list)   # star_clip | ribbon | cat_ears | glasses | heart
    ref: str | None = None         # photo mode: reference photo path relative to the pack dir (gitignored)


@dataclass
class Pack:
    id: str
    name: str
    characters: list[Character]
    path: Path

    @staticmethod
    def load(path: str | Path) -> "Pack":
        path = Path(path)
        data = json.loads((path / "pack.json").read_text())
        chars = [Character(**c) for c in data["characters"]]
        if not chars:
            raise ValueError(f"{path}/pack.json has no characters")
        return Pack(id=data["id"], name=data.get("name", data["id"]), characters=chars, path=path)

    def index(self, char_id: str) -> int:
        for i, c in enumerate(self.characters):
            if c.id == char_id:
                return i
        raise KeyError(f"character {char_id!r} not in pack {self.id}: {[c.id for c in self.characters]}")


# ---- drawing canvas: color layer + alpha layer, composited once per face ---------------------------
class Canvas:
    def __init__(self, h: int, w: int):
        self.color = np.zeros((h, w, 3), np.uint8)
        self.alpha = np.zeros((h, w), np.uint8)

    def poly(self, pts: np.ndarray, color, alpha: int = 255):
        p = np.round(pts).astype(np.int32).reshape(1, -1, 2)
        cv2.fillPoly(self.color, p, color, lineType=cv2.LINE_AA)
        cv2.fillPoly(self.alpha, p, int(alpha), lineType=cv2.LINE_AA)

    def ellipse(self, center, axes, angle_deg, color, start=0, end=360, thickness=-1, alpha: int = 255):
        c = (int(round(center[0])), int(round(center[1])))
        a = (max(1, int(round(axes[0]))), max(1, int(round(axes[1]))))
        cv2.ellipse(self.color, c, a, angle_deg, start, end, color, thickness, cv2.LINE_AA)
        cv2.ellipse(self.alpha, c, a, angle_deg, start, end, int(alpha), thickness, cv2.LINE_AA)

    def circle(self, center, r, color, alpha: int = 255):
        c = (int(round(center[0])), int(round(center[1])))
        cv2.circle(self.color, c, max(1, int(round(r))), color, -1, cv2.LINE_AA)
        cv2.circle(self.alpha, c, max(1, int(round(r))), int(alpha), -1, cv2.LINE_AA)

    def stroke(self, pts: np.ndarray, color, thickness: float, closed=False, alpha: int = 255):
        p = np.round(pts).astype(np.int32).reshape(1, -1, 2)
        t = max(1, int(round(thickness)))
        cv2.polylines(self.color, p, closed, color, t, cv2.LINE_AA)
        cv2.polylines(self.alpha, p, closed, int(alpha), t, cv2.LINE_AA)

    def soft_circle(self, center, r, color, strength: float):
        """Gaussian-feathered blob (blush). Drawn into a temporary patch and max-merged into the alpha."""
        r = int(r)
        if r < 2:
            return
        cx, cy = int(center[0]), int(center[1])
        h, w = self.alpha.shape
        x0, y0, x1, y1 = max(0, cx - 2 * r), max(0, cy - 2 * r), min(w, cx + 2 * r), min(h, cy + 2 * r)
        if x1 <= x0 or y1 <= y0:
            return
        patch = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.circle(patch, (cx - x0, cy - y0), r, 1.0, -1)
        k = int(r) | 1
        patch = cv2.GaussianBlur(patch, (k, k), r / 2) * strength
        a = (patch * 255).astype(np.uint8)
        region = self.color[y0:y1, x0:x1]
        m = patch[..., None]
        region[:] = (region * (1 - m) + np.array(color, np.float32) * m).astype(np.uint8)
        self.alpha[y0:y1, x0:x1] = np.maximum(self.alpha[y0:y1, x0:x1], a)

    def composite(self, frame: np.ndarray, feather: int = 3) -> np.ndarray:
        ys, xs = np.nonzero(self.alpha)
        if len(ys) == 0:
            return frame
        pad = feather * 2
        y0, y1 = max(0, ys.min() - pad), min(frame.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(frame.shape[1], xs.max() + pad + 1)
        a = self.alpha[y0:y1, x0:x1].astype(np.float32) / 255.0
        if feather > 0:
            a = cv2.GaussianBlur(a, (feather * 2 + 1, feather * 2 + 1), 0)
        a = a[..., None]
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        col = self.color[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (roi * (1 - a) + col * a).astype(np.uint8)
        return frame


# ---- renderer -----------------------------------------------------------------------------------------
class AvatarRenderer:
    def __init__(self, line_scale: float = 1.0):
        self.line_scale = line_scale

    def render(self, frame: np.ndarray, face: Face, ch: Character) -> np.ndarray:
        h, w = frame.shape[:2]
        p = face.pts
        # face-local frame: origin between the eyes, u = image-right along the eye line, v = up, unit = face width
        origin = (p[33] + p[263]) / 2
        W = float(np.linalg.norm(p[454] - p[234]))
        if W < 8:
            return frame
        u = (p[263] - p[33])
        u = u / (np.linalg.norm(u) + 1e-6)
        v = np.array([u[1], -u[0]], np.float32)

        def L(x, y):  # local -> pixel
            return origin + (x * u + y * v) * W

        def LP(coords):
            return np.array([L(x, y) for x, y in coords], np.float32)

        skin, hair, lip = hex_bgr(ch.skin), hex_bgr(ch.hair), hex_bgr(ch.lip)
        eye_col, blush = hex_bgr(ch.eye), hex_bgr(ch.blush)
        ink = shade(hair, 0.45)
        canvas = Canvas(h, w)
        yaw, pitch = face.yaw, face.pitch
        chin_y = float(np.dot(p[152] - origin, v) / W)   # ~ -0.95

        # 1. hair, back layer (behind the face)
        bx = -yaw * 0.16
        self._hair_back(canvas, LP, ch.hair_style, hair, bx, chin_y)
        if "cat_ears" in ch.accessories:
            for sx in (-1, 1):
                ear = LP([(sx * 0.30 + bx, 0.72), (sx * 0.62 + bx, 1.18), (sx * 0.66 + bx, 0.62)])
                canvas.poly(ear, hair)
                canvas.poly(LP([(sx * 0.37 + bx, 0.74), (sx * 0.58 + bx, 1.05), (sx * 0.60 + bx, 0.68)]), blush)

        # 2. face: expanded landmark oval so no real skin shows at the edge
        oval = p[FACE_OVAL]
        cen = oval.mean(axis=0)
        oval = cen + (oval - cen) * 1.07
        oval = oval + v * W * 0.02
        canvas.poly(oval, skin)
        # ears when the head is fairly frontal
        for sx, vis in ((-1, 1 - max(0, -yaw) * 2.5), (1, 1 - max(0, yaw) * 2.5)):
            if vis > 0.2:
                canvas.ellipse(L(sx * 0.52, -0.12), (0.075 * W * vis, 0.12 * W), 0, skin)
        # neck under chin
        canvas.poly(LP([(-0.20, chin_y + 0.05), (0.20, chin_y + 0.05), (0.22, chin_y - 0.45), (-0.22, chin_y - 0.45)]), shade(skin, 0.92))

        # 3. blush
        for sx in (-1, 1):
            canvas.soft_circle(L(sx * 0.32, -0.20), 0.11 * W, blush, 0.55)

        # 4. mouth
        self._mouth(canvas, p, face.shapes, lip, ink, W)

        # 5. nose: small stroke under the tip
        tip = p[2]
        canvas.ellipse(tip + v * W * 0.01, (0.055 * W, 0.028 * W), math.degrees(face.roll), shade(skin, 0.72),
                       start=10, end=170, thickness=max(1, int(0.012 * W * self.line_scale)))

        # 6. eyes + brows
        for E in (EYE_R, EYE_L):
            self._eye(canvas, p, face.shapes, E, eye_col, ink, ch.eye_shape, W)
        for B in (BROW_R, BROW_L):
            canvas.stroke(p[B], ink, 0.045 * W * self.line_scale)

        # 7. hair, front layer (bangs), then accessories
        self._bangs(canvas, LP, ch.bangs, hair, -yaw * 0.05)
        self._accessories(canvas, L, LP, p, ch, W, hair, ink, bx)

        return canvas.composite(frame, feather=max(1, int(W * 0.006)))

    # -- pieces ------------------------------------------------------------------------------------------
    @staticmethod
    def _hair_back(c: Canvas, LP, style: str, hair, bx: float, chin_y: float):
        # cap runs over the top from image-right to image-left; the rest of each outline continues
        # down the left side and back up the right so the polygon never self-intersects.
        cap = _arc(bx, 0.20, 0.66, 0.72, 0, 180, 24)
        if style == "short":
            c.poly(LP(cap + [(-0.62 + bx, -0.10), (0.62 + bx, -0.10)]), hair)
        elif style == "bob":
            bottom = chin_y + 0.30
            c.poly(LP(cap + [(-0.66 + bx, bottom + 0.05), (-0.52 + bx, bottom), (0.0 + bx, bottom - 0.05),
                             (0.52 + bx, bottom), (0.66 + bx, bottom + 0.05)]), hair)
        elif style == "ponytail":
            c.poly(LP(cap + [(-0.64 + bx, -0.30), (0.64 + bx, -0.30)]), hair)
            side = -1 if bx <= 0 else 1                      # tail swings to the side away from the turn
            c.poly(LP(_arc(side * 0.62 + bx, 0.10, 0.20, 0.45, 0, 360, 20)), hair)
            c.poly(LP(_arc(side * 0.70 + bx, -0.55, 0.14, 0.42, 0, 360, 20)), shade(hair, 0.9))
        else:  # long
            bottom = chin_y - 0.55
            c.poly(LP(cap + [(-0.70 + bx, 0.10), (-0.66 + bx, bottom + 0.15), (-0.50 + bx, bottom),
                             (-0.20 + bx, bottom + 0.35), (0.20 + bx, bottom + 0.35),
                             (0.50 + bx, bottom), (0.66 + bx, bottom + 0.15), (0.70 + bx, 0.10)]), hair)
            # inner strands that frame the face
            for sx in (-1, 1):
                c.poly(LP([(sx * 0.48 + bx, 0.40), (sx * 0.62 + bx, 0.10), (sx * 0.56 + bx, chin_y + 0.1),
                           (sx * 0.40 + bx, chin_y + 0.05), (sx * 0.44 + bx, -0.2)]), shade(hair, 0.85))

    @staticmethod
    def _bangs(c: Canvas, LP, style: str, hair, bx: float):
        top = _arc(bx, 0.20, 0.66, 0.72, 0, 180, 24)         # image-right -> image-left over the top
        if style == "none":
            edge = [(-0.58 + bx, 0.44), (0.0 + bx, 0.50), (0.58 + bx, 0.44)]
        elif style == "side":
            # swept across the forehead, image-left to image-right
            edge = [(-0.62 + bx, 0.36), (-0.50 + bx, 0.26), (-0.34 + bx, 0.18), (-0.16 + bx, 0.30),
                    (0.02 + bx, 0.22), (0.20 + bx, 0.34), (0.36 + bx, 0.26), (0.50 + bx, 0.42), (0.62 + bx, 0.36)]
        else:  # full: soft scallops just above the brows
            n = 8
            edge = []
            for i in range(n + 1):
                x = -0.58 + 1.16 * i / n
                y = 0.36 if i in (0, n) else (0.21 if i % 2 else 0.24)
                edge.append((x + bx, y))
        c.poly(LP(top + edge), hair)

    @staticmethod
    def _eye(c: Canvas, p, shapes, E, eye_col, ink, eye_shape: float, W: float):
        outer, inner = p[E["outer"]], p[E["inner"]]
        center = (outer + inner) / 2
        ew = float(np.linalg.norm(inner - outer))
        ang = math.degrees(math.atan2(inner[1] - outer[1], inner[0] - outer[0]))
        blink = shapes.get(E["blink"], 0.0)
        # fall back to landmark opening when blendshapes are missing
        if not shapes:
            opening = float(np.linalg.norm(p[E["top"]] - p[E["bottom"]])) / max(ew, 1e-3)
            blink = float(np.clip(1 - opening / 0.28, 0, 1))
        open_ = 1 - blink
        ax_w = 0.62 * ew
        ax_h = ax_w * eye_shape * max(open_, 0.05)
        t_line = max(1.0, 0.09 * ew)
        if open_ < 0.22:
            # closed eye: a lash arc
            c.ellipse(center, (ax_w, ax_w * eye_shape * 0.35), ang, ink, start=180, end=360, thickness=int(t_line * 1.2))
            return
        c.ellipse(center, (ax_w, ax_h), ang, (255, 255, 255))
        # iris follows the tracked iris landmark, clamped inside the eye
        iris = p[E["iris"]]
        d = iris - center
        lim = ax_w * 0.45
        if np.linalg.norm(d) > lim:
            d = d / np.linalg.norm(d) * lim
        ic = center + d
        r_iris = 0.30 * ew
        c.circle(ic, r_iris, eye_col)
        c.circle(ic, r_iris * 0.5, (20, 15, 15))
        c.circle(ic + np.array([-r_iris * 0.35, -r_iris * 0.35]), r_iris * 0.22, (255, 255, 255))
        # outline: heavy on the top lash line, thin below
        c.ellipse(center, (ax_w, ax_h), ang, ink, start=180, end=360, thickness=int(t_line))
        c.ellipse(center, (ax_w, ax_h), ang, ink, start=0, end=180, thickness=max(1, int(t_line * 0.45)))

    @staticmethod
    def _mouth(c: Canvas, p, shapes, lip, ink, W: float):
        outer = p[LIPS_OUTER]
        inner = p[LIPS_INNER]
        cen = outer.mean(axis=0)
        outer = cen + (outer - cen) * 1.12          # slightly plumper toon lips
        c.poly(outer, lip)
        gap = float(np.linalg.norm(p[13] - p[14])) / W
        if gap > 0.03:
            c.poly(inner, (40, 20, 60))
            # teeth: upper 45% of the inner mouth
            y0, y1 = inner[:, 1].min(), inner[:, 1].max()
            top = inner[inner[:, 1] <= y0 + (y1 - y0) * 0.45]
            if len(top) >= 3:
                hull = cv2.convexHull(np.round(top).astype(np.int32)).reshape(-1, 2)
                c.poly(hull.astype(np.float32), (245, 245, 250))
        c.stroke(outer, shade(lip, 0.7), max(1, 0.012 * W), closed=True)

    @staticmethod
    def _accessories(c: Canvas, L, LP, p, ch: Character, W: float, hair, ink, bx: float):
        for a in ch.accessories:
            if a == "star_clip":
                c.poly(LP(_star(0.36 + bx * 0.3, 0.42, 0.075)), (60, 210, 255))
            elif a == "ribbon":
                x, y = -0.40 + bx, 0.62
                c.poly(LP([(x, y), (x - 0.18, y + 0.12), (x - 0.16, y - 0.10)]), (120, 90, 235))
                c.poly(LP([(x, y), (x + 0.18, y + 0.12), (x + 0.16, y - 0.10)]), (120, 90, 235))
                c.circle(L(x, y), 0.035 * W, (90, 60, 200))
            elif a == "glasses":
                frame_col = (70, 60, 60)
                t = max(2, int(0.02 * W))
                for E in (EYE_R, EYE_L):
                    cen = (p[E["outer"]] + p[E["inner"]]) / 2
                    c.ellipse(cen, (0.17 * W, 0.15 * W), 0, frame_col, thickness=t)
                c.stroke(np.array([p[133], p[362]]), frame_col, t)            # bridge
            elif a == "heart":
                c.poly(LP(_heart(-0.34 + bx * 0.3, -0.22, 0.06)), (110, 80, 240))
            elif a == "hoops":
                for sx in (-1, 1):
                    c.ellipse(L(sx * 0.53, -0.32), (0.05 * W, 0.075 * W), 0, (90, 200, 240), thickness=max(1, int(0.014 * W)))


def _arc(cx, cy, rx, ry, a0, a1, n):
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, n)]


def _star(cx, cy, r, n=5):
    pts = []
    for i in range(2 * n):
        a = math.pi / 2 + i * math.pi / n
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _heart(cx, cy, r):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 24):
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * r / 16, cy + y * r / 16))
    return pts
