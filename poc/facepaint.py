"""Paint a face texture onto the tracked mesh - the 2D path to "someone else's face", no renderer.

Two directions, same 898 triangles:

    # a photo you have  ->  a canonical-UV texture (this is the step nobody has an asset for)
    python poc/facepaint.py --from-photo chaewon.jpg --out poc/packs/chaewon.png

    # that texture  ->  live on your face
    python poc/track.py --texture poc/packs/chaewon.png

The texture layout is MediaPipe's canonical_face_model UV map (poc/assets/, fetched from the
MediaPipe repo - the pip wheel does not ship it). Because --from-photo *builds* that layout by
inverse-warping a tracked photo, any frontal face photo works as input; you never have to
hand-author UVs.

Why this and not a 3D model: the tracker already gives 468 points per frame in the same topology
as the canonical model, so a texture can be warped triangle-by-triangle straight onto the face at
camera speed. It is photoreal because it is a photograph. What it cannot do is hair, ears or the
silhouette - those live outside the mesh, and that is the .glb/renderer path, not this one.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CANONICAL_OBJ = HERE / "assets" / "canonical_face_model.obj"
N_CANONICAL = 468          # the tracker emits 478; the last 10 are irises, not in the mesh

# Face-oval ring: used to trim the texture to the face and to feather the edge.
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def load_canonical(obj_path: Path = CANONICAL_OBJ) -> tuple[np.ndarray, np.ndarray]:
    """Parse the canonical face model -> (tris (898,3) landmark indices, uv (468,2) in 0..1).

    The OBJ indexes positions and texture coords separately (`f v/vt ...`) even though there is
    one of each per vertex, so the uv table is rebuilt keyed by *vertex* index - that is the index
    MediaPipe's landmark array uses.
    """
    if not obj_path.exists():
        raise FileNotFoundError(
            f"{obj_path} not found. Fetch it once:\n"
            "  curl -sSfL -o poc/assets/canonical_face_model.obj \\\n"
            "    https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/"
            "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
        )
    vt: list[tuple[float, float]] = []
    tris: list[tuple[int, int, int]] = []
    pairs: list[tuple[int, int]] = []
    for line in obj_path.read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "vt":
            vt.append((float(f[1]), float(f[2])))
        elif f[0] == "f":
            corners = []
            for tok in f[1:4]:
                v_s, _, t_s = tok.partition("/")
                v_i = int(v_s) - 1
                t_i = int(t_s.split("/")[0]) - 1 if t_s else v_i
                corners.append(v_i)
                pairs.append((v_i, t_i))
            tris.append(tuple(corners))

    uv = np.zeros((N_CANONICAL, 2), dtype=np.float32)
    seen = np.zeros(N_CANONICAL, dtype=bool)
    for v_i, t_i in pairs:
        if v_i < N_CANONICAL:
            uv[v_i] = vt[t_i]
            seen[v_i] = True
    if not seen.all():
        missing = int((~seen).sum())
        print(f"warning: {missing} canonical vertices had no UV; they will sample texture (0,0)")
    return np.array(tris, dtype=np.int32), uv


def uv_to_pixels(uv: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """UV (origin bottom-left, as OBJ stores it) -> image pixels (origin top-left)."""
    w, h = size
    px = np.empty_like(uv)
    px[:, 0] = uv[:, 0] * (w - 1)
    px[:, 1] = (1.0 - uv[:, 1]) * (h - 1)
    return px


def _warp_triangles(src: np.ndarray, dst: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
                    tris: np.ndarray, coverage: np.ndarray | None = None) -> None:
    """Affine-warp every triangle from src into dst, in place.

    Each triangle is warped inside its own bounding box rather than over the whole image - that is
    what keeps 898 triangles inside a frame budget.
    """
    H, W = dst.shape[:2]
    sh, sw = src.shape[:2]
    for tri in tris:
        d = dst_pts[tri]
        s = src_pts[tri]
        dx, dy, dw, dh = cv2.boundingRect(d.astype(np.float32))
        # clip the destination rect to the image; the affine stays valid because the local
        # coordinates are measured from the clipped origin
        x0, y0 = max(dx, 0), max(dy, 0)
        x1, y1 = min(dx + dw, W), min(dy + dh, H)
        if x1 <= x0 or y1 <= y0:
            continue
        sx, sy, sw_r, sh_r = cv2.boundingRect(s.astype(np.float32))
        sx0, sy0 = max(sx, 0), max(sy, 0)
        sx1, sy1 = min(sx + sw_r, sw), min(sy + sh_r, sh)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        s_local = (s - (sx0, sy0)).astype(np.float32)
        d_local = (d - (x0, y0)).astype(np.float32)
        M = cv2.getAffineTransform(s_local, d_local)
        crop = src[sy0:sy1, sx0:sx1]
        warped = cv2.warpAffine(crop, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(m, d_local.astype(np.int32), 255, cv2.LINE_8)
        # copyTo with a mask stays in OpenCV's C++ loop; boolean fancy-indexing here costs ~3x
        # as much, and it runs 898 times per face per frame.
        cv2.copyTo(warped, m, dst[y0:y1, x0:x1])
        if coverage is not None:
            cv2.copyTo(m, m, coverage[y0:y1, x0:x1])


def photo_to_texture(photo: np.ndarray, landmarks: np.ndarray, tris: np.ndarray, uv: np.ndarray,
                     size: int = 1024) -> np.ndarray:
    """Inverse-warp a tracked photo into the canonical UV layout -> a usable face texture."""
    tex = np.zeros((size, size, 3), dtype=np.uint8)
    dst_pts = uv_to_pixels(uv, (size, size))
    _warp_triangles(photo, tex, landmarks[:N_CANONICAL], dst_pts, tris)
    return tex


class FacePainter:
    """Warps one texture onto tracked faces. Reusable across frames; holds no per-frame state."""

    def __init__(self, texture_path: str | Path, feather: int = 9, alpha: float = 1.0,
                 color_match: bool = True, trim_forehead: float = 0.0):
        tex = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
        if tex is None:
            raise FileNotFoundError(f"cannot read texture {texture_path}")
        self.texture = tex
        self.tris, self.uv = load_canonical()
        self.tex_pts = uv_to_pixels(self.uv, (tex.shape[1], tex.shape[0]))
        self.feather = max(0, int(feather))
        self.alpha = float(alpha)
        self.color_match = color_match
        # A reference photo with a fringe bakes hair into the texture's forehead, which then
        # ghosts onto the wearer. Cropping the top of the painted area is the cheap fix.
        self.trim_forehead = float(np.clip(trim_forehead, 0.0, 0.9))

    def render(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Composite the texture onto one tracked face. Returns the frame (modified in place).

        Everything happens inside the face's bounding box. At a webcam distance the face is a few
        percent of the frame, so the masking, blur and colour maths cost a fraction of what they
        would full-frame - that is the difference between ~35 ms and a frame budget that fits 30 fps.
        """
        if landmarks is None or len(landmarks) < N_CANONICAL:
            return frame
        H, W = frame.shape[:2]
        pts = landmarks[:N_CANONICAL].astype(np.float32)
        bx, by, bw, bh = cv2.boundingRect(pts)
        pad = self.feather * 2 + 4
        x0, y0 = max(bx - pad, 0), max(by - pad, 0)
        x1, y1 = min(bx + bw + pad, W), min(by + bh + pad, H)
        if x1 <= x0 or y1 <= y0:
            return frame
        roi = frame[y0:y1, x0:x1]
        local = pts - (x0, y0)

        canvas = np.zeros_like(roi)
        coverage = np.zeros(roi.shape[:2], dtype=np.uint8)
        _warp_triangles(self.texture, canvas, self.tex_pts, local, self.tris, coverage)

        # Trim to the face oval so stray triangles outside the silhouette cannot bleed, then
        # feather inward: eroding before the blur keeps the soft edge inside the painted area.
        clip = np.zeros_like(coverage)
        cv2.fillConvexPoly(clip, cv2.convexHull(local[OVAL].astype(np.int32)), 255)
        if self.trim_forehead:
            cut = int(local[:, 1].min() + self.trim_forehead * np.ptp(local[:, 1]))
            clip[:max(cut, 0)] = 0          # the feather below softens this straight edge
        mask = cv2.bitwise_and(coverage, clip)
        if self.feather:
            k = self.feather * 2 + 1
            mask = cv2.erode(mask, np.ones((self.feather, self.feather), np.uint8))
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        if self.color_match:
            canvas = _match_color(canvas, roi, mask)

        a = (mask.astype(np.float32) / 255.0 * self.alpha)[:, :, None]
        np.copyto(roi, (roi.astype(np.float32) * (1 - a) + canvas.astype(np.float32) * a)
                  .astype(np.uint8))
        return frame


def _match_color(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift the warped face's mean/std in LAB toward the frame it is landing on.

    Cheap lighting match - without it a studio-lit texture sits on a dim webcam face like a sticker.
    """
    sel = mask > 32
    if sel.sum() < 64:
        return src
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    for c in range(3):
        sm, ss = s[:, :, c][sel].mean(), s[:, :, c][sel].std() + 1e-6
        rm, rs = r[:, :, c][sel].mean(), r[:, :, c][sel].std() + 1e-6
        # clamp the std ratio: an almost-flat channel would otherwise blow up into banding
        out[:, :, c] = (s[:, :, c] - sm) * float(np.clip(rs / ss, 0.5, 2.0)) + rm
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _track_one(image: np.ndarray, model: str):
    """Landmarks for a still image. Imported lazily so --uv-template needs no model download."""
    sys.path.insert(0, str(HERE))
    from live import ensure_model
    from tracker import FaceTracker
    tr = FaceTracker(str(ensure_model(Path(model))), max_faces=1, smoothing=0.0)
    try:
        faces = tr.track(image, int(time.monotonic() * 1000))
    finally:
        tr.close()
    return faces[0].pts if faces else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-photo", metavar="IMG", help="build a canonical-UV texture from a face photo")
    ap.add_argument("--out", default=None, help="where to write the texture PNG")
    ap.add_argument("--size", type=int, default=1024, help="texture resolution (default 1024)")
    ap.add_argument("--apply-to", metavar="IMG", help="test: paint a texture onto this photo")
    ap.add_argument("--texture", default=None, help="texture to use with --apply-to")
    ap.add_argument("--trim-forehead", type=float, default=0.0, metavar="0..0.9",
                    help="crop this fraction off the top of the painted face - use it when the "
                         "reference photo had a fringe and hair is ghosting onto the forehead")
    ap.add_argument("--uv-template", metavar="PNG", help="write the UV wireframe, to paint over by hand")
    ap.add_argument("--model", default=str(HERE / "models" / "face_landmarker.task"))
    args = ap.parse_args(argv)

    tris, uv = load_canonical()

    if args.uv_template:
        size = args.size
        img = np.full((size, size, 3), 18, dtype=np.uint8)
        pts = uv_to_pixels(uv, (size, size)).astype(np.int32)
        for tri in tris:
            cv2.polylines(img, [pts[tri].reshape(-1, 1, 2)], True, (70, 120, 90), 1, cv2.LINE_AA)
        cv2.imwrite(args.uv_template, img)
        print(f"wrote {args.uv_template} ({size}x{size}, {len(tris)} triangles) - "
              f"paint inside this layout and the result is a valid texture")
        return 0

    if args.from_photo:
        photo = cv2.imread(args.from_photo, cv2.IMREAD_COLOR)
        if photo is None:
            print(f"cannot read {args.from_photo}")
            return 1
        lm = _track_one(photo, args.model)
        if lm is None:
            print(f"no face found in {args.from_photo}. Needs a roughly frontal, unobstructed face.")
            return 1
        tex = photo_to_texture(photo, lm, tris, uv, size=args.size)
        out = args.out or str(Path(args.from_photo).with_suffix("")) + "-texture.png"
        cv2.imwrite(out, tex)
        filled = (cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY) > 0).mean()
        print(f"wrote {out} ({args.size}x{args.size}, {filled:.0%} of the UV square filled)")
        print(f"try it live:  python poc/track.py --texture {out}")
        return 0

    if args.apply_to:
        if not args.texture:
            print("--apply-to needs --texture")
            return 1
        target = cv2.imread(args.apply_to, cv2.IMREAD_COLOR)
        if target is None:
            print(f"cannot read {args.apply_to}")
            return 1
        lm = _track_one(target, args.model)
        if lm is None:
            print(f"no face found in {args.apply_to}")
            return 1
        painter = FacePainter(args.texture, trim_forehead=args.trim_forehead)
        t0 = time.perf_counter()
        out_img = painter.render(target, lm)
        ms = (time.perf_counter() - t0) * 1000
        out = args.out or "facepaint-test.png"
        cv2.imwrite(out, out_img)
        print(f"wrote {out}  ({ms:.1f} ms for {len(tris)} triangles)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
