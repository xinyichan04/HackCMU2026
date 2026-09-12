"""Photoreal one-shot face swap: InsightFace buffalo_l (detect + ArcFace identity) + inswapper_128.

A character's identity is ONE reference photo. Nothing is trained here: inswapper was trained once by its
authors on many identities. At runtime it takes the target face crop plus the reference's 512-d ArcFace
embedding and regenerates the crop as that identity with the target's pose, expression and lighting, then
pastes it back with its own alignment + blending.

Models (all gitignored under poc/models/, auto-downloaded on first use):
  insightface/models/buffalo_l/   detector (det_10g) + ArcFace (w600k_r50) — fetched by insightface itself
  inswapper_128.onnx              swapper (fp32, 554 MB; or --swapper inswapper_128_fp16.onnx, 278 MB)
  GFPGANv1.4.pth                  optional face enhancer (--enhance); halves fps, keep off for live use
Licences: inswapper / ArcFace / RetinaFace are non-commercial research; GFPGAN Apache-2.0.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

MODEL_URLS = {
    "inswapper_128.onnx": [
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
    ],
    "inswapper_128_fp16.onnx": [
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
    ],
    "GFPGANv1.4.pth": [
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth",
    ],
}


def ensure_download(path: Path) -> Path:
    if path.exists():
        return path
    urls = MODEL_URLS.get(path.name)
    if not urls:
        sys.exit(f"{path} is missing and has no known download URL")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for url in urls:
        try:
            print(f"downloading {path.name} from {url} ...", flush=True)
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
            return path
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}", flush=True)
            tmp.unlink(missing_ok=True)
    sys.exit(f"could not download {path.name}; fetch it manually into {path.parent}")


def onnx_providers(choice: str = "auto") -> list[str]:
    """Execution providers for onnxruntime. 'auto' = CoreML on macOS, CUDA if present elsewhere, else CPU."""
    import onnxruntime as ort
    have = set(ort.get_available_providers())
    if choice == "auto":
        if sys.platform == "darwin" and "CoreMLExecutionProvider" in have:
            choice = "coreml"
        elif "CUDAExecutionProvider" in have:
            choice = "cuda"
        else:
            choice = "cpu"
    table = {
        "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }
    provs = [p for p in table[choice] if p in have] or ["CPUExecutionProvider"]
    print(f"onnxruntime providers: {provs}", flush=True)
    return provs


class FaceSwapper:
    """detect() -> insightface Face objects (bbox, kps, det_score, normed_embedding); swap() one face at a time."""

    def __init__(self, swapper_path: Path | None = None, det_size: int = 640, provider: str = "auto",
                 enhancer: bool = False):
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        provs = onnx_providers(provider)
        self.app = FaceAnalysis(name="buffalo_l", root=str(MODELS / "insightface"),
                                allowed_modules=["detection", "recognition"], providers=provs)
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))
        path = ensure_download(Path(swapper_path) if swapper_path else MODELS / "inswapper_128.onnx")
        self.swapper = get_model(str(path), providers=provs)
        self._refs: dict[str, object] = {}
        self._enh = None
        if enhancer:
            self.enable_enhancer()

    # -- references -----------------------------------------------------------------------------------
    def load_reference(self, path: str | Path):
        """Largest face in the photo becomes the identity. Cached by path."""
        key = str(Path(path).resolve())
        if key in self._refs:
            return self._refs[key]
        img = cv2.imread(key)
        if img is None:
            raise FileNotFoundError(f"reference photo not readable: {key}")
        faces = self.app.get(img)
        if not faces:
            raise ValueError(f"no face found in reference photo {key} (use a frontal, well-lit, >=512px face)")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        self._refs[key] = face
        print(f"reference loaded: {Path(key).name} (det {face.det_score:.2f})", flush=True)
        return face

    def reference_for(self, character, pack_dir: Path, override: str | None = None):
        """Reference Face for a pack character, or None if the character has no photo on disk."""
        p = override or (str(Path(pack_dir) / character.ref) if getattr(character, "ref", None) else None)
        if not p or not Path(p).exists():
            return None
        return self.load_reference(p)

    # -- per frame ------------------------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray, max_faces: int = 1) -> list:
        faces = self.app.get(frame_bgr)
        faces.sort(key=lambda f: f.bbox[0])            # left to right in the image
        return faces[:max_faces]

    def swap(self, frame_bgr: np.ndarray, target_face, ref_face) -> np.ndarray:
        return self.swapper.get(frame_bgr, target_face, ref_face, paste_back=True)

    # -- optional enhancer ----------------------------------------------------------------------------
    def enable_enhancer(self) -> bool:
        if self._enh is not None:
            return True
        try:
            from gfpgan import GFPGANer
        except Exception as e:  # noqa: BLE001
            print(f"enhancer unavailable ({e}); pip install gfpgan basicsr facexlib torch", flush=True)
            return False
        model = ensure_download(MODELS / "GFPGANv1.4.pth")
        self._enh = GFPGANer(model_path=str(model), upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
        return True

    @property
    def enhancer_on(self) -> bool:
        return self._enh is not None

    def disable_enhancer(self):
        self._enh = None

    def enhance(self, frame_bgr: np.ndarray, face) -> np.ndarray:
        """GFPGAN on a padded crop around the face only, pasted back; ~80-150 ms per face on a Mac."""
        if self._enh is None:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = face.bbox
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.6, (y1 - y0) * 1.6
        X0, Y0 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
        X1, Y1 = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
        if X1 - X0 < 16 or Y1 - Y0 < 16:
            return frame_bgr
        crop = frame_bgr[Y0:Y1, X0:X1]
        _, _, out = self._enh.enhance(crop, has_aligned=False, only_center_face=True, paste_back=True)
        if out is not None and out.shape[:2] == crop.shape[:2]:
            frame_bgr[Y0:Y1, X0:X1] = out
        return frame_bgr


def draw_debug_photo(frame: np.ndarray, faces):
    for f in faces:
        x0, y0, x1, y1 = [int(v) for v in f.bbox]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
        for x, y in f.kps:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.putText(frame, f"det {f.det_score:.2f}", (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 1, cv2.LINE_AA)
