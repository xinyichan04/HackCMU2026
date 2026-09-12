import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from avatar import AvatarRenderer, Pack
from synth import synthetic_face
from tracker import FaceTracker

POC = Path(__file__).resolve().parents[1]
MODEL = POC / "models" / "face_landmarker.task"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run poc/live.py once to download the model")


@pytest.fixture(scope="module")
def tracker():
    t = FaceTracker(str(MODEL), max_faces=2, smoothing=0.0)
    yield t
    t.close()


def settled(tracker, img, ts):
    """VIDEO mode re-detects one frame late when the number of faces changes."""
    tracker.track(img, ts)
    return tracker.track(img, ts + 33)


def test_pack_loads_and_indexes():
    pack = Pack.load(POC / "packs" / "le-sserafim")
    assert pack.index("chaewon") == 0
    assert len(pack.characters) >= 2
    with pytest.raises(KeyError):
        pack.index("nobody")


def test_tracker_finds_synthetic_face(tracker):
    faces = tracker.track(synthetic_face(), 0)
    assert len(faces) == 1
    f = faces[0]
    assert f.pts.shape == (478, 2)
    assert "jawOpen" in f.shapes
    x0, y0, x1, y1 = f.bbox
    assert 400 < x0 < 640 < x1 < 900 and 100 < y0 < 360 < y1 < 650
    assert abs(f.yaw) < 0.3 and abs(f.roll) < 0.2


def test_tracker_two_faces(tracker):
    img = synthetic_face(cx=380, scale=0.8)
    right = synthetic_face(cx=900, scale=0.8)
    img[:, 640:] = right[:, 640:]
    faces = settled(tracker, img, 1000)
    assert len(faces) == 2


def test_render_every_character_changes_the_face_region(tracker, tmp_path):
    pack = Pack.load(POC / "packs" / "le-sserafim")
    renderer = AvatarRenderer()
    base = synthetic_face()
    faces = settled(tracker, base, 2000)
    outs = []
    for i, ch in enumerate(pack.characters):
        out = renderer.render(base.copy(), faces[0], ch)
        assert out.shape == base.shape and out.dtype == np.uint8
        x0, y0, x1, y1 = faces[0].bbox
        diff = np.abs(out[y0:y1, x0:x1].astype(int) - base[y0:y1, x0:x1].astype(int)).mean()
        assert diff > 20, f"{ch.id}: face region barely changed (mean diff {diff:.1f})"
        cv2.imwrite(str(tmp_path / f"{ch.id}.png"), out)
        outs.append(out)
    # characters must actually look different from each other (data-driven)
    assert np.abs(outs[0].astype(int) - outs[1].astype(int)).mean() > 1


def test_render_speed(tracker):
    pack = Pack.load(POC / "packs" / "le-sserafim")
    renderer = AvatarRenderer()
    base = synthetic_face()
    face = settled(tracker, base, 3000)[0]
    renderer.render(base.copy(), face, pack.characters[0])  # warm up
    n = 20
    t = time.perf_counter()
    for _ in range(n):
        renderer.render(base.copy(), face, pack.characters[0])
    ms = (time.perf_counter() - t) / n * 1000
    assert ms < 40, f"render {ms:.1f} ms/frame"


def test_live_cli_on_video_file(tmp_path):
    # build a short synthetic clip with a moving face, run the CLI headless, check it records
    src = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for i in range(24):
        w.write(synthetic_face(cx=560 + i * 6, mouth_open=i % 8 < 4))
    w.release()
    out = tmp_path / "out.mp4"
    png = tmp_path / "last.png"
    r = subprocess.run([sys.executable, str(POC / "live.py"), "--source", str(src), "--record", str(out),
                        "--no-preview", "--save-frame", str(png), "--character", "eunchae"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.exists() and out.stat().st_size > 10_000
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
    assert png.exists()
