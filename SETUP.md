# Real-time Chaewon reenactment filter — setup & usage

Animate a still photo of a character (default: Chaewon, LE SSERAFIM) with your **live
webcam expressions** (reenactment / puppeteering, not identity face-swap). Preview shows
in a window on screen. Runs on Apple Silicon with **no NVIDIA GPU** and **no training**.

**Responsible use:** the preview carries a persistent "AI filter — not real" watermark.
Chaewon is a real person — use this for the demo/learning only, don't present outputs as
real or use them to impersonate her.

## What powers it

- Engine: [`ivanfioravanti/fasterliveportrait-mlx`](https://github.com/ivanfioravanti/fasterliveportrait-mlx)
  — an Apple **MLX** port of LivePortrait (warping-based reenactment; pretrained, zero training).
- Installed at: `~/projects/fasterliveportrait-mlx` (managed by `uv`).
- Pretrained weights: HF `ivanfioravanti/FasterLivePortrait-MLX-weights` (auto-downloaded, cached).

## One-time setup (already done on this machine)

```bash
brew install ffmpeg uv                       # both already present
git clone https://github.com/ivanfioravanti/fasterliveportrait-mlx.git ~/projects/fasterliveportrait-mlx
cd ~/projects/fasterliveportrait-mlx
uv sync                                       # creates .venv with all deps
uv run python scripts/download_mlx_weights.py --repo-id ivanfioravanti/FasterLivePortrait-MLX-weights
```

## Run the live filter

From the workspace:

```bash
./run-filter.sh                 # Chaewon, turbo profile -> opens preview window; press q to quit
./run-filter.sh sources/chaewon_alt.jpg     # alternate Chaewon photo
./run-filter.sh sources/sakura.jpg          # another character (drop the photo in sources/ first)
```

First launch prompts macOS for **camera permission** (System Settings → Privacy & Security →
Camera → allow Terminal). The window title lists live hotkeys (S=stitching, X=animation
region, etc.); press **q** to exit.

Equivalent raw command:

```bash
cd ~/projects/fasterliveportrait-mlx
uv run python run.py --cfg configs/mlx_infer.yaml \
  --src_image <path>/sources/chaewon.jpg \
  --dri_video 0 --realtime --paste-back --mlx-profile turbo
```

## IMPORTANT: 8 GB RAM / memory

This Mac has **8 GB RAM**, which is the tight constraint. Heavy runs can push the system deep
into swap and stall. **Before running the live filter, close memory-hungry apps** (browser,
Spotify, etc.). Check pressure:

```bash
sysctl vm.swapusage        # want plenty of free swap, not ~0 free
memory_pressure | grep "free perc"
```

## Performance / profiles

- `turbo` — realtime path (reuses the warp field for up to 3 frames). Use this live.
- `quality` — much slower, offline only.
- Measured on this M3/8 GB: **<MEASURED_FPS>** (turbo). <!-- filled in after probe -->
- First run of a profile is slow (MLX compiles Metal kernels once); later runs are faster.

### Levers if it's too slow / thrashing

- Free RAM (close apps) — biggest lever on 8 GB.
- Lower webcam capture resolution.
- Try `--animation-region exp` (drive expression only) or toggle stitching off (S key).
- Keep the source image modest (the installed `chaewon.jpg` is 469×653, already fine).

## Add more characters (scale)

No retraining. Drop a clear, front-facing photo into `sources/` (see `sources/README.md`)
and pass its path to `run-filter.sh`. Switching identity = switching the source image.

## Watermark

On by default via a small patch in `run.py` (`_draw_watermark`). Toggle/customize:

```bash
FLIP_WATERMARK=0 ./run-filter.sh                       # disable
FLIP_WATERMARK_TEXT="deepfake demo" ./run-filter.sh    # custom text
```
