#!/usr/bin/env bash
# Real-time reenactment filter: animate a still portrait with your live webcam expressions.
# Usage:
#   ./run-filter.sh                         # uses sources/chaewon.jpg, turbo profile
#   ./run-filter.sh sources/sakura.jpg      # different character (no retraining)
#   ./run-filter.sh sources/chaewon.jpg quality   # slower, higher quality
#
# Press 'q' in the preview window to quit.
set -euo pipefail

REPO="${FLIP_REPO:-$HOME/projects/fasterliveportrait-mlx}"
SRC="${1:-$(cd "$(dirname "$0")" && pwd)/sources/chaewon.jpg}"
PROFILE="${2:-turbo}"

if [[ ! -f "$SRC" ]]; then
  echo "Source image not found: $SRC"
  echo "Drop a clear, front-facing photo at sources/chaewon.jpg (see sources/README.md)."
  exit 1
fi

cd "$REPO"
exec uv run python run.py \
  --cfg configs/mlx_infer.yaml \
  --src_image "$SRC" \
  --dri_video 0 \
  --realtime \
  --paste-back \
  --mlx-profile "$PROFILE"
