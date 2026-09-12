#!/usr/bin/env bash
# Vendors the RVC WebUI repo into external/rvc and downloads the pretrained
# assets our training/inference code depends on. Safe to re-run (skips
# anything already present).
#
# Usage:
#   conda activate rvc-train        # or rvc-live on native Windows
#   bash environment/setup_rvc.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVC_DIR="$REPO_ROOT/external/rvc"
RVC_REMOTE="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"

if [ ! -d "$RVC_DIR/.git" ]; then
  echo "Cloning RVC WebUI into $RVC_DIR ..."
  git clone --depth 1 "$RVC_REMOTE" "$RVC_DIR"
else
  echo "RVC WebUI already vendored at $RVC_DIR, skipping clone."
fi

# assets/weights and assets/indices aren't tracked in git (no files ever
# live there at clone time) and nothing in RVC's own training code creates
# them -- normally that's done at webui.py's Gradio startup, which we never
# run. Without this, a completed training run crashes at the very last step
# (extracting the final .pth into assets/weights).
mkdir -p "$RVC_DIR/assets/weights" "$RVC_DIR/assets/indices"

echo "Installing torch/torchaudio (CUDA 12.8 build) ..."
pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "Installing RVC's own Python dependencies into the active environment ..."
# The vendored requirements file pins a mirror as its own --index-url line
# (meant for contributors in mainland China); strip that so pip resolves
# against the default PyPI index instead.
REQ_FILE="$RVC_DIR/requirments_cu128_py312.txt"
grep -v '^--index-url' "$REQ_FILE" > /tmp/rvc_requirements_filtered.txt
pip install -r /tmp/rvc_requirements_filtered.txt
rm -f /tmp/rvc_requirements_filtered.txt

# Pretrained weights RVC needs: HuBERT content encoder, RMVPE pitch
# extractor, and the v2 base generator/discriminator our small dataset
# fine-tunes from. These are hosted on RVC's HuggingFace weights repo.
WEIGHTS_BASE="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

download() {
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "Already present: $dest"
  else
    echo "Downloading $(basename "$dest") ..."
    mkdir -p "$(dirname "$dest")"
    curl -fL --retry 3 -o "$dest" "$url"
  fi
}

# This RVC version loads HuBERT via Transformers (a converted checkpoint),
# not the legacy flat hubert_base.pt -- it needs the whole model folder.
download "$WEIGHTS_BASE/hubert_base/config.json" "$RVC_DIR/assets/hubert_base/config.json"
download "$WEIGHTS_BASE/hubert_base/preprocessor_config.json" "$RVC_DIR/assets/hubert_base/preprocessor_config.json"
download "$WEIGHTS_BASE/hubert_base/pytorch_model.bin" "$RVC_DIR/assets/hubert_base/pytorch_model.bin"
download "$WEIGHTS_BASE/rmvpe.pt" "$RVC_DIR/assets/rmvpe/rmvpe.pt"
download "$WEIGHTS_BASE/pretrained_v2/f0G40k.pth" "$RVC_DIR/assets/pretrained_v2/f0G40k.pth"
download "$WEIGHTS_BASE/pretrained_v2/f0D40k.pth" "$RVC_DIR/assets/pretrained_v2/f0D40k.pth"

# "mute" reference clips: training's filelist always mixes in a couple of
# silent reference lines per speaker (logs/mute/...), required by
# train/train.py regardless of dataset size.
if [ ! -d "$RVC_DIR/logs/mute" ]; then
  echo "Downloading mute.zip (training filler clips) ..."
  curl -fL --retry 3 -o /tmp/rvc_mute.zip "$WEIGHTS_BASE/mute.zip"
  mkdir -p "$RVC_DIR/logs"
  unzip -q -o /tmp/rvc_mute.zip -d "$RVC_DIR/logs"
  rm -f /tmp/rvc_mute.zip
else
  echo "Already present: $RVC_DIR/logs/mute"
fi

# The trained voice model is committed under models/ so teammates don't have
# to retrain it. Inference resolves a bare model name against assets/weights
# (see resolve_model_path in live/converter.py), so install it there. The
# index is copied into assets/indices, which is where RVC's own
# get_index_path_from_model looks when --index isn't passed explicitly.
if [ -f "$REPO_ROOT/models/chaewon_custom.pth" ]; then
  echo "Installing committed voice model into $RVC_DIR/assets ..."
  cp -f "$REPO_ROOT/models/chaewon_custom.pth" "$RVC_DIR/assets/weights/chaewon_custom.pth"
  cp -f "$REPO_ROOT/models/chaewon_custom.index" "$RVC_DIR/assets/indices/chaewon_custom.index"
else
  echo "No committed model at models/chaewon_custom.pth, skipping model install."
fi

echo "Done. RVC vendored at $RVC_DIR with required pretrained weights."
