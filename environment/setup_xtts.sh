#!/usr/bin/env bash
# Installs XTTS-v2 (zero-shot voice cloning text-to-speech) into its own
# conda env. Kept separate from rvc-train on purpose: coqui-tts pulls in
# numpy>=2 and (initially) transformers>=5, which directly conflict with
# RVC's pinned numpy<2 / transformers<4.50 -- installing both into one
# environment silently upgrades RVC's dependencies out from under it.
#
# Usage:
#   conda env create -f environment/environment-xtts.yml   # once
#   conda activate xtts
#   bash environment/setup_xtts.sh
set -euo pipefail

pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

pip install coqui-tts

# coqui-tts doesn't pin an upper bound on transformers, so a plain install
# grabs the latest (5.x), which removed an internal API
# (transformers.pytorch_utils.isin_mps_friendly) that coqui-tts's
# tortoise/xtts layers still import. Pin the latest compatible 4.x release.
pip install "transformers<5"

echo "Done. Model weights (~1.9GB) download automatically on first use,"
echo "cached to ~/.local/share/tts/. Test with:"
echo '  COQUI_TOS_AGREED=1 tts --text "Hello" --model_name tts_models/multilingual/multi-dataset/xtts_v2 \'
echo '    --speaker_wav data/raw/training5.mp3 --language_idx en --use_cuda --out_path testing/test.wav'
