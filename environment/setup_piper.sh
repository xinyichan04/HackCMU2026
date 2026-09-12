#!/usr/bin/env bash
# Installs Piper (offline neural TTS) and downloads one voice model, used as
# the text-to-speech step feeding into RVC's voice conversion (RVC only
# converts existing audio -- it can't generate speech from text -- so
# inference/text_to_voice.py runs Piper first, then pipes the result
# through the same conversion path as inference/convert_offline.py).
#
# Usage:
#   conda activate rvc-train
#   bash environment/setup_piper.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_DIR="$REPO_ROOT/external/piper/voices"
VOICE_NAME="en_US-lessac-medium"
VOICES_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

pip install piper-tts

mkdir -p "$VOICE_DIR"
for ext in onnx onnx.json; do
  dest="$VOICE_DIR/$VOICE_NAME.$ext"
  if [ -f "$dest" ]; then
    echo "Already present: $dest"
  else
    echo "Downloading $VOICE_NAME.$ext ..."
    curl -fL --retry 3 -o "$dest" "$VOICES_BASE/$VOICE_NAME.$ext"
  fi
done

echo "Done. Piper voice installed at $VOICE_DIR/$VOICE_NAME.onnx"
