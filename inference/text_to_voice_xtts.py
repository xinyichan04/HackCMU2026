"""Text -> speech, directly in a cloned voice, via XTTS-v2 zero-shot cloning.

Unlike inference/text_to_voice.py (Piper TTS -> RVC conversion, two separate
models bolted together), this is a single model that takes text plus a short
reference clip of the target voice and produces speech in that voice
directly -- the same technique Fish Audio and similar services use for
their large voice libraries (one model trained once on a huge, diverse,
transcribed corpus; every "new voice" is just a reference clip supplied at
inference time, not a per-voice trained model).

Must be run in the `xtts` conda env, NOT rvc-train -- coqui-tts's
dependencies (numpy>=2, transformers) conflict with RVC's pinned versions.
See environment/setup_xtts.sh.

Usage:
    conda activate xtts
    python inference/text_to_voice_xtts.py --text "Hello there!"
    python inference/text_to_voice_xtts.py --text "..." --speaker-wav data/raw/training15.mp3
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "testing"
DEFAULT_SPEAKER_WAV = REPO_ROOT / "data" / "raw" / "training5.mp3"


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug[:max_len] or "output"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Text to speak. Omit to use --text-file or stdin.")
    parser.add_argument("--text-file", type=Path, help="Read text from this file instead of --text.")
    parser.add_argument("--speaker-wav", type=Path, default=DEFAULT_SPEAKER_WAV,
                         help=f"Reference clip of the target voice (~6-30s of clean audio). Default: {DEFAULT_SPEAKER_WAV.relative_to(REPO_ROOT)}")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Explicit output path (overrides --output-dir naming).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of CUDA.")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.text_file:
        text = args.text_file.read_text()
    else:
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        print("No text provided (use --text, --text-file, or pipe text via stdin).", file=sys.stderr)
        return 1

    if not args.speaker_wav.is_file():
        print(f"Reference clip not found: {args.speaker_wav}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.output_dir / f"xtts_{timestamp}_{slugify(text)}.wav"

    from TTS.api import TTS

    print(f"Loading XTTS-v2 ({'CPU' if args.cpu else 'GPU'}) ...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=not args.cpu)

    print(f"Cloning voice from {args.speaker_wav} ...")
    tts.tts_to_file(
        text=text,
        speaker_wav=str(args.speaker_wav),
        language=args.language,
        file_path=str(output_path),
    )
    print(f"Done: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
