"""Extract audio from arbitrary input files (mp4, mov, m4a, mp3, wav, ...)
into plain mono wav files, via ffmpeg. This is the only custom data-prep
step: RVC's own train/preprocess.py already does silence-based slicing,
resampling, and loudness normalization internally once it receives wav
input, so we don't duplicate that logic here.

Usage:
    python data/prep/extract_audio.py \\
        --input-dir data/raw \\
        --output-dir data/processed
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Anything ffmpeg can demux; video containers are included since a phone
# recording of the target speaker often comes in as mp4/mov.
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".webm", ".wma",
}


def extract_one(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn",              # drop any video stream
        "-ac", "1",         # mono
        "-acodec", "pcm_s16le",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}:\n{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"No such input directory: {args.input_dir}", file=sys.stderr)
        return 1

    sources = [
        p for p in sorted(args.input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not sources:
        print(f"No supported audio/video files found in {args.input_dir}")
        return 1

    for src in sources:
        rel_dir = src.relative_to(args.input_dir).parent
        # Keep the original extension in the output name (e.g. "take1.mp4"
        # -> "take1.mp4.wav") rather than just swapping to .wav: two source
        # files that share a stem but differ only in extension (a common
        # accident when raw recordings get re-exported) would otherwise
        # collide on the same output filename and silently overwrite.
        dest = args.output_dir / rel_dir / f"{src.name}.wav"
        print(f"{src} -> {dest}")
        extract_one(src, dest)

    print(f"Extracted {len(sources)} file(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
