"""File-to-file voice conversion: convert a wav (or any ffmpeg-readable
audio file) into the trained target voice, with no audio devices involved.

This is the M3 milestone gate -- it proves the trained model itself sounds
like the target speaker, before any streaming/hardware work is attempted.
It's a thin wrapper around RVC's own infer/cli.py rather than a
reimplementation.

Usage:
    python inference/convert_offline.py --input some_clip.wav --output out.wav
    python inference/convert_offline.py --input some_clip.wav --output out.wav \\
        --model my_voice.pth --pitch 0 --index-rate 0.75
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RVC_DIR = REPO_ROOT / "external" / "rvc"
DEFAULT_CONFIG = REPO_ROOT / "training" / "config.yaml"


def default_model_name() -> str:
    if DEFAULT_CONFIG.is_file():
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
        return f"{cfg['experiment_name']}.pth"
    return "my_voice.pth"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source audio file (any format ffmpeg reads).")
    parser.add_argument("--output", required=True, help="Output audio file path (.wav/.flac/.mp3/.m4a).")
    parser.add_argument("--model", default=None,
                         help=f"Model filename in assets/weights, or a path. Defaults to {default_model_name()!r} "
                              "(from training/config.yaml's experiment_name).")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--f0-method", choices=["pm", "rmvpe"], default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model = args.model or default_model_name()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    cmd = [
        sys.executable, "-m", "infer.cli",
        "--model", model,
        "--input", str(input_path),
        "--output", str(output_path),
        "--pitch", str(args.pitch),
        "--f0-method", args.f0_method,
        "--index-rate", str(args.index_rate),
        "--protect", str(args.protect),
    ]
    if args.overwrite:
        cmd.append("--overwrite")

    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=RVC_DIR)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
