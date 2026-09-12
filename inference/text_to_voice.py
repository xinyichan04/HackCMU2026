"""Text -> speech (Piper TTS) -> voice conversion (RVC).

RVC only converts existing audio -- it has no way to generate speech from
text on its own. This adds the missing front end: synthesize generic
speech with Piper (offline neural TTS), then run that through the same
RVC conversion path as inference/convert_offline.py to re-voice it as the
target speaker. Outputs land in testing/ by default (gitignored via the
repo's blanket *.wav rule, so generated clips never accidentally get
committed).

Usage:
    python inference/text_to_voice.py --text "Hello there!"
    python inference/text_to_voice.py --text "..." --model chaewon.pth --pitch 12
    echo "Piped text works too" | python inference/text_to_voice.py
"""

import argparse
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RVC_DIR = REPO_ROOT / "external" / "rvc"
DEFAULT_PIPER_VOICE = REPO_ROOT / "external" / "piper" / "voices" / "en_US-lessac-medium.onnx"
DEFAULT_CONFIG = REPO_ROOT / "training" / "config.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "testing"


def default_model_name() -> str:
    if DEFAULT_CONFIG.is_file():
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
        return f"{cfg['experiment_name']}.pth"
    return "my_voice.pth"


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug[:max_len] or "output"


def synthesize_speech(text: str, voice: Path, out_wav: Path) -> None:
    if not voice.is_file():
        raise FileNotFoundError(f"Piper voice not found at {voice}. Run environment/setup_piper.sh first.")
    cmd = [sys.executable, "-m", "piper", "-m", str(voice), "-f", str(out_wav)]
    result = subprocess.run(cmd, input=text, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"piper failed:\n{result.stderr}")


def convert_voice(input_wav: Path, output_wav: Path, model: str, pitch: int,
                   f0_method: str, index_rate: float, protect: float) -> None:
    cmd = [
        sys.executable, "-m", "infer.cli",
        "--model", model,
        "--input", str(input_wav),
        "--output", str(output_wav),
        "--pitch", str(pitch),
        "--f0-method", f0_method,
        "--index-rate", str(index_rate),
        "--protect", str(protect),
        "--overwrite",
    ]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=RVC_DIR)
    if result.returncode != 0:
        raise RuntimeError("infer.cli failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Text to speak. Omit to use --text-file or stdin.")
    parser.add_argument("--text-file", type=Path, help="Read text from this file instead of --text.")
    parser.add_argument("--model", default=None, help=f"RVC model filename. Defaults to {default_model_name()!r}.")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--f0-method", choices=["pm", "rmvpe"], default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--voice", type=Path, default=DEFAULT_PIPER_VOICE, help="Piper .onnx voice model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Explicit output path (overrides --output-dir naming).")
    parser.add_argument("--keep-tts", action="store_true", help="Also save the untransformed intermediate TTS audio.")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.output_dir / f"{timestamp}_{slugify(text)}.wav"

    with tempfile.TemporaryDirectory() as tmp:
        tts_wav = Path(tmp) / "tts.wav"
        print(f"Synthesizing speech with Piper ({args.voice.name}) ...")
        synthesize_speech(text, args.voice, tts_wav)

        if args.keep_tts:
            kept_tts = output_path.with_name(output_path.stem + "_tts_only.wav")
            kept_tts.write_bytes(tts_wav.read_bytes())
            print(f"Kept intermediate TTS audio: {kept_tts}")

        print(f"Converting to target voice ({args.model or default_model_name()}) ...")
        convert_voice(
            tts_wav, output_path, args.model or default_model_name(),
            args.pitch, args.f0_method, args.index_rate, args.protect,
        )

    print(f"Done: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
