"""Measure a speaker's median pitch and print the --pitch value to use.

RVC does not normalize pitch to the target speaker: it transposes the source
F0 by a fixed number of semitones and conditions on the result. With the
default --pitch 0 the output lands on the *source* speaker's pitch, which is
why a deep voice converts to a deep-sounding Chaewon no matter how good the
model is. The shift has to be supplied per speaker, and this computes it.

Usage:
    python scripts/measure_pitch.py --input myvoice.wav
    python scripts/measure_pitch.py --record 5            # needs a mic
    python scripts/measure_pitch.py --input clip.wav --target-from data/processed
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Median F0 of the chaewon_custom training set, measured over all 24 processed
# clips (~11 min): 230.5 Hz, 10th-90th percentile 201.8-285.4 Hz. Baked in as a
# default so this works from the Windows checkout, where data/ is gitignored
# and the training audio is not present. Override with --target-from.
DEFAULT_TARGET_F0 = 230.5
SAMPLE_RATE = 16000
# Covers a deep male voice through a high female one; anything outside this is
# far more likely to be an octave error than a real fundamental.
F0_MIN, F0_MAX = 60, 600


def median_f0(paths, label):
    import librosa
    import numpy as np

    voiced = []
    for p in paths:
        audio, sr = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
        if len(audio) < sr * 0.3:
            continue
        f0, _, _ = librosa.pyin(audio, fmin=F0_MIN, fmax=F0_MAX, sr=sr, frame_length=1024)
        clean = f0[~np.isnan(f0)]
        if clean.size:
            voiced.append(clean)

    if not voiced:
        print(f"{label}: no voiced frames found -- is the clip silent or noise-only?",
              file=sys.stderr)
        return None

    allf = np.concatenate(voiced)
    med = float(np.median(allf))
    print(f"{label:22s} median={med:6.1f} Hz  "
          f"p10={np.percentile(allf, 10):6.1f}  p90={np.percentile(allf, 90):6.1f}  "
          f"voiced frames={allf.size}")
    return med


def record(seconds: float, dest: Path) -> None:
    """Record from the default input device.

    Uses sounddevice where it works (native Windows), falling back to ffmpeg
    against PulseAudio, since PortAudio enumerates no devices under WSL.
    """
    import numpy as np
    import soundfile as sf

    try:
        import sounddevice as sd
        if not [d for d in sd.query_devices() if d["max_input_channels"] > 0]:
            raise RuntimeError("no input devices")
        print(f"Recording {seconds:.0f}s -- speak normally ...", file=sys.stderr)
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        sf.write(str(dest), audio.squeeze(-1), SAMPLE_RATE)
        return
    except Exception as exc:
        print(f"sounddevice capture unavailable ({exc}); trying ffmpeg/PulseAudio.",
              file=sys.stderr)

    import subprocess
    print(f"Recording {seconds:.0f}s -- speak normally ...", file=sys.stderr)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "pulse", "-i", "RDPSource",
         "-t", str(seconds), "-ac", "1", "-ar", str(SAMPLE_RATE), str(dest)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="Audio file of the speaker.")
    parser.add_argument("--record", type=float, default=None,
                        help="Record this many seconds from the mic instead.")
    parser.add_argument("--target-from", default=None,
                        help="Directory of target-speaker wavs to measure instead of using "
                             f"the baked-in {DEFAULT_TARGET_F0} Hz.")
    args = parser.parse_args()

    if not args.input and args.record is None:
        parser.error("pass --input FILE or --record SECONDS")

    import numpy as np

    if args.target_from:
        files = sorted(Path(args.target_from).glob("*.wav"))
        if not files:
            print(f"No .wav files in {args.target_from}", file=sys.stderr)
            return 1
        target = median_f0(files[:12], "TARGET (measured)")
        if target is None:
            return 1
    else:
        target = DEFAULT_TARGET_F0
        print(f"{'TARGET (baked in)':22s} median={target:6.1f} Hz")

    with tempfile.TemporaryDirectory() as tmp:
        if args.record is not None:
            src_path = Path(tmp) / "recorded.wav"
            record(args.record, src_path)
        else:
            src_path = Path(args.input)
            if not src_path.is_file():
                print(f"No such file: {src_path}", file=sys.stderr)
                return 1

        source = median_f0([src_path], "SOURCE")
        if source is None:
            return 1

    semitones = 12 * np.log2(target / source)
    print()
    print(f"  --pitch {round(semitones):+d}        (exact: {semitones:+.1f} semitones)")
    if abs(semitones) >= 10:
        print()
        print("  A shift this large drags the formants up with it, which is what makes a")
        print("  converted voice sound thin. Pair it with --formant -0.3 (to -0.5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
