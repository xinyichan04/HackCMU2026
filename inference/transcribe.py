"""Speech-to-text with NVIDIA Parakeet TDT, via the NeMo toolkit.

Runs in the `parakeet` conda env (environment/environment-parakeet.yml), NOT
in rvc-train -- see that file for why the two stacks are kept apart.

Parakeet TDT 0.6B v2 is an offline/buffered model: the throughput figures it
posts on the Open ASR leaderboard are batch RTFx over long files, which is a
different quantity from live streaming latency. --benchmark below therefore
reports both, so the number that actually matters for a live demo (wall-clock
latency to transcribe one short utterance) is visible rather than inferred.

Usage:
    python inference/transcribe.py --input clip.wav
    python inference/transcribe.py --input clip.wav --benchmark --runs 5
    python inference/transcribe.py --list-devices
    python inference/transcribe.py --mic --seconds 5
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
# Parakeet's encoder expects 16 kHz mono; NeMo resamples internally for file
# input, but mic capture is opened at this rate directly to avoid a resample.
SAMPLE_RATE = 16000


def load_model(name: str):
    # Imported lazily: `import nemo` pulls in a large dependency tree and
    # several seconds of startup, which we don't want to pay for --list-devices.
    import nemo.collections.asr as nemo_asr

    print(f"Loading {name} ...", file=sys.stderr)
    t0 = time.perf_counter()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=name)
    model.eval()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return model


def audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


def transcribe(model, paths: list[str]) -> list[str]:
    out = model.transcribe(paths)
    # NeMo's return type varies by model class and version: some return plain
    # strings, others Hypothesis objects carrying .text alongside timestamps.
    return [getattr(item, "text", item) for item in out]


def run_benchmark(model, path: Path, runs: int) -> None:
    import torch

    dur = audio_duration(path)

    # The first call pays CUDA context init, kernel autotuning and cuDNN
    # algorithm selection -- including it would overstate steady-state latency
    # by an order of magnitude, so it is run and discarded.
    print("Warm-up run (discarded) ...", file=sys.stderr)
    transcribe(model, [str(path)])

    times = []
    for i in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = transcribe(model, [str(path)])[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        print(f"  run {i + 1}/{runs}: {times[-1] * 1000:7.1f} ms", file=sys.stderr)

    times.sort()
    median = times[len(times) // 2]
    print()
    print(f"audio duration : {dur:.2f} s")
    print(f"median latency : {median * 1000:.1f} ms")
    print(f"min / max      : {times[0] * 1000:.1f} / {times[-1] * 1000:.1f} ms")
    print(f"RTFx           : {dur / median:.1f}x realtime")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak VRAM      : {peak:.2f} GiB")
    print()
    print(f"transcript     : {text}")


def list_devices() -> int:
    """List PulseAudio sources, not PortAudio devices.

    sounddevice/PortAudio is unusable for capture under WSL: PortAudio
    discovers ALSA devices by iterating hardware cards, WSL exposes none, so
    it reports zero devices even with libasound2-plugins installed and ALSA
    correctly routed to the WSLg PulseServer (ffmpeg -f alsa -i default works
    fine at that point). Advertising the PCM with an ALSA `hint` block to
    force enumeration makes Pa_Initialize fail outright. PulseAudio is the
    supported capture path here, so query it directly.
    """
    out = subprocess.run(
        ["pactl", "list", "short", "sources"], capture_output=True, text=True,
    )
    if out.returncode != 0:
        print("pactl failed -- is pulseaudio-utils installed and PULSE_SERVER set?",
              file=sys.stderr)
        return 1
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if not lines:
        print("No PulseAudio sources found.", file=sys.stderr)
        return 1
    for line in lines:
        fields = line.split("\t")
        name = fields[1] if len(fields) > 1 else line
        # .monitor sources are loopbacks of output, not microphones.
        kind = "monitor (output loopback)" if name.endswith(".monitor") else "INPUT"
        print(f"{name}\t{kind}")
    return 0


def capture(seconds: float, source: str, dest: Path) -> None:
    """Record from a PulseAudio source into a 16 kHz mono wav via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "pulse", "-i", source,
        "-t", str(seconds),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        str(dest),
    ]
    subprocess.run(cmd, check=True)


def run_mic(model, seconds: float, source: str) -> int:
    import numpy as np
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "capture.wav"
        print(f"Recording {seconds:.0f}s from {source} ...", file=sys.stderr)
        capture(seconds, source, wav)

        audio, _ = sf.read(str(wav))
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak < 1e-4:
            print(f"WARNING: captured audio is essentially silent (peak={peak:.2e}). "
                  "The source opened but no signal reached it -- check that Windows "
                  "is not muting the mic for the WSL session.", file=sys.stderr)

        t0 = time.perf_counter()
        text = transcribe(model, [str(wav)])[0]
        elapsed = time.perf_counter() - t0

    print(f"\nlatency: {elapsed * 1000:.1f} ms for {seconds:.1f}s of audio "
          f"({seconds / elapsed:.1f}x realtime)")
    print(f"transcript: {text}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="Audio file to transcribe.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--benchmark", action="store_true", help="Time repeated runs on --input.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--mic", action="store_true", help="Record from the microphone, then transcribe.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Mic recording length.")
    parser.add_argument("--source", default="RDPSource",
                        help="PulseAudio source name (see --list-devices). "
                             "RDPSource is the WSLg microphone.")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        return list_devices()

    if not args.mic and not args.input:
        parser.error("pass --input FILE, or --mic, or --list-devices")

    model = load_model(args.model)

    if args.mic:
        return run_mic(model, args.seconds, args.source)

    path = Path(args.input).resolve()
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    if args.benchmark:
        run_benchmark(model, path, args.runs)
    else:
        print(transcribe(model, [str(path)])[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
