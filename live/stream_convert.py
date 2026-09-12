"""Live voice conversion: mic -> chunked conversion -> speaker, or
file-in/file-out for testing the same streaming code path without a mic.

The file-in/file-out mode (M4) and the live mic mode (M5) both go through
VoiceConverter's identical convert_chunk()/convert_file() path, so passing
M4 is real evidence the streaming logic (chunking, resampling, SOLA
crossfade) works before ever touching real audio hardware.

Usage:
    # M4: no mic/speaker needed
    python live/stream_convert.py --input-file in.wav --output-file out.wav

    # M5: live demo (run in the native Windows rvc-live env)
    python live/stream_convert.py --list-devices
    python live/stream_convert.py --input-device 1 --output-device 3
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from converter import VoiceConverter  # noqa: E402
import audio_io  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "training" / "config.yaml"


def default_model_name() -> str:
    if DEFAULT_CONFIG.is_file():
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
        return f"{cfg['experiment_name']}.pth"
    return "my_voice.pth"


def run_live(converter: VoiceConverter, input_device, output_device) -> None:
    # Producer/consumer split so GPU inference never runs inside the
    # PortAudio callback thread -- a slow block would otherwise risk
    # under/overruns rather than just a dropped/delayed block.
    in_q: "queue.Queue[object]" = queue.Queue(maxsize=4)
    out_q: "queue.Queue[object]" = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            try:
                block = in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                out_q.put(converter.convert_chunk(block), timeout=0.5)
            except queue.Full:
                pass  # consumer (audio callback) fell behind; drop this block

    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        try:
            in_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop input rather than build unbounded latency
        try:
            outdata[:, 0] = out_q.get_nowait()
        except queue.Empty:
            outdata.fill(0.0)  # brief underrun -> silence, not a glitch/crash

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    print(
        f"Streaming live: block={converter.block_frame} samples "
        f"(~{converter.block_time * 1000:.0f}ms) @ {converter.sample_rate} Hz. Ctrl+C to stop."
    )
    try:
        stream = audio_io.build_stream(
            audio_callback, converter.sample_rate, converter.block_frame, input_device, output_device,
        )
    except Exception as exc:
        # PortAudio reports device/format mismatches as bare error codes
        # (-9997 invalid sample rate, -9998 invalid channel count), which give
        # no hint that the fix is a different host API or rate. WASAPI in
        # shared mode is the usual culprit: it is locked to the device's mix
        # format, so it rejects both the model's 40 kHz rate and the mono
        # channel count this pipeline uses. MME and DirectSound go through the
        # Windows mixer, which converts, and accept both.
        stop_event.set()
        worker_thread.join(timeout=2)
        raise RuntimeError(
            f"Could not open the audio stream at {converter.sample_rate} Hz mono "
            f"(input={input_device}, output={output_device}): {exc}\n"
            "If this is PortAudio error -9997/-9998 on Windows, the device is refusing "
            "the rate or the mono channel count. Pick a DirectSound or MME device from "
            "--list-devices rather than a WASAPI one, and/or pass --sample-rate 48000."
        ) from exc
    with stream:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping.")
    stop_event.set()
    worker_thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                         help=f"Model filename in assets/weights, or a path. Defaults to {default_model_name()!r}.")
    parser.add_argument("--index", default="", help="Explicit .index path; auto-resolved from the model name if omitted.")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--f0-method", choices=["pm", "rmvpe"], default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--nprobe", type=int, default=8, help="IVF clusters searched per retrieval query (higher = more reliable, still cheap).")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="Open the audio device at this rate instead of the model's own "
                             "(40000 Hz for a 40k model). RVC's output is resampled to match, so "
                             "use this when a device rejects the model rate -- e.g. WASAPI in "
                             "shared mode, which is locked to the device's mix format.")
    parser.add_argument("--block-time", type=float, default=0.25, help="Chunk size in seconds (RVC's own default).")
    parser.add_argument("--crossfade-time", type=float, default=0.05, help="SOLA crossfade length in seconds.")
    parser.add_argument("--extra-time", type=float, default=2.5, help="Look-back context window in seconds.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device", default=None, help="Index or name substring.")
    parser.add_argument("--output-device", default=None, help="Index or name substring.")
    parser.add_argument("--input-file", default=None, help="M4 mode: convert this file instead of live audio.")
    parser.add_argument("--output-file", default=None, help="M4 mode: write converted output here.")
    args = parser.parse_args()

    if args.list_devices:
        print(audio_io.list_devices())
        return 0

    if bool(args.input_file) != bool(args.output_file):
        parser.error("--input-file and --output-file must be used together")

    # Resolve to absolute paths now: converter.load() chdirs the process
    # into external/rvc (RVC's own internals need that), which would
    # otherwise silently break relative --input-file/--output-file paths.
    if args.input_file:
        args.input_file = Path(args.input_file).resolve()
        args.output_file = Path(args.output_file).resolve()
    if args.index:
        args.index = str(Path(args.index).resolve())

    converter = VoiceConverter(
        block_time=args.block_time,
        crossfade_time=args.crossfade_time,
        extra_time=args.extra_time,
        f0_method=args.f0_method,
        pitch=args.pitch,
        index_rate=args.index_rate,
        nprobe=args.nprobe,
        sample_rate=args.sample_rate,
    )
    converter.load(args.model or default_model_name(), args.index)

    if args.input_file:
        print(f"Converting {args.input_file} -> {args.output_file} "
              f"(block={converter.block_frame} samples @ {converter.sample_rate} Hz)")
        converter.convert_file(args.input_file, args.output_file)
        print("Done.")
        return 0

    input_device = audio_io.resolve_device(args.input_device, "input")
    output_device = audio_io.resolve_device(args.output_device, "output")
    run_live(converter, input_device, output_device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
