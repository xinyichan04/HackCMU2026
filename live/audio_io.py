"""sounddevice device listing and stream construction helpers.

Kept separate from live/stream_convert.py so the same helpers are reusable
regardless of what's driving the actual conversion loop.

`sounddevice` is imported lazily inside each function rather than at module
level: it requires the native PortAudio library, which isn't installed in
the WSL side of this project on purpose (mic/speaker I/O runs on native
Windows -- see the project plan). Importing it lazily means
live/stream_convert.py's --input-file/--output-file mode (M4) still works
in WSL without touching audio hardware or needing PortAudio installed there
at all; only --list-devices and live mode (M5, run on native Windows)
actually need it.
"""


def list_devices() -> str:
    import sounddevice as sd
    return str(sd.query_devices())


def resolve_device(value, kind: str):
    """Accepts an int index, a numeric string, a substring of a device
    name, or None (-> sounddevice's current default for `kind`,
    'input'/'output')."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    import sounddevice as sd
    devices = sd.query_devices()
    matches = [
        i for i, d in enumerate(devices)
        if value.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0
    ]
    if not matches:
        raise ValueError(f"No {kind} device matching {value!r}. Run --list-devices to see options.")
    if len(matches) > 1:
        names = ", ".join(f"{i}:{devices[i]['name']}" for i in matches)
        raise ValueError(f"Ambiguous {kind} device {value!r}, matches: {names}")
    return matches[0]


def build_stream(callback, samplerate: int, blocksize: int, input_device=None, output_device=None):
    import sounddevice as sd
    return sd.Stream(
        callback=callback,
        blocksize=blocksize,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=(input_device, output_device),
    )
