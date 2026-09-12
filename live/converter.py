"""Embeddable real-time voice conversion core.

Wraps RVC's own real-time inference engine (infer.rtrvc.RVC) with the same
block/crossfade (SOLA -- Synchronous OverLap-Add, from DDSP-SVC) math as
RVC's own realtime_gui.py, factored out of that GUI so it can be embedded
in another process: no argparse, no global state, no GUI dependency, and no
CLI-only assumptions (see the project plan's "Portability" section --
this is the module a teammate's larger app is meant to import once its
architecture lands).

Deliberately left out of this MVP port (present in RVC's own GUI, but not
needed to prove the core pipeline works): input/output noise gating,
RMS volume-envelope mixing, WASAPI-exclusive mode, and CUDA Graph
prewarming. None of these affect whether the conversion+crossfade logic is
correct -- they're quality/perf refinements that can be layered on later.

Two important constraints inherited from vendoring RVC as-is (documented
rather than hidden, since they matter for whoever embeds this):
  - RVC's internals reference paths relative to its own repo root (e.g.
    "assets/rmvpe/rmvpe.pt"), so load() chdirs the process into
    external/rvc. If this is embedded in a process that needs a different
    cwd for unrelated work, run VoiceConverter in its own subprocess.
  - configs.config.Config() parses real sys.argv on construction; load()
    temporarily blanks sys.argv around that call (the same guard RVC's own
    infer/cli.py uses) so a host script's own CLI args aren't misread.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RVC_DIR = REPO_ROOT / "external" / "rvc"


def resolve_model_path(value: str) -> str:
    """Mirrors infer/cli.py's resolve_model(): accepts an absolute path, a
    path relative to the RVC checkout, or a bare filename in
    assets/weights (where training/train.py leaves trained models)."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (RVC_DIR / candidate).resolve()
    if not candidate.is_file():
        candidate = (RVC_DIR / "assets" / "weights" / value).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Model not found: {value}")
    return str(candidate)


class VoiceConverter:
    def __init__(
        self,
        block_time: float = 0.25,
        crossfade_time: float = 0.05,
        extra_time: float = 2.5,
        f0_method: str = "rmvpe",
        pitch: int = 0,
        formant: float = 0.0,
        index_rate: float = 0.75,
        nprobe: int = 8,
        sample_rate: int | None = None,
    ):
        """Defaults (block_time/crossfade_time/extra_time) mirror RVC's own
        realtime_gui.py defaults -- they're already tuned for this exact
        algorithm, not a guess. sample_rate=None uses the model's own
        target sample rate.

        nprobe: how many IVF clusters the retrieval index searches per
        query. train_index.py always bakes in nprobe=1 (searches only the
        single nearest cluster), which is too restrictive for indexes with
        uneven cluster sizes -- a query landing in a sparse cluster (fewer
        than the requested 8 neighbors) silently falls back to no
        retrieval blending for that chunk. Raising it (safe to change at
        search time, no retraining needed) makes retrieval succeed far
        more consistently at a small, real-time-safe compute cost.
        """
        self.block_time = block_time
        self.crossfade_time = crossfade_time
        self.extra_time = extra_time
        self.f0_method = f0_method
        self.pitch = pitch
        self.formant = formant
        self.index_rate = index_rate
        self.nprobe = nprobe
        self._forced_sample_rate = sample_rate
        self._loaded = False

    def load(self, model_path: str, index_path: str = "") -> None:
        if str(RVC_DIR) not in sys.path:
            sys.path.insert(0, str(RVC_DIR))
        os.chdir(RVC_DIR)
        # Same defaults infer/cli.py sets, needed by RVC internals (e.g.
        # get_index_path_from_model) regardless of which entry point loaded them.
        os.environ.setdefault("weight_root", str(RVC_DIR / "assets" / "weights"))
        os.environ.setdefault("index_root", str(RVC_DIR / "logs"))
        os.environ.setdefault("outside_index_root", str(RVC_DIR / "assets" / "indices"))
        os.environ.setdefault("rmvpe_root", str(RVC_DIR / "assets" / "rmvpe"))

        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        try:
            from configs.config import Config
            self.config = Config()
        finally:
            sys.argv = original_argv

        resolved_model = resolve_model_path(model_path)
        index_rate = self.index_rate
        resolved_index = str(index_path) if index_path else ""
        if index_rate != 0 and not resolved_index:
            from infer.vc.utils import get_index_path_from_model
            resolved_index = get_index_path_from_model(Path(resolved_model).name, speaker_id=0)
            if not resolved_index:
                print(f"No retrieval index found for {model_path}; continuing without one (index_rate=0).",
                      file=sys.stderr)
                index_rate = 0

        from infer import rtrvc as rvc_for_realtime
        self.rvc = rvc_for_realtime.RVC(
            self.pitch, self.formant, resolved_model, resolved_index,
            index_rate, self.config,
        )
        if not hasattr(self.rvc, "tgt_sr"):
            # rtrvc.RVC.__init__ catches its own exceptions and just logs
            # them (printt(traceback.format_exc())) rather than raising, so
            # a failure here (e.g. a corrupted/unreadable index file) would
            # otherwise surface as a confusing unrelated AttributeError a
            # few lines down. Fail loudly with the actual likely cause.
            raise RuntimeError(
                f"Failed to initialize the RVC model (see the traceback printed above for the real "
                f"cause). A common culprit is a corrupted or unreadable index file at "
                f"{resolved_index!r} -- try --index-rate 0 to skip the index entirely, or pass a "
                f"different --index path, to confirm whether that's the issue."
            )
        if hasattr(self.rvc, "index"):
            self.rvc.index.nprobe = self.nprobe

        device = self.config.device
        self.sample_rate = self._forced_sample_rate or self.rvc.tgt_sr
        self.zc = self.sample_rate // 100
        self.block_frame = int(round(self.block_time * self.sample_rate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(round(self.crossfade_time * self.sample_rate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(round(self.extra_time * self.sample_rate / self.zc)) * self.zc

        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=device, dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc, device=device, dtype=torch.float32
        )
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=device, dtype=torch.float32)
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=device, dtype=torch.float32)
            ) ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window

        import torchaudio.transforms as tat
        self.resampler = tat.Resample(orig_freq=self.sample_rate, new_freq=16000, dtype=torch.float32).to(device)
        self.resampler2 = (
            tat.Resample(orig_freq=self.rvc.tgt_sr, new_freq=self.sample_rate, dtype=torch.float32).to(device)
            if self.rvc.tgt_sr != self.sample_rate else None
        )
        self._loaded = True

    def reset(self) -> None:
        """Clear rolling buffers. Call before converting a new, unrelated stream."""
        self.input_wav.zero_()
        self.input_wav_res.zero_()
        self.sola_buffer.zero_()
        self.rvc.cache_pitch.zero_()
        self.rvc.cache_pitchf.zero_()

    def convert_chunk(self, block: np.ndarray) -> np.ndarray:
        """Feed exactly `self.block_frame` mono float32 PCM samples at
        `self.sample_rate`, from a contiguous live stream (each call must
        follow directly on from the previous one -- this is stateful).
        Returns `self.block_frame` samples of converted audio.
        """
        if not self._loaded:
            raise RuntimeError("VoiceConverter.load() must be called first")
        if block.shape[0] != self.block_frame:
            raise ValueError(f"expected a block of {self.block_frame} samples, got {block.shape[0]}")

        device = self.config.device
        block_t = torch.from_numpy(np.asarray(block, dtype=np.float32)).to(device)

        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-self.block_frame:] = block_t

        resample_input = self.input_wav[-self.block_frame - 2 * self.zc:]
        self.input_wav_res[-160 * (self.block_frame // self.zc + 1):] = self.resampler(resample_input)[160:]

        infer_wav = self.rvc.infer(
            self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, self.f0_method,
        )
        if self.resampler2 is not None:
            infer_wav = self.resampler2(infer_wav)

        # SOLA: find the offset in the newly-inferred chunk whose start best
        # matches (by normalized cross-correlation) the crossfade tail left
        # over from the previous chunk, then fade across it. Without this,
        # consecutive independently-inferred chunks click at their seams.
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = torch.nn.functional.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            torch.nn.functional.conv1d(conv_input ** 2, self.sola_den_kernel) + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame: self.block_frame + self.sola_buffer_frame]

        return infer_wav[: self.block_frame].detach().cpu().numpy()

    def convert_file(self, in_path, out_path) -> None:
        """File-to-file conversion via the exact same per-block code path as
        the live stream. This is what lets the M4 milestone validate the
        streaming/crossfade logic without a microphone.
        """
        import soundfile as sf
        import librosa

        audio, _ = librosa.load(str(in_path), sr=self.sample_rate, mono=True)
        self.reset()
        pad = (-len(audio)) % self.block_frame
        audio = np.pad(audio, (0, pad))

        chunks = [
            self.convert_chunk(audio[start: start + self.block_frame])
            for start in range(0, len(audio), self.block_frame)
        ]
        sf.write(str(out_path), np.concatenate(chunks), self.sample_rate)
