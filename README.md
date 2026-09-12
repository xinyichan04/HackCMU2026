# HackCMU2026
le sserafim is so cute

## Using the trained voice model

The trained model is committed to this repo, so you do not need to retrain it:

| File | Size | What it is |
| --- | --- | --- |
| `models/chaewon_custom.pth` | 55 MB | RVC v2 generator, 40k sample rate, 250 epochs |
| `models/chaewon_custom.index` | 83 MB | FAISS retrieval index used for timbre matching |

Setup, from a fresh clone:

```bash
conda env create -f environment/environment-wsl.yml   # or -windows.yml
conda activate rvc-train
bash environment/setup_rvc.sh
```

`setup_rvc.sh` vendors the RVC WebUI into `external/`, downloads the pretrained
HuBERT/RMVPE/f0 assets, and copies the two files above into the locations
inference expects (`external/rvc/assets/weights` and `assets/indices`). The
`external/` checkout is roughly 3 GB once the pretrained assets land, which is
why it is gitignored rather than committed.

Convert a file:

```bash
python live/stream_convert.py --input-file input.wav --output-file output.wav
```

Live microphone conversion:

```bash
python live/stream_convert.py --list-devices      # find your device indices
python live/stream_convert.py --input-device 1 --output-device 5
```

Both default to `chaewon_custom.pth` (from `experiment_name` in
`training/config.yaml`) and auto-resolve the matching index. Useful flags:
`--pitch` in semitones, `--index-rate` (default 0.75), and `--nprobe` (default
8; lower values made retrieval intermittently miss).
