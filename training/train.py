"""Thin orchestrator around RVC's own CLI training scripts.

This deliberately does NOT reimplement RVC's preprocessing/training/index
logic -- it just runs the same sequence of scripts RVC's own webui.py runs
from its "Train" tab (preprocess -> f0 extraction -> feature extraction ->
filelist/config generation -> train -> build index), as plain subprocesses
inside the vendored external/rvc checkout. Single-speaker only (index/gpu 0)
-- multi-speaker training is out of scope for this MVP.

Usage:
    python training/train.py --trainset-dir data/processed \\
        [--config training/config.yaml]
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RVC_DIR = REPO_ROOT / "external" / "rvc"


def run(args, **kwargs):
    print(f"$ {' '.join(str(a) for a in args)}")
    subprocess.run(args, cwd=RVC_DIR, check=True, **kwargs)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sample_rate_hz(sr_label: str) -> int:
    return {"32k": 32000, "40k": 40000, "48k": 48000}[sr_label]


def write_filelist_and_config(exp_dir: Path, cfg: dict) -> None:
    """Reproduces the single-speaker branch of RVC webui.py's
    run_train_model(): builds logs/<exp>/filelist.txt and config.json.
    """
    version = cfg["version"]
    if_f0 = cfg["f0"]
    speaker_id = cfg["speaker_id"]
    sr_label = cfg["sample_rate"]

    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")

    def stems(d: Path) -> set:
        return {p.name.split(".")[0] for p in d.iterdir()}

    if if_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names = stems(gt_wavs_dir) & stems(feature_dir) & stems(f0_dir) & stems(f0nsf_dir)
    else:
        names = stems(gt_wavs_dir) & stems(feature_dir)

    if not names:
        raise RuntimeError(
            "No usable audio survived preprocessing/feature extraction -- "
            "check logs/<experiment>/preprocess.log and extract_f0_feature.log"
        )

    lines = []
    for name in sorted(names):
        if if_f0:
            lines.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|"
                f"{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{speaker_id}"
            )
        else:
            lines.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{speaker_id}")

    # RVC always mixes in a couple of silent "mute" reference lines per
    # speaker (from external/rvc/logs/mute, fetched by setup_rvc.sh) so the
    # model sees at least one guaranteed-consistent example.
    fea_dim = 256 if version == "v1" else 768
    mute_dir = RVC_DIR / "logs" / "mute"
    for _ in range(2):
        if if_f0:
            lines.append(
                f"{mute_dir}/0_gt_wavs/mute{sr_label}.wav|"
                f"{mute_dir}/3_feature{fea_dim}/mute.npy|"
                f"{mute_dir}/2a_f0/mute.wav.npy|{mute_dir}/2b-f0nsf/mute.wav.npy|{speaker_id}"
            )
        else:
            lines.append(
                f"{mute_dir}/0_gt_wavs/mute{sr_label}.wav|"
                f"{mute_dir}/3_feature{fea_dim}/mute.npy|{speaker_id}"
            )
    random.shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines))

    # Config path selection matches RVC's own rule: v2 at 40k reuses the v1
    # 40k config (no separate v2/40k.json ships with the repo).
    if version == "v1" or sr_label == "40k":
        config_src = RVC_DIR / "configs" / "v1" / f"{sr_label}.json"
    else:
        config_src = RVC_DIR / "configs" / "v2" / f"{sr_label}.json"
    config_data = json.loads(config_src.read_text())
    (exp_dir / "config.json").write_text(json.dumps(config_data, indent=4, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainset-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["experiment_name"]
    version = cfg["version"]
    sr_label = cfg["sample_rate"]
    sr_hz = sample_rate_hz(sr_label)
    if_f0 = 1 if cfg["f0"] else 0
    is_half = "False"  # fp32 throughout: simpler and safer for a one-off hackathon run

    exp_dir = RVC_DIR / "logs" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    trainset_dir = args.trainset_dir.resolve()
    if not any(trainset_dir.glob("*.wav")):
        print(f"No wav files found in {trainset_dir} -- run data/prep/extract_audio.py first.", file=sys.stderr)
        return 1

    print(f"== Step 1/5: preprocess (slice + resample + normalize) -> {exp_dir}")
    run([
        sys.executable, "-m", "train.preprocess",
        str(trainset_dir), str(sr_hz), str(cfg["n_cpu"]), str(exp_dir),
        "False",  # noparallel
        "3.7",    # per: max slice length in seconds, RVC's own default
    ])

    print("== Step 2/5: extract f0 (pitch) on GPU")
    run([
        sys.executable, "-m", "train.dataset.extract_f0",
        "cuda", "1", "0", cfg["gpu"], str(exp_dir), is_half,
    ])

    print("== Step 3/5: extract HuBERT content features on GPU")
    run([
        sys.executable, "-m", "train.dataset.extract_hubert_feature",
        "cuda:0", "1", "0", cfg["gpu"], str(exp_dir), version, is_half,
    ])

    print("== Step 4/5: build filelist.txt + config.json")
    write_filelist_and_config(exp_dir, cfg)

    print("== Step 5/5: train")
    path_str = "" if version == "v1" else "_v2"
    f0_str = "f0" if if_f0 else ""
    pretrained_g = RVC_DIR / "assets" / f"pretrained{path_str}" / f"{f0_str}G{sr_label}.pth"
    pretrained_d = RVC_DIR / "assets" / f"pretrained{path_str}" / f"{f0_str}D{sr_label}.pth"
    train_cmd = [
        sys.executable, "-m", "train.train",
        "-e", exp_name,
        "-sr", sr_label,
        "-f0", str(if_f0),
        "-bs", str(cfg["batch_size"]),
        "-g", cfg["gpu"],
        "-te", str(cfg["total_epoch"]),
        "-se", str(cfg["save_every_epoch"]),
        "-l", "1" if cfg["if_latest"] else "0",
        "-c", "1" if cfg["cache_data_in_gpu"] else "0",
        "-sw", "1" if cfg["save_every_weights"] else "0",
        "-v", version,
    ]
    if pretrained_g.is_file():
        train_cmd += ["-pg", str(pretrained_g)]
    else:
        print(f"WARNING: no pretrained generator at {pretrained_g}, training from scratch.", file=sys.stderr)
    if pretrained_d.is_file():
        train_cmd += ["-pd", str(pretrained_d)]
    run(train_cmd)

    print("== Building retrieval index")
    outside_index_root = RVC_DIR / "assets" / "indices"
    run([
        sys.executable, "-m", "train.train_index",
        exp_name, version, str(outside_index_root), str(cfg["n_cpu"]),
    ])

    weights_dir = RVC_DIR / "assets" / "weights"
    print(f"\nDone. Trained model(s) should now be in {weights_dir} "
          f"and the retrieval index in {exp_dir} / {outside_index_root}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
