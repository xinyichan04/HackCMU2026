"""Sanity-check that torch sees a CUDA GPU and can actually run on it.

Run this once per environment (rvc-train in WSL, rvc-live on native Windows)
after setup, and again after any environment change. Catches "torch
installed but silently on CPU" early, since that would otherwise show up
later as unexplained latency in the streaming pipeline.
"""

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not importable in this environment.")
        return 1

    print(f"torch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False. Running on CPU only.")
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA available: True")
    print(f"Device 0: {device_name}")

    # Trivial on-GPU matmul to confirm the device actually executes work,
    # not just that CUDA initialized.
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print(f"Matmul sanity check OK, result shape {tuple(c.shape)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
