"""
DL inference on Mac Mini (Apple Silicon MPS).

Suitable for: QC Netatmo, re-scoring, single-night inference, interactive testing.
NOT suitable for: full-campaign training or Prithvi fine-tuning (use RunPod for those).

Usage (console entry point installé via `uv sync`) :
    uv run run-on-mac --task smoke-test
    uv run run-on-mac --task netatmo-qc
    uv run run-on-mac --task unet-inference --night 2021-04-27
    uv run run-on-mac --task interactive

Device selection (automatic):
    MPS  — Apple Silicon (M1/M2/M3 Mac Mini), batch_size 1-2, ~16-32 GB unified memory
    CPU  — fallback if MPS not available
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Racine du dépôt (downscaling/scripts/run_on_mac.py → parents[2]), exposée dans
# la session interactive. La config se charge via Hydra (downscaling.config).
ROOT = Path(__file__).resolve().parents[2]


def _get_device():
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: torch not installed. Run: uv sync --extra dl")

    if torch.backends.mps.is_available():
        print("Device: MPS (Apple Silicon)")
        return torch.device("mps")
    print("Device: CPU (MPS not available)")
    return torch.device("cpu")


def smoke_test(device) -> None:
    import torch

    x = torch.randn(2, 4, 32, 32, device=device)
    print(f"smoke-test: tensor {x.shape} on {device}  ✓")


def netatmo_qc(device) -> None:
    """Run Netatmo quality-control re-scoring for the full archive."""
    from downscaling.config import load_config

    load_config()  # valide la composition Hydra avant le run réel
    print(f"netatmo-qc: device={device}")
    print("  → (stub) load Netatmo obs, score with U-Net, write QC flags")
    # TODO: implement with downscaling.deep_learning.inference.UNetInference


def unet_inference(night: str, device) -> None:
    """Run U-Net inference for a single night (fast, ~30s on MPS)."""
    from downscaling.config import load_config

    load_config()  # valide la composition Hydra avant le run réel
    print(f"unet-inference: night={night}  device={device}")
    print("  → (stub) load CERRA-Land, run U-Net FiLM, write T_min zarr")
    # TODO: implement with downscaling.deep_learning.inference.run_night()


def interactive(device) -> None:
    """Drop into an interactive Python session with device ready."""
    import code

    import torch

    banner = (
        f"\nInteractive DL session — device={device}\n"
        "  import torch; import numpy as np\n"
        "  ROOT is in scope ; config via downscaling.config.load_config()\n"
    )
    code.interact(banner=banner, local={"device": device, "torch": torch, "ROOT": ROOT})


TASKS = {
    "smoke-test": "Quick tensor allocation test on MPS/CPU",
    "netatmo-qc": "Netatmo QC re-scoring over full archive",
    "unet-inference": "U-Net FiLM inference for a single night (--night YYYY-MM-DD)",
    "interactive": "Interactive Python session with MPS device ready",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="DL downscaling — Apple MPS inference")
    parser.add_argument("--task", choices=list(TASKS), required=True, help="Task to run")
    parser.add_argument("--night", metavar="YYYY-MM-DD", help="Target night for unet-inference")
    args = parser.parse_args()

    device = _get_device()

    if args.task == "smoke-test":
        smoke_test(device)
    elif args.task == "netatmo-qc":
        netatmo_qc(device)
    elif args.task == "unet-inference":
        if not args.night:
            sys.exit("ERROR: --night YYYY-MM-DD required for unet-inference")
        unet_inference(args.night, device)
    elif args.task == "interactive":
        interactive(device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
