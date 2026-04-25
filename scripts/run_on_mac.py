"""
DL inference on Mac Mini (Apple Silicon MPS).

Suitable for: QC Netatmo, re-scoring, single-night inference, interactive testing.
NOT suitable for: full-campaign training or Prithvi fine-tuning (use RunPod for those).

Usage:
    python downscaling/run_on_mac.py --task smoke-test
    python downscaling/run_on_mac.py --task netatmo-qc
    python downscaling/run_on_mac.py --task unet-inference --night 2021-04-27
    python downscaling/run_on_mac.py --task interactive

Device selection (automatic):
    MPS  — Apple Silicon (M1/M2/M3 Mac Mini), batch_size 1-2, ~16-32 GB unified memory
    CPU  — fallback if MPS not available
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG_DEFAULT = ROOT / "downscaling" / "config" / "drome_ardeche.yml"


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


def netatmo_qc(config: Path, device) -> None:
    """Run Netatmo quality-control re-scoring for the full archive."""
    sys.path.insert(0, str(ROOT / "downscaling"))
    from downscaling.shared.loaders import load_config

    cfg = load_config(config)
    print(f"netatmo-qc: config={config}  device={device}")
    print("  → (stub) load Netatmo obs, score with U-Net, write QC flags")
    # TODO: implement with downscaling.deep_learning.inference.UNetInference


def unet_inference(night: str, config: Path, device) -> None:
    """Run U-Net inference for a single night (fast, ~30s on MPS)."""
    sys.path.insert(0, str(ROOT / "downscaling"))
    from downscaling.shared.loaders import load_config

    cfg = load_config(config)
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
        "  ROOT, CONFIG_DEFAULT are in scope\n"
    )
    code.interact(banner=banner, local={"device": device, "torch": torch,
                                        "ROOT": ROOT, "CONFIG_DEFAULT": CONFIG_DEFAULT})


TASKS = {
    "smoke-test":       "Quick tensor allocation test on MPS/CPU",
    "netatmo-qc":       "Netatmo QC re-scoring over full archive",
    "unet-inference":   "U-Net FiLM inference for a single night (--night YYYY-MM-DD)",
    "interactive":      "Interactive Python session with MPS device ready",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="DL downscaling — Apple MPS inference")
    parser.add_argument("--task", choices=list(TASKS), required=True,
                        help="Task to run")
    parser.add_argument("--night", metavar="YYYY-MM-DD",
                        help="Target night for unet-inference")
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT,
                        metavar="PATH", help="Downscaling config YAML")
    args = parser.parse_args()

    device = _get_device()

    if args.task == "smoke-test":
        smoke_test(device)
    elif args.task == "netatmo-qc":
        netatmo_qc(args.config, device)
    elif args.task == "unet-inference":
        if not args.night:
            sys.exit("ERROR: --night YYYY-MM-DD required for unet-inference")
        unet_inference(args.night, args.config, device)
    elif args.task == "interactive":
        interactive(device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
