#!/usr/bin/env python
"""release_to_hub.py — package and publish a reference checkpoint to HuggingFace Hub.

End-to-end pipeline for releasing a karpos-downscaling checkpoint as a public
reference artefact:

1. Pull the source checkpoint (local path or ``s3://`` URI).
2. Strip Lightning-specific keys and the ``model.`` prefix.
3. Save as ``weights.safetensors`` alongside a ``config.json`` derived from
   the training Hydra config and a ``README.md`` model card from the
   Obsidian vault.
4. Upload the bundle to the HuggingFace Hub.

Usage
-----
::

    uv run python downscaling/scripts/release_to_hub.py \\
        --checkpoint s3://karpos-downscaling/artifacts/runXXX/last.ckpt \\
        --release baronnies-v1 \\
        --hf-repo karpos/karpos-downscaling-baronnies-v1 \\
        --model-card ~/kDrive/obsidian_vault/karpos-downscaling-hf-model-card.md \\
        --config-template ~/kDrive/obsidian_vault/karpos-downscaling-hf-config.json

Pass ``--no-upload`` to stop after the local bundle is built (useful for inspection).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

log = logging.getLogger("release_to_hub")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _pull_checkpoint(source: str, dest_dir: Path) -> Path:
    """Pull the checkpoint locally if it is on S3, otherwise return as-is.

    Uses a direct ``GetObject`` call against the Scaleway endpoint —
    simpler and lower-privilege than the prefix-listing pull in
    ``s3_sync.py``.
    """
    if not source.startswith("s3://"):
        return Path(source).expanduser().resolve()

    bucket, _, key = source[len("s3://") :].partition("/")
    if not bucket or not key:
        raise SystemExit(f"Malformed S3 URI: {source!r}")
    local = dest_dir / Path(key).name
    log.info("Pulling %s → %s", source, local)

    import boto3

    access = os.environ.get("SCW_ACCESS_KEY")
    secret = os.environ.get("SCW_SECRET_KEY")
    if not (access and secret):
        raise SystemExit(
            "SCW_ACCESS_KEY / SCW_SECRET_KEY missing in environment."
        )
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("SCW_S3_ENDPOINT", "https://s3.fr-par.scw.cloud"),
        region_name=os.environ.get("SCW_S3_REGION", "fr-par"),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=Config(s3={"addressing_style": "virtual"}, signature_version="s3v4"),
    )
    # Stream GetObject directly into the local file (skips HeadObject pre-check
    # which fails with 403 on some Scaleway buckets even when GetObject is allowed).
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    with open(local, "wb") as out:
        for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
            out.write(chunk)
    log.info("Downloaded %.1f MiB", local.stat().st_size / 1024**2)
    return local


def _extract_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load and normalise a Lightning or plain PyTorch checkpoint."""
    log.info("Loading checkpoint %s", ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    if not isinstance(state, dict):
        raise SystemExit(
            f"Unsupported checkpoint format: expected dict, got {type(state).__name__}"
        )

    normalised: dict[str, torch.Tensor] = {}
    skipped_non_tensor = 0
    skipped_optimizer = 0
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            skipped_non_tensor += 1
            continue
        if any(token in key for token in ("optimizer", "loss_scaler", "scheduler")):
            skipped_optimizer += 1
            continue
        clean_key = key
        for prefix in ("model.", "module.", "_orig_mod."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
                break
        normalised[clean_key] = value.detach().cpu()

    if not normalised:
        raise SystemExit("Empty state_dict after normalisation — checkpoint format unsupported.")
    log.info(
        "Normalised state_dict: %d tensors, skipped %d non-tensor / %d optimizer entries",
        len(normalised),
        skipped_non_tensor,
        skipped_optimizer,
    )
    return normalised


def _build_config(template_path: Path, state_dict: dict[str, torch.Tensor]) -> dict:
    """Read the JSON template and inject runtime-known fields (parameter count)."""
    config = json.loads(template_path.read_text())
    n_params = sum(t.numel() for t in state_dict.values())
    config.setdefault("model_meta", {})
    config["model_meta"]["n_parameters"] = int(n_params)
    config["model_meta"]["n_tensors"] = int(len(state_dict))
    log.info("Model: %d parameters across %d tensors", n_params, len(state_dict))
    return config


def _assemble_bundle(
    state_dict: dict[str, torch.Tensor],
    config: dict,
    model_card_src: Path,
    bundle_dir: Path,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bundle_dir / "weights.safetensors"
    save_file(state_dict, weights_path)
    log.info("Wrote %s (%.1f MiB)", weights_path, weights_path.stat().st_size / 1024**2)

    config_path = bundle_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    log.info("Wrote %s", config_path)

    readme_path = bundle_dir / "README.md"
    shutil.copyfile(model_card_src, readme_path)
    log.info("Wrote %s (copied from %s)", readme_path, model_card_src)


def _upload(bundle_dir: Path, hf_repo: str, commit_message: str) -> None:
    log.info("Uploading bundle %s → %s", bundle_dir, hf_repo)
    from huggingface_hub import upload_folder

    upload_folder(
        folder_path=str(bundle_dir),
        repo_id=hf_repo,
        repo_type="model",
        commit_message=commit_message,
    )
    log.info("Upload complete — see https://huggingface.co/%s", hf_repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Source checkpoint path (local or s3://)")
    parser.add_argument("--release", required=True, help="Release name, e.g. baronnies-v1")
    parser.add_argument("--hf-repo", required=True, help="Target HF repo id, e.g. karpos/karpos-downscaling-baronnies-v1")
    parser.add_argument(
        "--model-card",
        default=os.path.expanduser("~/kDrive/obsidian_vault/karpos-downscaling-hf-model-card.md"),
        help="Path to the model-card README to upload",
    )
    parser.add_argument(
        "--config-template",
        default=os.path.expanduser("~/kDrive/obsidian_vault/karpos-downscaling-hf-config.json"),
        help="Path to the JSON template used as base config",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Output directory for the assembled bundle. Default: artifacts/<release>/",
    )
    parser.add_argument("--no-upload", action="store_true", help="Build bundle but skip the HF upload step")
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message for the HF upload. Default: 'Release <release>'",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    model_card = Path(args.model_card).expanduser()
    config_template = Path(args.config_template).expanduser()
    if not model_card.exists():
        raise SystemExit(f"Model-card file not found: {model_card}")
    if not config_template.exists():
        raise SystemExit(f"Config-template file not found: {config_template}")

    bundle_dir = Path(args.bundle_dir or f"artifacts/{args.release}").expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="karpos-release-") as tmp:
        tmp_path = Path(tmp)
        ckpt_path = _pull_checkpoint(args.checkpoint, tmp_path)
        state_dict = _extract_state_dict(ckpt_path)

    config = _build_config(config_template, state_dict)
    _assemble_bundle(state_dict, config, model_card, bundle_dir)

    if args.no_upload:
        log.info("--no-upload set; bundle assembled at %s and not pushed.", bundle_dir)
        return

    commit_message = args.commit_message or f"Release {args.release}"
    _upload(bundle_dir, args.hf_repo, commit_message)


if __name__ == "__main__":
    main()
