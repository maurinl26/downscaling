"""HuggingFace Hub integration for karpos-downscaling.

Provides convenience loaders for the reference checkpoints released on
https://huggingface.co/karpos.

Reference checkpoints are MVP-grade artefacts published for paper
reproduction and academic benchmarking; they are not the production
weights used in the Karpos commercial service.

Examples
--------
>>> from downscaling.hub import load_baronnies_v1
>>> model = load_baronnies_v1(device="cuda")
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import torch
from huggingface_hub import hf_hub_download

from downscaling.deep_learning.model import build_model

KARPOS_HF_ORG = "karpos26"

# Registry of released checkpoints.
# Add a new entry here when a region transitions from commercial-active
# to public release (cf. release strategy in the model card on the Hub).
RELEASES: Mapping[str, Mapping[str, str]] = {
    "baronnies-v1": {
        "repo_id": f"{KARPOS_HF_ORG}/karpos-downscaling-baronnies",
        "config_file": "config.json",
        "weights_file": "weights.safetensors",
    },
}


def _resolve_release(release: str) -> Mapping[str, str]:
    if release not in RELEASES:
        available = ", ".join(sorted(RELEASES))
        raise ValueError(f"Unknown release {release!r}. Available releases: {available}.")
    return RELEASES[release]


def _load_config(repo_id: str, filename: str) -> dict:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(path) as handle:
        return json.load(handle)


def _load_state_dict(repo_id: str, filename: str) -> dict:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    if filename.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_from_hub(
    release: str,
    *,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> torch.nn.Module:
    """Load a karpos-downscaling reference checkpoint from HuggingFace Hub.

    Parameters
    ----------
    release : str
        Name of the released checkpoint. See ``RELEASES`` for the full list.
        Currently available: ``'baronnies-v1'``.
    device : str or torch.device, default 'cpu'
        Device on which to place the model.
    eval_mode : bool, default True
        Whether to switch the model to eval mode (dropout disabled,
        normalisation statistics frozen).

    Returns
    -------
    torch.nn.Module
        The instantiated model with loaded weights, ready for inference.

    Notes
    -----
    The released weights are MVP-grade reference checkpoints intended for
    paper reproduction and academic benchmarking. They are **not**
    calibrated for direct operational frost detection — see the model
    card on HuggingFace Hub for limitations and intended use.

    Examples
    --------
    >>> model = load_from_hub("baronnies-v1", device="cuda")
    >>> # expects input shape (B, met_in_ch + dem_in_ch, H, W)
    """
    spec = _resolve_release(release)
    config = _load_config(spec["repo_id"], spec["config_file"])
    state_dict = _load_state_dict(spec["repo_id"], spec["weights_file"])

    model = build_model(
        architecture=config.get("architecture", "unet"),
        met_in_ch=config["met_in_ch"],
        dem_in_ch=config["dem_in_ch"],
        base_ch=config.get("base_ch", 64),
        n_levels=config.get("n_levels", 4),
        use_film=config.get("use_film", True),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    if eval_mode:
        model.eval()
    return model


def load_baronnies_v1(
    *,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> torch.nn.Module:
    """Shorthand for :func:`load_from_hub` with ``release='baronnies-v1'``.

    See :func:`load_from_hub` for full documentation.
    """
    return load_from_hub("baronnies-v1", device=device, eval_mode=eval_mode)


__all__ = ["RELEASES", "load_from_hub", "load_baronnies_v1"]
