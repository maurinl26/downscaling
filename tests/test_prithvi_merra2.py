"""Test du pipeline d'entrée MERRA-2 réel (issue #1) — sans données ni poids.

Prouve la chaîne complète contre l'API officielle : échantillon brut (forme
``Merra2Dataset.__getitem__``) → ``preproc`` officielle (via ``merra2_collate``)
→ ``batch`` modèle → forward d'un **vrai** ``PrithviWxC`` (config jouet).

Sauté sans le package ``PrithviWxC``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PrithviWxC")

from downscaling.prtihvi_wxc.merra2_input import ZERO_PADDING, merra2_collate

# Réutilise le backbone jouet prouvé (C=4, C_STATIC=2, T=2, grille 6×8).
from tests.test_prithvi_real import C_STATIC, H_LR, W_LR, C, T, _tiny_backbone

# in_channels=4 = n_surface(2) + n_vertical(1) × n_levels(2)
N_SUR, N_VERT, N_LEV = 2, 1, 2


def _raw_sample() -> dict:
    """Échantillon brut au format ``Merra2Dataset.__getitem__``."""
    gen = torch.Generator().manual_seed(0)
    H, W = H_LR, W_LR
    return {
        "sur_vals": torch.randn(N_SUR, T, H, W, generator=gen),  # (param, time, lat, lon)
        "sur_tars": torch.randn(N_SUR, 1, H, W, generator=gen),  # cible : 1 pas de temps
        "ulv_vals": torch.randn(
            N_VERT, N_LEV, T, H, W, generator=gen
        ),  # (param, level, time, lat, lon)
        "ulv_tars": torch.randn(N_VERT, N_LEV, 1, H, W, generator=gen),
        "sur_static": torch.randn(C_STATIC, H, W, generator=gen),  # absolute pos enc → C_STATIC
        "sur_climate": torch.randn(N_SUR, H, W, generator=gen),
        "ulv_climate": torch.randn(N_VERT, N_LEV, H, W, generator=gen),
        "lead_time": 6,
        "input_time": -6,
    }


def test_preproc_builds_model_batch():
    """La collate officielle assemble x/y/static/climate aux bonnes formes."""
    batch = merra2_collate(ZERO_PADDING)([_raw_sample()])
    assert {"x", "y", "static", "climate", "lead_time", "input_time"} <= set(batch)
    assert batch["x"].shape == (1, T, C, H_LR, W_LR)  # surface + vertical = 160 réels
    assert batch["y"].shape == (1, C, H_LR, W_LR)
    assert batch["static"].shape == (1, C_STATIC, H_LR, W_LR)
    assert batch["climate"].shape == (1, C, H_LR, W_LR)


def test_full_pipeline_raw_to_forecast():
    """Échantillon brut → preproc → forward du vrai PrithviWxC → prévision."""
    backbone = _tiny_backbone()
    batch = merra2_collate(ZERO_PADDING)([_raw_sample()])
    with torch.no_grad():
        out = backbone(batch)
    assert tuple(out.shape) == (1, C, H_LR, W_LR)
