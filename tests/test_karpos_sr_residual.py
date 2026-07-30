"""Smoke tests de la cible RÉSIDU KarposSR v2 (first-guess + résidu) — sans GPU.

Valide : (1) validation de ``target_mode`` ; (2) invariant résidu
``pred_residual = pred_raw + first_guess`` ; (3) plancher physique (résidu → 0
⇒ pred = first-guess) ; (4) le Dataset émis contient bien ``t2m`` ET
``t2m_prerbf`` (branchement RBF Sencrop du harness --loo).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
pytest.importorskip("lightning.pytorch")

import torch.nn as nn  # noqa: E402

from downscaling.deep_learning.model import build_model  # noqa: E402
from downscaling.deep_learning.sparse_calibration import (  # noqa: E402
    UNetSparseCalibrationModule,
)
from downscaling.scripts.recalibrate_dl_film import _dl_output_dataset  # noqa: E402

MET_CH, DEM_CH, SIZE = 5, 4, 16


def _batch():
    return {
        "x_met": torch.randn(1, MET_CH, SIZE, SIZE),
        "x_dem": torch.randn(1, DEM_CH, SIZE, SIZE),
    }


def _module(model, target_mode):
    return UNetSparseCalibrationModule(
        model, kelvin_to_celsius=False, elevation_aware=False, target_mode=target_mode
    )


def test_target_mode_validation():
    with pytest.raises(ValueError):
        _module(build_model(architecture="srcnn", met_in_ch=MET_CH, dem_in_ch=DEM_CH), "bogus")


def test_residual_equals_raw_plus_first_guess():
    """pred(residual) == pred(raw) + first_guess (= canal météo cible)."""
    model = build_model(architecture="srcnn", met_in_ch=MET_CH, dem_in_ch=DEM_CH)
    batch = _batch()
    with torch.no_grad():
        pred_raw = _module(model, "raw")._predict_target(batch)
        pred_res = _module(model, "residual")._predict_target(batch)
    first_guess = batch["x_met"][:, 0:1]  # target_channel = 0
    assert pred_res.shape == pred_raw.shape == first_guess.shape
    assert torch.allclose(pred_res, pred_raw + first_guess, atol=1e-5)


class _ZeroModel(nn.Module):
    """Modèle qui sort toujours 0 → teste le plancher physique du résidu."""

    def forward(self, x_met, x_dem, cond_vec=None):
        b, _, h, w = x_met.shape
        return torch.zeros(b, 1, h, w)


def test_residual_floor_is_first_guess():
    """Résidu → 0 ⇒ pred = first-guess (plancher physique, jamais pire que CERRA 1 km)."""
    batch = _batch()
    with torch.no_grad():
        pred = _module(_ZeroModel(), "residual")._predict_target(batch)
    assert torch.allclose(pred, batch["x_met"][:, 0:1], atol=1e-6)


def test_dl_output_dataset_has_t2m_and_prerbf():
    da = xr.DataArray(
        np.zeros((2, 4, 3), dtype="float32"),
        dims=("time", "latitude", "longitude"),
        name="t2m",
    )
    ds = _dl_output_dataset(da)
    assert set(ds.data_vars) == {"t2m", "t2m_prerbf"}
    # backbone identique (le DL n'applique pas de RBF)
    assert np.array_equal(ds["t2m"].values, ds["t2m_prerbf"].values)


# --- v2b : first-guess lapse -------------------------------------------------


def test_first_guess_lapse_requires_dz():
    """first_guess='lapse' sans dz → erreur claire."""
    with pytest.raises(ValueError):
        UNetSparseCalibrationModule(
            _ZeroModel(),
            kelvin_to_celsius=False,
            elevation_aware=False,
            target_mode="residual",
            first_guess="lapse",
            first_guess_dz=None,
        )


def test_first_guess_lapse_numerically_exact():
    """first_guess lapse = x_met + lapse_rate·dz (plancher, résidu ZeroModel → 0)."""
    dz = torch.randn(SIZE, SIZE)
    m = UNetSparseCalibrationModule(
        _ZeroModel(),
        kelvin_to_celsius=False,
        elevation_aware=False,
        target_mode="residual",
        first_guess="lapse",
        first_guess_dz=dz,
    )
    batch = _batch()
    with torch.no_grad():
        pred = m._predict_target(batch)
    expected = batch["x_met"][:, 0:1] + m.lapse_rate * dz
    assert pred.shape == expected.shape
    assert torch.allclose(pred, expected, atol=1e-5)


def test_first_guess_lapse_differs_from_bilinear():
    """Le first-guess lapse doit corriger le bilinéaire par le terme d'altitude."""
    dz = torch.full((SIZE, SIZE), 500.0)  # 500 m de dénivelé maille↔coarse
    m = UNetSparseCalibrationModule(
        _ZeroModel(),
        kelvin_to_celsius=False,
        elevation_aware=False,
        target_mode="residual",
        first_guess="lapse",
        first_guess_dz=dz,
    )
    batch = _batch()
    with torch.no_grad():
        pred = m._predict_target(batch)
    bilinear = batch["x_met"][:, 0:1]
    # 500 m × -6.5e-3 K/m ≈ -3.25 °C de correction, non nul.
    assert not torch.allclose(pred, bilinear, atol=1e-3)
    assert torch.allclose((pred - bilinear).mean(), torch.tensor(m.lapse_rate * 500.0), atol=1e-4)
