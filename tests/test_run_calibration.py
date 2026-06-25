"""Tests de l'entry point ``run-calibration`` (chemin B).

Compose la config (groupe ``calibration``) sans torch, et couvre les helpers
testables sans données : extraction des grilles depuis le MNT et chargement des
poids U-Net (Lightning .ckpt / torch .pt).
"""

from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from downscaling.paths import CONFIG_DIR

# ---------------------------------------------------------------------------
# Composition de la config (sans torch)
# ---------------------------------------------------------------------------


def test_calibration_group_composes():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")
    cal = cfg.calibration
    assert cal.target_channel == 0
    assert cal.reduce == "min"
    assert cal.elevation_aware is True
    # surcharge en dotlist
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg2 = compose(config_name="config", overrides=["calibration.reduce=mean"])
    assert cfg2.calibration.reduce == "mean"


# ---------------------------------------------------------------------------
# Helpers (torch requis pour _load_unet_weights)
# ---------------------------------------------------------------------------


def test_grids_from_dem_1d_coords():
    pytest.importorskip("torch")
    import xarray as xr

    from downscaling.scripts.run_calibration import grids_from_dem

    lat = np.linspace(44.0, 45.0, 5)
    lon = np.linspace(4.0, 5.0, 6)
    dem = xr.Dataset(
        {"elevation": (("lat", "lon"), np.zeros((5, 6)))},
        coords={"lat": lat, "lon": lon},
    )
    lat_grid, lon_grid, elev = grids_from_dem(dem)
    np.testing.assert_allclose(lat_grid, lat)
    np.testing.assert_allclose(lon_grid, lon)
    assert elev.shape == (5, 6)


def test_grids_from_dem_fallback_to_indices():
    pytest.importorskip("torch")
    import xarray as xr

    from downscaling.scripts.run_calibration import grids_from_dem

    dem = xr.Dataset({"elevation": (("y", "x"), np.zeros((3, 4)))})
    lat_grid, lon_grid, _ = grids_from_dem(dem)
    np.testing.assert_array_equal(lat_grid, np.arange(3))
    np.testing.assert_array_equal(lon_grid, np.arange(4))


def test_load_unet_weights_lightning_and_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from downscaling.scripts.run_calibration import _load_unet_weights

    model = torch.nn.Conv2d(2, 2, 1)
    ref = {k: torch.randn_like(v) for k, v in model.state_dict().items()}

    # Format Lightning : clés préfixées `model.` + bruit non-modèle ignoré.
    ckpt = tmp_path / "lit.ckpt"
    torch.save(
        {
            "state_dict": {
                **{f"model.{k}": v for k, v in ref.items()},
                "criterion.x": torch.zeros(1),
            }
        },
        ckpt,
    )
    _load_unet_weights(model, ckpt)
    for k, v in ref.items():
        assert torch.equal(model.state_dict()[k], v)

    # Format torch manuel : model_state_dict direct.
    pt = tmp_path / "manual.pt"
    ref2 = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    torch.save({"model_state_dict": ref2}, pt)
    _load_unet_weights(model, pt)
    for k, v in ref2.items():
        assert torch.equal(model.state_dict()[k], v)
