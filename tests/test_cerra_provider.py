"""Tests du coarse_provider CERRA (chemin B) — NetCDF synthétique, sans réseau.

Valide que ``CERRACoarseProvider`` produit des entrées U-Net ``(x_met, x_dem)``
normalisées, réduit correctement la nuit (Tmin par maille), expose les dates, et
se branche sur ``UNetStationDataset``.

Sauté sans l'extra ``dl`` (torch).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("netCDF4")

import xarray as xr

from downscaling.deep_learning.cerra_provider import CERRACoarseProvider

MET_VARS = ["t2m", "u10", "v10"]
H, W = 8, 10
DATE = "2021-04-27"


def _write_cerra(path):
    """CERRA jouet : champs horaires sur la nuit 20h → 07h, grille fine."""
    times = pd.date_range(f"{DATE} 20:00", "2021-04-28 07:00", freq="h")
    gen = np.random.default_rng(0)
    data = {
        v: (("time", "y", "x"), gen.normal(280, 3, (len(times), H, W)).astype("float32"))
        for v in MET_VARS
    }
    xr.Dataset(data, coords={"time": times, "y": np.arange(H), "x": np.arange(W)}).to_netcdf(path)


def _write_dem(path):
    gen = np.random.default_rng(1)
    dem = {
        v: (("y", "x"), gen.normal(0, 1, (H, W)).astype("float32"))
        for v in ("elevation", "slope", "aspect", "curvature")
    }
    xr.Dataset(dem, coords={"y": np.arange(H), "x": np.arange(W)}).to_netcdf(path)


def _make_provider(tmp_path, **kwargs):
    cerra_dir = tmp_path / "cerra"
    cerra_dir.mkdir(exist_ok=True)
    _write_cerra(cerra_dir / f"cerra_{DATE}.nc")
    dem_file = tmp_path / "dem.nc"
    _write_dem(dem_file)
    return CERRACoarseProvider(cerra_dir, dem_file, met_vars=MET_VARS, **kwargs)


@pytest.fixture
def provider(tmp_path):
    # Mode réduit (réduction coarse avant U-Net).
    return _make_provider(tmp_path, reduce="min", hourly=False)


def test_provider_hourly_returns_time_stack(tmp_path):
    """hourly=True : pile horaire (T, C, H, W) — descente puis min côté module."""
    prov = _make_provider(tmp_path, hourly=True)
    x_met, x_dem = prov(DATE)
    assert x_met.ndim == 4
    assert x_met.shape[0] == 12  # nuit 20h → 07h incluse = 12 heures
    assert x_met.shape[1:] == (len(MET_VARS), H, W)
    assert x_dem.shape == (4, H, W)


def test_provider_returns_normalised_shapes(provider):
    x_met, x_dem = provider(DATE)
    assert x_met.shape == (len(MET_VARS), H, W)
    assert x_dem.shape == (4, H, W)  # elevation/slope/aspect/curvature
    assert torch.isfinite(x_met).all() and torch.isfinite(x_dem).all()


def test_night_min_reduction(provider):
    """Le canal t2m = minimum nocturne par maille (reduce='min')."""
    ds = xr.open_dataset(provider.path(DATE))
    expected_min = ds["t2m"].min("time").values  # pas de normalisation (stats vides)
    x_met, _ = provider(DATE)
    np.testing.assert_allclose(x_met[0].numpy(), expected_min, rtol=1e-5)


def test_dates_discovery(provider):
    assert provider.dates() == [DATE]


def test_plugs_into_unet_station_dataset(provider, tmp_path):
    """coarse_provider branché sur UNetStationDataset → sample complet."""
    from downscaling.deep_learning.sparse_calibration import UNetStationDataset
    from tests.test_stations import _sencrop_csv  # réutilise l'export Sencrop jouet

    obs_dir = tmp_path / "sencrop"
    obs_dir.mkdir()
    # Place un export Sencrop pour la nuit, nommé selon le template attendu.
    src = _sencrop_csv(tmp_path)
    (obs_dir / f"sencrop_{DATE}.csv").write_text(src.read_text())

    lat_grid = np.linspace(44.6, 44.7, H)
    lon_grid = np.linspace(4.9, 5.0, W)
    ds = UNetStationDataset(
        [DATE],
        provider,
        obs_dir,
        lat_grid,
        lon_grid,
        min_stations=1,
    )
    assert len(ds) == 1
    sample = ds[0]
    assert sample["x_met"].shape == (len(MET_VARS), H, W)
    assert sample["x_dem"].shape == (4, H, W)
    assert sample["obs_tmin"].ndim == 1 and sample["obs_tmin"].shape == sample["obs_dz"].shape


# ---------------------------------------------------------------------------
# Robustesse aux vraies données CERRA
# ---------------------------------------------------------------------------


def _write_named_cerra(path, var_names, coords="yx", ny=H, nx=W):
    """CERRA jouet avec noms de variables / coordonnées paramétrables."""
    times = pd.date_range(f"{DATE} 20:00", "2021-04-28 07:00", freq="h")
    gen = np.random.default_rng(2)
    if coords == "cf":  # latitude/longitude/valid_time (style Copernicus)
        dims = ("valid_time", "latitude", "longitude")
        cc = {
            "valid_time": times,
            "latitude": np.linspace(44, 45, ny),
            "longitude": np.linspace(4, 5, nx),
        }
    elif coords == "latlon":
        dims = ("time", "lat", "lon")
        cc = {"time": times, "lat": np.linspace(44, 45, ny), "lon": np.linspace(4, 5, nx)}
    else:
        dims = ("time", "y", "x")
        cc = {"time": times, "y": np.arange(ny), "x": np.arange(nx)}
    data = {
        v: (dims, gen.normal(280, 3, (len(times), ny, nx)).astype("float32")) for v in var_names
    }
    xr.Dataset(data, coords=cc).to_netcdf(path)


def test_var_map_renames_cerra_names(tmp_path):
    """var_map ramène les noms CERRA (ex: 2t) vers le canonique (t2m)."""
    cerra = tmp_path / "cerra"
    cerra.mkdir()
    _write_named_cerra(cerra / f"cerra_{DATE}.nc", ["2t", "10u", "10v"])
    _write_dem(tmp_path / "dem.nc")
    prov = CERRACoarseProvider(
        cerra,
        tmp_path / "dem.nc",
        met_vars=["t2m", "u10", "v10"],
        var_map={"t2m": "2t", "u10": "10u", "v10": "10v"},
        hourly=False,
    )
    assert prov.inspect(DATE)["met_missing"] == []
    x_met, _ = prov(DATE)
    assert x_met.shape == (3, H, W)


def test_missing_var_raises_listing_available(tmp_path):
    """Variable absente (non mappée) → KeyError listant les vars disponibles."""
    cerra = tmp_path / "cerra"
    cerra.mkdir()
    _write_named_cerra(cerra / f"cerra_{DATE}.nc", ["2t"])
    _write_dem(tmp_path / "dem.nc")
    prov = CERRACoarseProvider(cerra, tmp_path / "dem.nc", met_vars=["t2m"], hourly=False)
    with pytest.raises(KeyError, match="2t"):
        prov(DATE)


def test_cf_coords_harmonized(tmp_path):
    """latitude/longitude/valid_time (Copernicus) → harmonisés en lat/lon/time."""
    cerra = tmp_path / "cerra"
    cerra.mkdir()
    _write_named_cerra(cerra / f"cerra_{DATE}.nc", MET_VARS, coords="cf")
    _write_dem(tmp_path / "dem.nc")
    prov = CERRACoarseProvider(cerra, tmp_path / "dem.nc", met_vars=MET_VARS, hourly=True)
    x_met, _ = prov(DATE)
    assert x_met.shape == (12, len(MET_VARS), H, W)


def test_inspect_reports_grid_and_vars(provider):
    info = provider.inspect(DATE)
    assert info["met_missing"] == []
    assert info["n_time"] == 12
    assert info["field_grid"] == info["dem_grid"] == (H, W)
    assert info["grid_matches_dem"] is True


def _write_dem_latlon(path):
    gen = np.random.default_rng(1)
    dem = {
        v: (("lat", "lon"), gen.normal(0, 1, (H, W)).astype("float32"))
        for v in ("elevation", "slope", "aspect", "curvature")
    }
    xr.Dataset(dem, coords={"lat": np.linspace(44, 45, H), "lon": np.linspace(4, 5, W)}).to_netcdf(
        path
    )


def test_regrid_to_dem_grid(tmp_path):
    """regrid=True : CERRA coarse (4×5) rééchantillonné sur la grille MNT (8×10)."""
    cerra = tmp_path / "cerra"
    cerra.mkdir()
    _write_named_cerra(cerra / f"cerra_{DATE}.nc", MET_VARS, coords="latlon", ny=4, nx=5)
    _write_dem_latlon(tmp_path / "dem.nc")
    prov = CERRACoarseProvider(
        cerra,
        tmp_path / "dem.nc",
        met_vars=MET_VARS,
        regrid=True,
        hourly=True,
    )
    x_met, x_dem = prov(DATE)
    assert x_met.shape == (12, len(MET_VARS), H, W)  # rééchantillonné sur le MNT
    assert x_dem.shape == (4, H, W)
