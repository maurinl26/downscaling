"""Smoke test for the Lot D pod pipeline scripts (downscaling#11).

Goal: with mocked data, verify that each script can :
- be imported (syntax + module-level checks)
- assemble its main objects (datasets, models, pipelines) without exceptions
- produce a Zarr output of the expected shape on a tiny CPU-friendly run

Each test is wrapped in `pytest.importorskip` or skipped on missing deps so
the smoke test still passes in lightweight CI environments.

Run :
    cd downscaling
    uv run pytest tests/test_lot_d_smoke.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_bbox() -> dict[str, float]:
    return {"lat_min": 44.0, "lat_max": 45.5, "lon_min": 4.0, "lon_max": 5.5}


@pytest.fixture
def mock_sencrop_root(tmp_path: Path) -> Path:
    """Create a tiny bulk Sencrop root: stations_integrated.csv + 2022.csv/part-*.csv."""
    root = tmp_path / "sencrop"
    root.mkdir()

    # 5 stations
    stations = pd.DataFrame(
        {
            "device_id": [38133, 51334, 22748, 7174, 7169],
            "serial": ["A", "B", "C", "D", "E"],
            "bucket_id": [22971, 24529, 14711, 10429, 9130],
            "city": ["Plaisians", "Plaisians", "Mirabel", "Buis", "Cécile"],
            "latitude": [44.23, 44.23, 44.32, 44.29, 44.22],
            "longitude": [5.29, 5.32, 5.09, 5.25, 4.90],
            "altitude_m": [585.0, 649.0, 279.0, 621.0, 154.0],
            "activation_date": ["2021-03-17"] * 5,
            "is_public": [True] * 5,
        }
    )
    stations.to_csv(root / "stations_integrated.csv", index=False)

    # 2022 partition — Spark-style directory with one part CSV
    year_dir = root / "2022.csv"
    year_dir.mkdir()
    timestamps = pd.date_range("2022-02-15", "2022-02-17", freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    rows = []
    for ts in timestamps:
        for bid in stations["bucket_id"]:
            rows.append(
                {
                    "station_id": int(bid),
                    "timestamp": ts.isoformat(),
                    "temperature": float(rng.normal(-1.5, 2.0)),
                    "temperature_source": "station",
                    "humidity": float(rng.uniform(60, 95)),
                    "humidity_source": "station",
                }
            )
    pd.DataFrame(rows).to_csv(
        year_dir / "part-00000-mock.csv", index=False
    )
    return root


@pytest.fixture
def mock_cerra(tmp_path: Path, mock_bbox: dict) -> Path:
    """Tiny CERRA atm NetCDF — 3 nights, 4x4 grid."""
    times = pd.date_range("2022-02-15", "2022-02-17", freq="6h")
    lats = np.linspace(mock_bbox["lat_min"], mock_bbox["lat_max"], 4)
    lons = np.linspace(mock_bbox["lon_min"], mock_bbox["lon_max"], 4)
    t = np.full((len(times), len(lats), len(lons)), -2.0, dtype=np.float32)
    ds = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), t)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    path = tmp_path / "cerra_atm_2022.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def mock_dem(tmp_path: Path, mock_bbox: dict) -> Path:
    """Tiny DEM NetCDF — 8x8 grid (higher res than CERRA)."""
    lats = np.linspace(mock_bbox["lat_min"], mock_bbox["lat_max"], 8)
    lons = np.linspace(mock_bbox["lon_min"], mock_bbox["lon_max"], 8)
    elev = np.full((len(lats), len(lons)), 400.0, dtype=np.float32)
    ds = xr.Dataset(
        {"elevation": (("latitude", "longitude"), elev)},
        coords={"latitude": lats, "longitude": lons},
    )
    path = tmp_path / "dem.nc"
    ds.to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_sencrop_loader_bulk(mock_sencrop_root: Path) -> None:
    """Patched load_sencrop reads the bulk root and returns StationObs."""
    from downscaling.prtihvi_wxc.sencrop import load_sencrop, _is_bulk_root

    assert _is_bulk_root(mock_sencrop_root)
    obs = load_sencrop(mock_sencrop_root, date="2022-02-15")
    assert obs.station_id.size == 5, "expected 5 mock stations"
    assert obs.lat.shape == (5,)
    assert obs.t_raw.shape[0] == 5


def test_download_cerra_imports() -> None:
    """The download entrypoint module imports without side effects."""
    import importlib
    mod = importlib.import_module("downscaling.scripts.download_cerra_for_recalibration")
    assert hasattr(mod, "main")
    assert hasattr(mod, "REQUIRED_ENV")


def test_recalibrate_statistical_imports() -> None:
    import importlib
    mod = importlib.import_module("downscaling.scripts.recalibrate_statistical")
    assert hasattr(mod, "main")
    # _residual_correction should be present
    assert hasattr(mod, "_residual_correction")


def test_recalibrate_dl_film_imports() -> None:
    """DL FiLM script imports torch + lightning, skip if unavailable."""
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    import importlib
    mod = importlib.import_module("downscaling.scripts.recalibrate_dl_film")
    assert hasattr(mod, "main")
    assert hasattr(mod, "BulkSencropDataset")


def test_recalibrate_statistical_smoke_run(
    tmp_path: Path,
    mock_cerra: Path,
    mock_dem: Path,
    mock_sencrop_root: Path,
) -> None:
    """End-to-end statistical recalibration on mock data, produces a Zarr."""
    pytest.importorskip("sklearn")  # statistical pipeline pulls scikit-learn
    out = tmp_path / "recalibrate_statistical_out"
    sys_argv = sys.argv[:]
    sys.argv = [
        "recalibrate_statistical.py",
        "--year", "2022",
        "--cerra-atm",  str(mock_cerra),
        "--cerra-land", str(mock_cerra),  # placeholder, same file
        "--dem",        str(mock_dem),
        "--sencrop",    str(mock_sencrop_root),
        "--out",        str(out),
    ]
    try:
        from downscaling.scripts.recalibrate_statistical import main
        rc = main()
    finally:
        sys.argv = sys_argv

    assert rc == 0, "recalibrate_statistical.main() returned non-zero on smoke data"
    assert (out / "2022.zarr").exists() or (out / "2022.zarr").is_dir(), \
        "expected Zarr output"


def test_recalibrate_dl_film_dataset_build(
    mock_cerra: Path,
    mock_dem: Path,
    mock_sencrop_root: Path,
) -> None:
    """Build the BulkSencropDataset on mock data, verify it has samples."""
    pytest.importorskip("torch")
    from downscaling.scripts.recalibrate_dl_film import BulkSencropDataset, _build_coarse_provider

    provider, ds_cerra, ds_dem = _build_coarse_provider(mock_cerra, mock_dem)
    lat = ds_dem["latitude"].values
    lon = ds_dem["longitude"].values
    dem = ds_dem["elevation"].values

    dataset = BulkSencropDataset(
        dates=["2022-02-15", "2022-02-16"],
        coarse_provider=provider,
        sencrop_root=mock_sencrop_root,
        lat_grid=lat,
        lon_grid=lon,
        elevation_grid=dem,
        min_stations=3,
    )
    assert len(dataset) >= 1, "expected at least 1 night in mock"
    sample = dataset[0]
    assert "x_met" in sample and "x_dem" in sample and "obs_tmin" in sample


def test_pod_entrypoint_shellcheck() -> None:
    """The pod orchestrator bash script has valid syntax (bash -n)."""
    import subprocess

    script = Path(__file__).resolve().parent.parent / "scripts" / "recalibration_pipeline.sh"
    if not script.exists():
        pytest.skip(f"{script} not present")
    rc = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True
    ).returncode
    assert rc == 0, "bash -n failed on the entrypoint script"
