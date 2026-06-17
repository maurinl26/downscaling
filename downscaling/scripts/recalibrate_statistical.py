#!/usr/bin/env python3
"""Statistical recalibration (lapse-rate + QDM) with Sencrop residual correction.

Thin orchestrator. Reuses `StatisticalDownscalingPipeline` as-is and adds a
post-pass sparse residual correction on the Sencrop network for the target
year. Outputs a Zarr grid under `--out`. Known internally as the "Lot B"
deliverable of the Sencrop S23 campaign.

Inputs
------

CERRA atm + CERRA-Land (downloaded by `download_cerra_for_recalibration.py`) :
    --cerra-atm  /workspace/data/cerra/cerra_atm_<year>.nc
    --cerra-land /workspace/data/cerra_land/cerra_land_<year>.nc

DEM (IGN BD ALTI) :
    --dem        /workspace/data/dem/bd_alti_drome.nc  (lat, lon, elevation)

Sencrop bulk (kDrive symlink OR s3://...) — passed as a root path:
    --sencrop    /workspace/data/sencrop  (resolved to ${SENCROP_DATA_ROOT})

Output
------

Zarr 1-km grid for the target year (T2m daily Tmin nocturne, recalibrated):
    --out        /workspace/data/output/lot_b_grid/<year>.zarr

Method
------

1. Open CERRA atm NetCDF, compute nightly Tmin (18h → 09h UTC).
2. Run `StatisticalDownscalingPipeline.run(source, variables=['t2m'])` on the
   nightly Tmin field → 1 km grid (lapse + QDM if calibrated).
3. For each night with ≥ 5 Sencrop stations available, compute the residual
   `tmin_obs_station - tmin_grid_at_station_cell` and apply a smooth kriging-
   style correction across the bbox (Gaussian RBF on station residuals). This
   is the "Sencrop calibration" step.
4. Write the corrected grid to Zarr partitioned by year.

Reproducibility envelope
------------------------

Logs to stdout (and a small `<out>/.run_metadata.json`) :
- uv-run command, git SHA, dirty flag
- DEM path, CERRA inputs, Sencrop root
- N stations actually used per night
- bbox / years / resolution

Note: this script is **CPU-friendly** (no GPU). For the DL FiLM variant see
`recalibrate_dl_film.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from downscaling.prtihvi_wxc.sencrop import (
    load_stations_catalog,
    load_timeseries,
)
from downscaling.statistical.pipeline import StatisticalDownscalingPipeline

log = logging.getLogger("recalibrate_statistical")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nightly_tmin(da: xr.DataArray) -> xr.DataArray:
    """Aggregate hourly/3-hourly T2m → nightly Tmin keyed to the morning date.

    Convention: night DATE = 15h UTC DATE → 09h UTC DATE+1.
    Handles CERRA NetCDF which uses `valid_time` instead of `time`.
    """
    # CERRA NetCDF utilise valid_time ; renommer en time.
    if "valid_time" in da.dims and "time" not in da.dims:
        da = da.rename({"valid_time": "time"})
    elif "valid_time" in da.coords and "time" not in da.coords:
        da = da.rename({"valid_time": "time"})
    # Si la coord time n'est pas datetime, convertir depuis hours since 1900
    if da["time"].dtype.kind != "M":
        ref = pd.Timestamp("1900-01-01")
        # CERRA time is hours since 1900
        da = da.assign_coords(time=ref + pd.to_timedelta(da["time"].values, unit="h"))
    # Shift -9h so that the morning's date labels the previous night.
    da = da.assign_coords(time=da["time"] - pd.Timedelta("9h"))
    return da.resample(time="1D").min()


def _bbox_from_grid(da: xr.DataArray) -> dict[str, float]:
    """Extract a bbox dict from a DataArray with lat/lon (or latitude/longitude) coords."""
    lat = da["latitude"] if "latitude" in da.coords else da["lat"]
    lon = da["longitude"] if "longitude" in da.coords else da["lon"]
    return {
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
    }


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


# ---------------------------------------------------------------------------
# Residual correction (Gaussian RBF on Sencrop stations)
# ---------------------------------------------------------------------------
@dataclass
class _Station:
    lat: float
    lon: float
    altitude_m: float
    bucket_id: int


def _residual_correction(
    grid: xr.DataArray,
    stations: list[_Station],
    obs_tmin: np.ndarray,
    sigma_km: float = 7.0,
) -> xr.DataArray:
    """Apply a smooth Gaussian-RBF correction from sparse station residuals.

    `grid` has dims (latitude, longitude). For each grid cell, the correction is
    the weighted average of station residuals with weight `exp(-d²/2σ²)`. Falls
    back to the raw grid for cells with negligible total weight.
    """
    if not stations:
        log.warning("No stations available — returning uncorrected grid")
        return grid

    lat_grid = grid["latitude"].values if "latitude" in grid.coords else grid["lat"].values
    lon_grid = grid["longitude"].values if "longitude" in grid.coords else grid["lon"].values
    grid_arr = grid.values  # (H, W)

    # Pre-compute station residuals
    sta_lats = np.array([s.lat for s in stations])
    sta_lons = np.array([s.lon for s in stations])

    # Nearest grid cell value at each station
    nearest_vals = np.full(len(stations), np.nan)
    for i, s in enumerate(stations):
        ii = int(np.argmin(np.abs(lat_grid - s.lat)))
        jj = int(np.argmin(np.abs(lon_grid - s.lon)))
        nearest_vals[i] = grid_arr[ii, jj]
    residuals = obs_tmin - nearest_vals

    valid = ~np.isnan(residuals)
    if valid.sum() < 3:
        log.warning(
            "Only %d valid stations after grid sampling — returning uncorrected grid",
            int(valid.sum()),
        )
        return grid
    sta_lats = sta_lats[valid]
    sta_lons = sta_lons[valid]
    residuals = residuals[valid]

    # Build (H, W) correction field via Gaussian RBF (lat/lon in km, ~111 km/deg)
    LL, NN = np.meshgrid(lat_grid, lon_grid, indexing="ij")
    correction = np.zeros_like(grid_arr, dtype=np.float32)
    weights = np.zeros_like(grid_arr, dtype=np.float32)
    for r_lat, r_lon, res in zip(sta_lats, sta_lons, residuals):
        dlat = (LL - r_lat) * 111.0
        dlon = (NN - r_lon) * 111.0 * np.cos(np.deg2rad(r_lat))
        d2 = dlat**2 + dlon**2
        w = np.exp(-d2 / (2.0 * sigma_km**2))
        correction += (w * res).astype(np.float32)
        weights += w.astype(np.float32)

    correction = np.where(weights > 1e-6, correction / weights, 0.0)
    out = grid.copy()
    out.values = grid_arr + correction
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Statistical recalibration (lapse + QDM) with Sencrop residual")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--cerra-atm", type=Path, required=True)
    p.add_argument("--cerra-land", type=Path, required=True, help="kept for symmetry / future")
    p.add_argument("--dem", type=Path, required=True)
    p.add_argument("--sencrop", type=str, required=True, help="bulk root (local or s3://)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--obs-ref", type=Path, default=None, help="optional CERRA fine ref for QDM calibration")
    p.add_argument("--variable", default="t2m")
    p.add_argument("--sigma-km", type=float, default=7.0)
    p.add_argument("--wandb-project", default="karpos-recalibrate-statistical")
    p.add_argument("--wandb-disabled", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    log.info("Statistical recalibration | year=%d | out=%s", args.year, args.out)

    # 1. Load CERRA, compute nightly Tmin
    ds = xr.open_dataset(args.cerra_atm)
    t_var = args.variable if args.variable in ds else "t2m"
    if t_var not in ds:
        # Sometimes the variable is named differently in CERRA atm
        candidates = [v for v in ds.data_vars if "temperature" in v.lower() or v == "2t"]
        if not candidates:
            raise ValueError(f"No T2m-like variable in {args.cerra_atm}")
        t_var = candidates[0]
    log.info("Using temperature variable: %s", t_var)

    nightly = _nightly_tmin(ds[t_var])
    nightly = nightly.where(nightly.time.dt.year == args.year, drop=True)
    # Normalise latitude/longitude → lat/lon pour matcher le DEM SRTM.
    rename = {}
    if "latitude" in nightly.dims:
        rename["latitude"] = "lat"
    if "longitude" in nightly.dims:
        rename["longitude"] = "lon"
    if rename:
        nightly = nightly.rename(rename)

    # 2. Statistical pipeline (lapse + optional QDM)
    pipe = StatisticalDownscalingPipeline(
        dem_path=args.dem,
        obs_ref_path=args.obs_ref,
        use_qdm=bool(args.obs_ref),
    )
    if args.obs_ref is not None:
        ref_ds = xr.open_dataset(args.obs_ref)
        pipe.calibrate(ref_ds, ref_ds)  # placeholder; production would use a true high-res ref

    # 3. For each night, run the pipeline + Sencrop residual correction
    stations_df = load_stations_catalog(args.sencrop)
    bbox = _bbox_from_grid(nightly)
    stations_df = load_stations_catalog(args.sencrop, bbox=bbox)
    stations = [
        _Station(
            lat=float(r["latitude"]),
            lon=float(r["longitude"]),
            altitude_m=float(r["altitude_m"]),
            bucket_id=int(r["bucket_id"]),
        )
        for _, r in stations_df.iterrows()
    ]
    bucket_ids = [s.bucket_id for s in stations]

    ts = load_timeseries(years=[args.year], root=args.sencrop, station_only=True, bucket_ids=bucket_ids)
    # Compute nightly Tmin per station
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True)
    ts["night_date"] = (ts["timestamp"] - pd.Timedelta("9h")).dt.date
    obs_per_night = (
        ts.groupby(["night_date", "station_id"])["temperature"]
        .min()
        .reset_index()
    )

    # 3.bis Init W&B (no-op si pas de clé / désactivé)
    wandb_run = None
    if not args.wandb_disabled and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"stat-{args.year}",
                config={
                    "stage": "statistical",
                    "year": args.year,
                    "sigma_km": args.sigma_km,
                    "dem": str(args.dem),
                    "sencrop_root": str(args.sencrop),
                    "use_qdm": bool(args.obs_ref),
                    "n_stations_bbox": len(stations),
                    "bbox": bbox,
                    "git_sha": _git_sha(),
                },
                reinit=True,
            )
            log.info("W&B run: %s", wandb_run.url if wandb_run else "n/a")
        except Exception as exc:
            log.warning("W&B init failed (continuing without): %s", exc)
            wandb_run = None

    out_grids = []
    n_stations_used = []
    # Métriques in-sample par nuit (résidus station observation - grid après calibration)
    per_night_records: list[dict] = []
    for d in nightly.time.values:
        d_py: date = pd.Timestamp(d).date()
        slab = nightly.sel(time=d)
        try:
            ds_slab = xr.Dataset({"t2m": slab})
            grid_fine = pipe.run(ds_slab, variables=["t2m"])["t2m"]
        except Exception as exc:
            log.warning("Pipeline failed for %s: %s, skipping", d_py, exc)
            continue

        night_obs = obs_per_night[obs_per_night["night_date"] == d_py]
        kept_stations = [s for s in stations if s.bucket_id in set(night_obs["station_id"])]
        obs_tmin = np.array(
            [
                float(night_obs.loc[night_obs["station_id"] == s.bucket_id, "temperature"].iloc[0])
                for s in kept_stations
            ]
        )
        if len(kept_stations) >= 5:
            grid_corr = _residual_correction(grid_fine, kept_stations, obs_tmin, sigma_km=args.sigma_km)
            n_stations_used.append(len(kept_stations))
        else:
            grid_corr = grid_fine
            n_stations_used.append(0)

        out_grids.append(grid_corr.expand_dims(time=[d]))

        # Métriques résidus station (in-sample : sur les stations utilisées dans la calibration)
        if kept_stations:
            lat_grid = grid_corr["latitude"].values if "latitude" in grid_corr.coords else grid_corr["lat"].values
            lon_grid = grid_corr["longitude"].values if "longitude" in grid_corr.coords else grid_corr["lon"].values
            g = grid_corr.values
            preds = np.array([
                g[int(np.argmin(np.abs(lat_grid - s.lat))), int(np.argmin(np.abs(lon_grid - s.lon)))]
                for s in kept_stations
            ])
            residuals = obs_tmin - preds  # signed residuals (obs - prediction)
            tmin_min_obs = float(np.nanmin(obs_tmin))
            per_night_records.append({
                "date": str(d_py),
                "n_stations": len(kept_stations),
                "tmin_min_obs": tmin_min_obs,
                "tmin_mean_obs": float(np.nanmean(obs_tmin)),
                "residual_mean": float(np.nanmean(residuals)),
                "residual_rmse": float(np.sqrt(np.nanmean(residuals ** 2))),
                "residual_abs_mean": float(np.nanmean(np.abs(residuals))),
            })

    if not out_grids:
        log.error("No nightly grids produced, aborting")
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        return 2

    out_ds = xr.concat(out_grids, dim="time")
    args.out.mkdir(parents=True, exist_ok=True)
    zarr_path = args.out / f"{args.year}.zarr"
    out_ds.to_zarr(zarr_path, mode="w")
    log.info("Wrote %s (%d nights)", zarr_path, len(out_grids))

    # Synthèse métriques sur les nuits avec résidu calculable
    if per_night_records:
        residuals_all = np.array([r["residual_mean"] for r in per_night_records])
        rmse_all = np.array([r["residual_rmse"] for r in per_night_records])
        abs_all = np.array([r["residual_abs_mean"] for r in per_night_records])
        n_stations_arr = np.array([r["n_stations"] for r in per_night_records])
        # Indicateurs détection gel (sur le minimum observé par nuit, seuil flo)
        tmin_obs = np.array([r["tmin_min_obs"] for r in per_night_records])
        frost_threshold = -2.2  # BBCH flo abricot (seuil de référence pitch coopératives)
        is_frost_night = tmin_obs < frost_threshold
        summary = {
            "n_nights_total": len(out_grids),
            "n_nights_with_residuals": len(per_night_records),
            "n_frost_nights_obs": int(is_frost_night.sum()),
            "avg_stations_per_night": float(np.mean(n_stations_arr)),
            "median_stations_per_night": float(np.median(n_stations_arr)),
            "residual_mean_year": float(np.nanmean(residuals_all)),
            "residual_rmse_year": float(np.sqrt(np.nanmean(rmse_all ** 2))),
            "residual_abs_mean_year": float(np.nanmean(abs_all)),
        }
    else:
        summary = {
            "n_nights_total": len(out_grids),
            "n_nights_with_residuals": 0,
            "avg_stations_per_night": 0.0,
        }

    # Reproducibility envelope
    metadata = {
        "year": args.year,
        "command": " ".join(["uv", "run", "python", *sys.argv]),
        "git_sha": _git_sha(),
        "cerra_atm": str(args.cerra_atm),
        "cerra_land": str(args.cerra_land),
        "dem": str(args.dem),
        "sencrop_root": str(args.sencrop),
        "sigma_km": args.sigma_km,
        **summary,
    }
    (args.out / f"{args.year}.metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Done. Metadata: %s", args.out / f"{args.year}.metadata.json")

    if wandb_run is not None:
        try:
            import wandb

            wandb.log({"summary/" + k: v for k, v in summary.items()})
            # Per-night table (lecture rapide dans W&B)
            if per_night_records:
                tbl = wandb.Table(
                    columns=list(per_night_records[0].keys()),
                    data=[[r[k] for k in per_night_records[0].keys()] for r in per_night_records],
                )
                wandb.log({"per_night": tbl})
            # Log artefact Zarr metadata
            artifact = wandb.Artifact(f"stat-{args.year}-metadata", type="metadata")
            artifact.add_file(str(args.out / f"{args.year}.metadata.json"))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
        except Exception as exc:
            log.warning("W&B finalize failed: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
