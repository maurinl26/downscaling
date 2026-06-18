#!/usr/bin/env python3
"""Classify each frost-flo night into a synoptic regime (rule-based on ERA5).

C5.2 (issue maurinl26/downscaling#TBD). Sortie : un CSV par année avec un label
de régime par nuit + les features synoptiques. Le label est ensuite consommé
par `analyze_recalibrated_statistical.py --regimes-csv` pour stratifier POD/FAR.

Cinq régimes (rule-based, médiane spatiale sur bbox Drôme, fenêtre nuit 18-09 UTC) :

- **R1 Radiatif**         : wind10m < 2.5 m/s · tcc < 0.30 · MSLP > 1015 hPa
                            → gel rayonnement, cas Baronnies typique
- **R2 Advectif N/NE**    : wind10m > 4 m/s · dir ∈ [290°, 50°] · MSLP > 1010 hPa
                            → gel d'advection, masses Sibérie/Norvège
- **R3 Cyclonic**         : wind10m > 5 m/s · tcc > 0.60 · MSLP < 1015 hPa
                            → perturbation, pas de gel attendu
- **R4 Anticyclonic doux** : MSLP > 1020 hPa · wind10m < 3 m/s · tcc > 0.30
                            → blocage, pas radiatif, peu de gel
- **R0 Mixed/Transition** : le reste

Sortie CSV par année :

    date,regime,wind_med,wind_dir_med,tcc_med,mslp_med,dewpoint_dep_med

Usage
-----

    uv run python -m downscaling.scripts.flag_regimes \
      --era5-dir /tmp/karpos_synoptic \
      --years 2022 2023 2024 2025 2026 \
      --bbox-lat 44.0 45.5 --bbox-lon 4.0 5.5 \
      --out /tmp/karpos_synoptic/regimes
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger("flag_regimes")


def _circular_median_dir(directions_deg: np.ndarray) -> float:
    """Median of wind directions (degrees), accounts for circular wrap."""
    finite = directions_deg[np.isfinite(directions_deg)]
    if finite.size == 0:
        return np.nan
    # Vector mean
    rad = np.deg2rad(finite)
    u_mean = np.nanmean(np.sin(rad))
    v_mean = np.nanmean(np.cos(rad))
    dir_mean = np.rad2deg(np.arctan2(u_mean, v_mean)) % 360.0
    return float(dir_mean)


def _normalize_era5(ds: xr.Dataset) -> xr.Dataset:
    """Normalize ERA5 NetCDF : valid_time→time, lat/lon naming, time decoding."""
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if ds["time"].dtype.kind != "M":
        ref = pd.Timestamp("1900-01-01")
        ds = ds.assign_coords(time=ref + pd.to_timedelta(ds["time"].values, unit="h"))
    # lat/lon naming
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds


def _night_window(ds: xr.Dataset, d: date) -> xr.Dataset:
    """Select ERA5 timesteps for the night of date `d` : 18h UTC (d-1) → 09h UTC (d)."""
    start = pd.Timestamp(d) - pd.Timedelta("6h")   # 18h day before
    end = pd.Timestamp(d) + pd.Timedelta("9h")     # 09h current day
    return ds.sel(time=slice(start, end))


def _night_features(ds_night: xr.Dataset, bbox: dict[str, float]) -> dict[str, float]:
    """Compute synoptic features for one night (median over bbox + time window)."""
    sub = ds_night.sel(
        lat=slice(bbox["lat_max"], bbox["lat_min"]),  # ERA5 lat descending
        lon=slice(bbox["lon_min"], bbox["lon_max"]),
    )
    if sub.sizes.get("time", 0) == 0 or sub.sizes.get("lat", 0) == 0:
        return {}

    u10 = sub["u10"].values
    v10 = sub["v10"].values
    wind = np.sqrt(u10**2 + v10**2)
    wind_dir = (np.rad2deg(np.arctan2(u10, v10)) % 360.0)  # 0=N, 90=E

    tcc = sub["tcc"].values  # 0-1
    msl = sub["msl"].values / 100.0  # Pa → hPa
    t2m = sub["t2m"].values
    d2m = sub["d2m"].values
    dewpoint_dep = t2m - d2m  # K (proxy clear-sky)

    return {
        "wind_med": float(np.nanmedian(wind)),
        "wind_dir_med": _circular_median_dir(wind_dir),
        "tcc_med": float(np.nanmedian(tcc)),
        "mslp_med": float(np.nanmedian(msl)),
        "dewpoint_dep_med": float(np.nanmedian(dewpoint_dep)),
    }


def _classify(f: dict[str, float]) -> str:
    """Apply rule-based regime classification to feature dict.

    Taxonomie use-case Karpos (gel arboriculture) :
    cadran 2×2 vent × ciel, plus une bande "intermédiaire" pour les nuits
    ambigües. Optimise l'interprétation produit : "où le modèle marche bien
    et où il rate".

    - **R1 Radiatif**       : vent ≤ 3.0 m/s · tcc ≤ 0.50 → gel rayonnement
                              probable (cas Baronnies typique)
    - **R2 Advectif venté** : vent > 3.0 m/s · tcc ≤ 0.50 → mélange forcé,
                              gel possible par advection
    - **R3 Couvert venté**  : vent > 3.0 m/s · tcc > 0.50 → perturbé,
                              gel rare
    - **R4 Couvert calme**  : vent ≤ 3.0 m/s · tcc > 0.50 → nuageux nocturne,
                              limite le radiatif
    - **R0** : feature manquant (catch-all NaN)
    """
    if not f or any(np.isnan(v) for v in f.values()):
        return "R0"

    wind_calm = f["wind_med"] <= 3.0
    sky_clear = f["tcc_med"] <= 0.50

    if wind_calm and sky_clear:
        return "R1"
    if not wind_calm and sky_clear:
        return "R2"
    if not wind_calm and not sky_clear:
        return "R3"
    return "R4"  # wind_calm and not sky_clear


def _dates_for_year(ds: xr.Dataset, year: int, months: tuple[int, ...]) -> list[date]:
    """Unique morning dates in the requested year/months."""
    times = pd.DatetimeIndex(ds["time"].values)
    mask = (times.year == year) & np.isin(times.month, list(months))
    dates = pd.Index(times[mask].normalize()).unique().sort_values()
    return [d.date() for d in dates]


def main() -> int:
    p = argparse.ArgumentParser(description="Classify frost-flo nights into synoptic regimes")
    p.add_argument("--era5-dir", type=Path, required=True,
                   help="Directory containing era5_synoptic_<year>.nc files")
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--months", type=int, nargs="+", default=[2, 3, 4, 5],
                   help="Months of interest (default: 02 03 04 05 = flo abricot)")
    p.add_argument("--bbox-lat", type=float, nargs=2, default=[44.0, 45.5],
                   help="Latitude bounds (min max), default Drôme = 44.0 45.5")
    p.add_argument("--bbox-lon", type=float, nargs=2, default=[4.0, 5.5],
                   help="Longitude bounds (min max), default Drôme = 4.0 5.5")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (one CSV per year)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    bbox = {
        "lat_min": min(args.bbox_lat),
        "lat_max": max(args.bbox_lat),
        "lon_min": min(args.bbox_lon),
        "lon_max": max(args.bbox_lon),
    }
    log.info("Bbox: %s", bbox)

    summary_per_year: dict[int, dict[str, int]] = {}
    all_rows: list[dict] = []

    for year in args.years:
        path = args.era5_dir / f"era5_synoptic_{year}.nc"
        if not path.exists():
            log.warning("%d: %s manquant, skip", year, path)
            continue
        log.info("--- year %d ---", year)
        ds = _normalize_era5(xr.open_dataset(path))

        dates = _dates_for_year(ds, year, tuple(args.months))
        log.info("  %d nuits à classifier", len(dates))

        rows: list[dict] = []
        regime_counts: dict[str, int] = {"R0": 0, "R1": 0, "R2": 0, "R3": 0, "R4": 0}
        for d in dates:
            ds_night = _night_window(ds, d)
            feats = _night_features(ds_night, bbox)
            regime = _classify(feats)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            row = {"date": d.isoformat(), "regime": regime, **feats}
            rows.append(row)
            all_rows.append({"year": year, **row})

        df = pd.DataFrame(rows)
        out_csv = args.out / f"regimes_{year}.csv"
        df.to_csv(out_csv, index=False)
        log.info("  → %s", out_csv)
        log.info("  régimes : %s", regime_counts)
        summary_per_year[year] = regime_counts

    # Synthèse globale
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        out_all = args.out / "regimes_all.csv"
        df_all.to_csv(out_all, index=False)
        log.info("Synthèse globale : %s (%d nuits)", out_all, len(df_all))

        log.info("Bilan régimes par année :")
        for year, counts in summary_per_year.items():
            total = sum(counts.values())
            line = f"  {year}: total={total:3d}"
            for r in ("R1", "R2", "R3", "R4", "R0"):
                n = counts.get(r, 0)
                pct = (100.0 * n / total) if total else 0.0
                line += f" · {r}={n:3d} ({pct:4.1f}%)"
            log.info(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
