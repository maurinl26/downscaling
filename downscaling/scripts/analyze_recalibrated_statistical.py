#!/usr/bin/env python3
"""Analyse post-hoc des sorties Stage 2 (recalibrated_statistical) avec W&B.

Permet de calculer rigoureusement les métriques de calibration Sencrop
SANS interrompre / relancer le pipeline. Utile quand le pipeline d'origine
n'avait pas l'instrumentation W&B (cas runs antérieurs sur le pod
karpos-recalibration, 2026-06-16).

Pour chaque année dont le Zarr existe sous `--root`, le script :
1. Ouvre `<year>.zarr` (sortie recalibrate_statistical).
2. Recharge les stations Sencrop dans la bbox du Zarr.
3. Pour chaque nuit, extrait la valeur du grid à chaque station.
4. Calcule le résidu obs - prediction (in-sample, après calibration).
5. Synthétise indicateurs (RMSE / abs mean) à l'année.
6. Calcule indices de détection gel par seuil (POD / FAR / CSI) :
   - Seuil par défaut −2,2 °C (BBCH flo abricot, cohérent avec le pitch).
7. Logge tout sur W&B (un run par année, ou un run agrégé multi-années).

Usage
-----

    python -m downscaling.scripts.analyze_recalibrated_statistical \\
        --root /workspace/data/output/recalibrated_statistical \\
        --sencrop s3://karpos-backtest-data/sencrop \\
        --threshold-c -2.2 \\
        --wandb-project karpos-recalibrate-statistical

W&B
---

Le script utilise les mêmes conventions que `recalibrate_statistical.py` :
`WANDB_API_KEY` doit être présent dans l'env. `--wandb-disabled` saute
W&B et imprime le résumé sur stdout / metadata.json à côté de chaque Zarr.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from downscaling.prtihvi_wxc.sencrop import (
    load_stations_catalog,
    load_timeseries,
)

log = logging.getLogger("analyze_stat")


def _bbox_from_grid(da: xr.DataArray | xr.Dataset) -> dict[str, float]:
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lat = da[lat_name]
    lon = da[lon_name]
    return {
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
    }


def _contingency(obs: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float]:
    """POD / FAR / CSI / bias_score pour un seuil de gel donné.

    Convention :
    - obs < threshold → frost observed
    - pred < threshold → frost predicted
    - POD = TP / (TP + FN), FAR = FP / (FP + TP), CSI = TP / (TP + FP + FN)
    """
    obs_frost = obs < threshold
    pred_frost = pred < threshold
    TP = int(np.sum(obs_frost & pred_frost))
    FP = int(np.sum(~obs_frost & pred_frost))
    FN = int(np.sum(obs_frost & ~pred_frost))
    TN = int(np.sum(~obs_frost & ~pred_frost))
    pod = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    far = FP / (FP + TP) if (FP + TP) > 0 else float("nan")
    csi = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else float("nan")
    bias_score = (TP + FP) / (TP + FN) if (TP + FN) > 0 else float("nan")
    return {
        "threshold_c": threshold,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "bias_score": bias_score,
    }


def _analyze_year(year_zarr: Path, sencrop_root: str, threshold_c: float) -> dict:
    """Charge un Zarr annuel, extrait résidus station, calcule métriques."""
    ds = xr.open_zarr(year_zarr)
    var_name = list(ds.data_vars)[0]  # t2m généralement
    da = ds[var_name]
    # Auto-detect Kelvin (CERRA T2m) → convert to Celsius.
    # Détection robuste : (1) attrs["units"] si présent, (2) médiane GLOBALE
    # (toute la grille, toutes les nuits) au lieu de time=0 seul, pour résister à un
    # Zarr bimodal (zones près des stations ramenées en °C par RBF, zones lointaines
    # encore en K). Cf. audit à froid juin 2026, bug #17.
    src_units = str(da.attrs.get("units", "")).lower()
    sample_global = float(np.nanmedian(da.values))
    log.info("Sample value (median global): %.2f, units attr: %r", sample_global, src_units)
    if src_units in ("k", "kelvin") or sample_global > 100:
        log.info("Detected Kelvin units, converting to Celsius")
        da = da - 273.15
    bbox = _bbox_from_grid(da)
    year = int(year_zarr.stem)

    # Tolère time stocké en int (bug Stage 2 antérieur) : synthétise les dates
    # à partir de l'année + fenêtre frost-flo (fév 1 → ...). 90 nuits attendues.
    if da["time"].dtype.kind != "M":
        n = da.sizes.get("time", 0)
        start = pd.Timestamp(f"{year}-02-01")
        synth_times = start + pd.to_timedelta(da["time"].values, unit="D")
        da = da.assign_coords(time=synth_times)
        log.info("Synthesized time coord from int index → %s … %s", synth_times[0], synth_times[-1] if n > 0 else "n/a")

    # Stations & timeseries Sencrop dans la bbox
    stations_df = load_stations_catalog(sencrop_root, bbox=bbox)
    bucket_ids = stations_df["bucket_id"].astype(int).tolist()
    ts = load_timeseries(years=[year], root=sencrop_root, station_only=True, bucket_ids=bucket_ids)
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True)
    ts["night_date"] = (ts["timestamp"] - pd.Timedelta("9h")).dt.date
    obs_per_night = (
        ts.groupby(["night_date", "station_id"])["temperature"]
        .min()
        .reset_index()
    )

    # Grille lat/lon
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lat_grid = da[lat_name].values
    lon_grid = da[lon_name].values

    # Stations lookup (dedup au cas où le bulk a des duplicats par bucket)
    station_meta = (
        stations_df[["bucket_id", "latitude", "longitude"]]
        .drop_duplicates(subset=["bucket_id"], keep="first")
        .set_index("bucket_id")
    )

    per_night_records: list[dict] = []
    all_obs: list[float] = []
    all_pred: list[float] = []
    for d in da["time"].values:
        d_py = pd.Timestamp(d).date()
        slab = da.sel(time=d).values  # (lat, lon)
        nights = obs_per_night[obs_per_night["night_date"] == d_py]
        if nights.empty:
            continue
        rec_obs: list[float] = []
        rec_pred: list[float] = []
        for _, row in nights.iterrows():
            bid = int(row["station_id"])
            if bid not in station_meta.index:
                continue
            slat = float(station_meta.at[bid, "latitude"])
            slon = float(station_meta.at[bid, "longitude"])
            ii = int(np.argmin(np.abs(lat_grid - slat)))
            jj = int(np.argmin(np.abs(lon_grid - slon)))
            obs_val = float(row["temperature"])
            pred_val = float(slab[ii, jj])
            if np.isnan(obs_val) or np.isnan(pred_val):
                continue
            rec_obs.append(obs_val)
            rec_pred.append(pred_val)
            all_obs.append(obs_val)
            all_pred.append(pred_val)
        if not rec_obs:
            continue
        rec_obs_a = np.array(rec_obs)
        rec_pred_a = np.array(rec_pred)
        residuals = rec_obs_a - rec_pred_a
        per_night_records.append({
            "date": str(d_py),
            "n_stations": len(rec_obs),
            "tmin_obs_min": float(np.min(rec_obs_a)),
            "tmin_obs_mean": float(np.mean(rec_obs_a)),
            "tmin_pred_min": float(np.min(rec_pred_a)),
            "residual_mean": float(np.mean(residuals)),
            "residual_rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "residual_abs_mean": float(np.mean(np.abs(residuals))),
        })

    if not per_night_records:
        return {"year": year, "n_nights": 0, "n_pairs": 0}

    arr_obs = np.array(all_obs)
    arr_pred = np.array(all_pred)
    residuals_all = arr_obs - arr_pred

    # Métriques contingence à plusieurs seuils
    contingency = {
        f"thr_{int(thr * 10) / 10}": _contingency(arr_obs, arr_pred, thr)
        for thr in (threshold_c, 0.0, -5.0)
    }

    summary = {
        "year": year,
        "n_nights": len(per_night_records),
        "n_pairs_station_night": int(len(arr_obs)),
        "n_stations_bbox": int(len(stations_df)),
        "residual_mean_year": float(np.mean(residuals_all)),
        "residual_rmse_year": float(np.sqrt(np.mean(residuals_all ** 2))),
        "residual_abs_mean_year": float(np.mean(np.abs(residuals_all))),
        "residual_bias_p10": float(np.percentile(residuals_all, 10)),
        "residual_bias_p90": float(np.percentile(residuals_all, 90)),
        "contingency": contingency,
        "per_night": per_night_records,
        "bbox": bbox,
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Dir avec <year>.zarr")
    p.add_argument("--sencrop", type=str, required=True, help="Sencrop root (local ou s3://)")
    p.add_argument("--threshold-c", type=float, default=-2.2)
    p.add_argument("--wandb-project", default="karpos-recalibrate-statistical")
    p.add_argument("--wandb-disabled", action="store_true")
    p.add_argument("--years", type=int, nargs="*", default=None, help="Subset, défaut: tous les *.zarr trouvés")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    zarrs = sorted(args.root.glob("*.zarr"))
    if args.years:
        zarrs = [z for z in zarrs if int(z.stem) in args.years]
    if not zarrs:
        log.error("No <year>.zarr under %s", args.root)
        return 2
    log.info("Analyzing %d Zarr(s): %s", len(zarrs), [z.name for z in zarrs])

    # W&B init (optionnel)
    wandb_run = None
    if not args.wandb_disabled and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"analyze-stat-{'_'.join([z.stem for z in zarrs])}",
                config={
                    "stage": "statistical_posthoc",
                    "threshold_c": args.threshold_c,
                    "sencrop_root": args.sencrop,
                    "years": [int(z.stem) for z in zarrs],
                },
                reinit=True,
            )
            log.info("W&B run: %s", wandb_run.url)
        except Exception as exc:
            log.warning("W&B init failed: %s", exc)

    all_summaries: list[dict] = []
    for z in zarrs:
        log.info("--- year %s ---", z.stem)
        try:
            s = _analyze_year(z, args.sencrop, args.threshold_c)
        except Exception as exc:
            log.exception("Year %s failed: %s", z.stem, exc)
            continue
        all_summaries.append(s)
        # Write metadata.posthoc.json à côté du Zarr
        out_json = z.parent / f"{z.stem}.posthoc.json"
        out_json.write_text(json.dumps(s, indent=2, default=str))
        log.info("Wrote %s", out_json)
        if wandb_run is not None:
            try:
                import wandb

                # Flat log par année (préfixé)
                yr = s["year"]
                wandb.log({
                    f"year_{yr}/n_nights": s["n_nights"],
                    f"year_{yr}/n_pairs": s["n_pairs_station_night"],
                    f"year_{yr}/residual_rmse": s["residual_rmse_year"],
                    f"year_{yr}/residual_mean": s["residual_mean_year"],
                    f"year_{yr}/residual_abs_mean": s["residual_abs_mean_year"],
                    **{
                        f"year_{yr}/{k}/{m}": v[m]
                        for k, v in s["contingency"].items()
                        for m in ("POD", "FAR", "CSI", "bias_score", "TP", "FP", "FN")
                    },
                })
            except Exception as exc:
                log.warning("W&B log year %s failed: %s", s["year"], exc)

    # Synthèse multi-années (POD/FAR/CSI agrégés)
    valid = [s for s in all_summaries if "contingency" in s and s.get("n_pairs_station_night", 0) > 0]
    if valid:
        # Moyenne pondérée des résiduels
        weights = np.array([s["n_pairs_station_night"] for s in valid], dtype=float)
        rmse_w = float(np.sqrt(np.average(
            [s["residual_rmse_year"] ** 2 for s in valid], weights=weights)))
        bias_w = float(np.average([s["residual_mean_year"] for s in valid], weights=weights))
        abs_w = float(np.average([s["residual_abs_mean_year"] for s in valid], weights=weights))
        global_summary = {
            "n_years": len(valid),
            "total_pairs": int(weights.sum()),
            "weighted_rmse": rmse_w,
            "weighted_bias": bias_w,
            "weighted_abs_mean": abs_w,
        }
        log.info("Global summary: %s", global_summary)
        if wandb_run is not None:
            try:
                import wandb

                wandb.log({f"global/{k}": v for k, v in global_summary.items()})
            except Exception as exc:
                log.warning("W&B log global failed: %s", exc)

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
