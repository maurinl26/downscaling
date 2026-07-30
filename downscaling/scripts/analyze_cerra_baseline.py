#!/usr/bin/env python3
"""Baseline CERRA brute (sans calibration) — CSI/POD/FAR @ -2,2 °C (issue #89).

Premier maillon du chaînon de *lift* opposable :
    CERRA brute → Lot B (calibration statistique) → Lot C (DL FiLM).

Contrairement aux Lots B/C, la CERRA brute n'utilise **aucune** information de
capteur : pas de RBF, pas de leave-one-out (rien à retirer). On échantillonne la
réanalyse CERRA (5,5 km) à la maille la plus proche de chaque station et on
compare la **Tmin nocturne** au seuil gel −2,2 °C, sur les **mêmes stations /
nuits / vérité terrain Sencrop** que la validation LOO Lot B/C.

Définition de la nuit et vérité terrain : identiques à
``analyze_recalibrated_statistical._analyze_year`` (night_date = (ts − 9 h).date,
Tmin = min des obs de la nuit). CERRA (3-horaire) est agrégée en Tmin nocturne
par la même convention.

Sortie : ``<out-dir>/<year>.loo.json`` (format compatible loo_json/, mode "raw")
et ``<out-dir>/pooled.json`` (agrégat micro-poolé multi-années).

Usage :
    python -m downscaling.scripts.analyze_cerra_baseline \
        --cerra-glob '/…/regen_cerra_2023/cerra_atm_{year}.nc' \
        --sencrop /…/data/raw/sencrop \
        --out-dir docs/methodology/figures/loo_json/cerra_raw
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from downscaling.prtihvi_wxc.sencrop import load_stations_catalog, load_timeseries

log = logging.getLogger("cerra_baseline")


def _contingency(obs: np.ndarray, pred: np.ndarray, threshold: float) -> dict:
    """POD / FAR / CSI / bias_score — convention identique à analyze_recalibrated_statistical."""
    obs_frost = obs < threshold
    pred_frost = pred < threshold
    TP = int(np.sum(obs_frost & pred_frost))
    FP = int(np.sum(~obs_frost & pred_frost))
    FN = int(np.sum(obs_frost & ~pred_frost))
    TN = int(np.sum(~obs_frost & ~pred_frost))
    r = lambda a, b: a / b if b else float("nan")
    return {
        "threshold_c": threshold,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "POD": r(TP, TP + FN), "FAR": r(FP, FP + TP), "CSI": r(TP, TP + FP + FN),
        "bias_score": r(TP + FP, TP + FN),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _cerra_nightly_min(nc_path: str) -> xr.DataArray:
    """Ouvre la CERRA brute et agrège en Tmin nocturne par maille (°C).

    night_date = (valid_time − 9 h).date, cohérent avec la vérité terrain Sencrop.
    """
    ds = xr.open_dataset(nc_path)
    var = "t2m" if "t2m" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var]
    time_name = "valid_time" if "valid_time" in da.coords else "time"
    # Kelvin → Celsius (CERRA t2m en K).
    units = str(da.attrs.get("units", "")).lower()
    if units in ("k", "kelvin") or float(np.nanmedian(da.values)) > 100:
        da = da - 273.15
    times = pd.to_datetime(da[time_name].values)
    night = (times - pd.Timedelta("9h")).normalize()
    da = da.assign_coords(night=(time_name, night))
    nightly = da.groupby("night").min()
    return nightly  # dims: (night, y/lat, x/lon)


def _analyze_year(nc_path: str, year: int, sencrop_root: str, threshold_c: float) -> dict:
    nightly = _cerra_nightly_min(nc_path)
    lat_name = "latitude" if "latitude" in nightly.coords else "lat"
    lon_name = "longitude" if "longitude" in nightly.coords else "lon"
    lat_grid = np.asarray(nightly[lat_name].values)
    lon_grid = np.asarray(nightly[lon_name].values)
    bbox = {
        "lat_min": float(lat_grid.min()), "lat_max": float(lat_grid.max()),
        "lon_min": float(lon_grid.min()), "lon_max": float(lon_grid.max()),
    }

    stations_df = load_stations_catalog(sencrop_root, bbox=bbox)
    bucket_ids = stations_df["bucket_id"].astype(int).tolist()
    ts = load_timeseries(years=[year], root=sencrop_root, station_only=True, bucket_ids=bucket_ids)
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True)
    ts["night_date"] = (ts["timestamp"] - pd.Timedelta("9h")).dt.date
    obs_per_night = ts.groupby(["night_date", "station_id"])["temperature"].min().reset_index()

    station_meta = (
        stations_df[["bucket_id", "latitude", "longitude"]]
        .drop_duplicates(subset=["bucket_id"], keep="first").set_index("bucket_id")
    )
    # night_date (python date) → index dans la coord "night"
    night_index = {pd.Timestamp(n).date(): i for i, n in enumerate(nightly["night"].values)}
    vals = nightly.values  # (night, lat, lon)

    all_obs: list[float] = []
    all_pred: list[float] = []
    per_station: dict[int, dict] = {}
    for _, row in obs_per_night.iterrows():
        bid = int(row["station_id"])
        nd = row["night_date"]
        if bid not in station_meta.index or nd not in night_index:
            continue
        slat = float(station_meta.at[bid, "latitude"])
        slon = float(station_meta.at[bid, "longitude"])
        ii = int(np.argmin(np.abs(lat_grid - slat)))
        jj = int(np.argmin(np.abs(lon_grid - slon)))
        obs_v = float(row["temperature"])
        pred_v = float(vals[night_index[nd], ii, jj])
        if np.isnan(obs_v) or np.isnan(pred_v):
            continue
        all_obs.append(obs_v)
        all_pred.append(pred_v)
        rec = per_station.setdefault(bid, {"obs": [], "pred": []})
        rec["obs"].append(obs_v)
        rec["pred"].append(pred_v)

    arr_obs = np.array(all_obs)
    arr_pred = np.array(all_pred)
    agg = _contingency(arr_obs, arr_pred, threshold_c) if len(arr_obs) else {}
    station_rows = []
    for bid, rec in per_station.items():
        o = np.array(rec["obs"]); pv = np.array(rec["pred"])
        c = _contingency(o, pv, threshold_c)
        station_rows.append({
            "station_id": int(bid), "n_pairs": int(len(o)),
            "POD": c["POD"], "FAR": c["FAR"], "CSI": c["CSI"],
            "rmse": float(np.sqrt(np.mean((o - pv) ** 2))),
            "bias": float(np.mean(o - pv)),
        })

    return {
        "year": year,
        "threshold_c": threshold_c,
        "source": f"CERRA brute (réanalyse 5,5 km, sans calibration) — {Path(nc_path).name}",
        "calibration": "none",
        "n_stations": int(len(station_meta)),
        "bbox": bbox,
        "modes": {
            "raw": {
                "n_pairs": int(len(arr_obs)),
                "aggregate": {
                    **agg,
                    "rmse": float(np.sqrt(np.mean((arr_obs - arr_pred) ** 2))) if len(arr_obs) else float("nan"),
                    "bias": float(np.mean(arr_obs - arr_pred)) if len(arr_obs) else float("nan"),
                },
                "per_station": station_rows,
            }
        },
        "_obs": all_obs,  # retiré avant écriture ; sert au pooling
        "_pred": all_pred,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cerra-glob", required=True,
                   help="Patron des NetCDF CERRA bruts avec {year}, ex. '.../cerra_atm_{year}.nc'")
    p.add_argument("--sencrop", required=True)
    p.add_argument("--threshold-c", type=float, default=-2.2)
    p.add_argument("--years", type=int, nargs="*", default=[2022, 2023, 2024, 2025])
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    pooled_obs: list[float] = []
    pooled_pred: list[float] = []
    per_year_summary = []
    for year in args.years:
        nc = args.cerra_glob.format(year=year)
        if not Path(nc).exists():
            log.warning("CERRA %s manquant (%s) — skip", year, nc)
            continue
        s = _analyze_year(nc, year, args.sencrop, args.threshold_c)
        pooled_obs += s.pop("_obs")
        pooled_pred += s.pop("_pred")
        s["git_sha"] = sha
        (args.out_dir / f"{year}.loo.json").write_text(json.dumps(s, indent=2, default=str))
        a = s["modes"]["raw"]["aggregate"]
        per_year_summary.append((year, s["modes"]["raw"]["n_pairs"], a.get("POD"), a.get("FAR"), a.get("CSI")))
        log.info("CERRA brute %s : n=%d POD=%.2f FAR=%.2f CSI=%.2f",
                 year, s["modes"]["raw"]["n_pairs"], a.get("POD", float("nan")),
                 a.get("FAR", float("nan")), a.get("CSI", float("nan")))

    arr_o = np.array(pooled_obs); arr_p = np.array(pooled_pred)
    pooled_agg = _contingency(arr_o, arr_p, args.threshold_c) if len(arr_o) else {}
    pooled = {
        "lot": "cerra_raw",
        "description": "Baseline CERRA brute (sans calibration), micro-poolé sur les années. "
                       "Premier maillon du lift CERRA→Lot B→Lot C (issue #89).",
        "threshold_c": args.threshold_c,
        "years": [y for y, *_ in per_year_summary],
        "calibration": "none",
        "n_pairs": int(len(arr_o)),
        "aggregate": {
            **pooled_agg,
            "rmse": float(np.sqrt(np.mean((arr_o - arr_p) ** 2))) if len(arr_o) else float("nan"),
            "bias": float(np.mean(arr_o - arr_p)) if len(arr_o) else float("nan"),
        },
        "per_year": [
            {"year": y, "n_pairs": n, "POD": pod, "FAR": far, "CSI": csi}
            for (y, n, pod, far, csi) in per_year_summary
        ],
        "git_sha": sha,
        "command": " ".join(["python", *sys.argv]),
        "reconciliation_note": (
            "ATTENTION opposabilité : ce baseline échantillonne la CERRA BRUTE 5,5 km "
            "(maille la plus proche, Tmin nocturne). Il ne reproduit PAS le 0,17 annoncé "
            "au manuscrit GMD §5.1 (table 'CERRA brute 5,5 km'). Ce 0,17 correspond en "
            "réalité au FIRST-GUESS descendu bilinéairement à 1 km (+ lapse), c.-à-d. la "
            "variable t2m_prerbf des Zarrs Lot B/C — grille qui réchauffe les cuvettes et "
            "effondre le POD. Cette grille n'est plus disponible localement ni sur S3 "
            "(bucket karpos-backtest-data vide), donc le 0,17 n'est pas re-vérifiable ici "
            "(même blocage que #90). Constat honnête : la CERRA brute nearest-cell est "
            "COLD-BIASED (POD~0,58 / FAR~0,51) → CSI comparable au Lot C mais ~3x plus de "
            "fausses alertes. La valeur de la chaîne est donc la RÉDUCTION du FAR / la "
            "fiabilité, pas un gain de CSI sur une baseline froide. Le framing 'lift "
            "0,17→0,38' exige d'expliciter que 0,17 = first-guess 1 km, sinon challengeable."
        ),
    }
    (args.out_dir / "pooled.json").write_text(json.dumps(pooled, indent=2, default=str))
    log.info("POOLED CERRA brute : n=%d POD=%.3f FAR=%.3f CSI=%.3f",
             pooled["n_pairs"], pooled_agg.get("POD", float("nan")),
             pooled_agg.get("FAR", float("nan")), pooled_agg.get("CSI", float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
