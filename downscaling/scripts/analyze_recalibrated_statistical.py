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

    # Local
    python -m downscaling.scripts.analyze_recalibrated_statistical \\
        --root /workspace/data/output/recalibrated_statistical \\
        --sencrop s3://karpos-backtest-data/sencrop \\
        --threshold-c -2.2 \\
        --wandb-project karpos-recalibrate-statistical

    # S3 (Scaleway endpoint via AWS_ENDPOINT_URL ou AWS_S3_ENDPOINT)
    AWS_ENDPOINT_URL=https://s3.fr-par.scw.cloud \\
    python -m downscaling.scripts.analyze_recalibrated_statistical \\
        --root s3://karpos-backtest-data/recalibrated/statistical \\
        --sencrop s3://karpos-backtest-data/sencrop \\
        --years 2022 2023 \\
        --threshold-c -2.2 \\
        --wandb-disabled

W&B
---

Le script utilise les mêmes conventions que `recalibrate_statistical.py` :
`WANDB_API_KEY` doit être présent dans l'env. `--wandb-disabled` saute
W&B et imprime le résumé sur stdout / metadata.json à côté de chaque Zarr.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

from downscaling.prtihvi_wxc.sencrop import (
    load_stations_catalog,
    load_timeseries,
)
from downscaling.scripts.economic_value import relative_economic_value

log = logging.getLogger("analyze_stat")


# ---------------------------------------------------------------------------
# Helpers — root local ou S3 (cf. issue #55, alignement project-data-strategy-runpod-s3)
# ---------------------------------------------------------------------------
def _is_remote(url: str) -> bool:
    """True si ``url`` est une URI non-locale (``s3://``, ``gs://``, etc.)."""
    parsed = urlparse(url)
    return bool(parsed.scheme) and parsed.scheme not in ("", "file")


def _storage_options() -> dict:
    """Storage options pour fsspec / s3fs : endpoint custom (Scaleway) si défini.

    Cohérent avec ``recalibrate_statistical.py:291`` : on lit
    ``AWS_ENDPOINT_URL`` (convention boto3) ou ``AWS_S3_ENDPOINT`` (legacy).
    """
    # skip_instance_cache=True : un S3FileSystem frais par appel, lié au contexte
    # asyncio courant. Évite l'erreur s3fs "Token was created in a different Context"
    # quand on ouvre plusieurs zarr S3 en boucle dans le même process (#33 LOO).
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_S3_ENDPOINT")
    opts: dict = {"skip_instance_cache": True}
    if endpoint:
        opts["client_kwargs"] = {"endpoint_url": endpoint}
    return opts


def _list_zarrs(root: str) -> list[str]:
    """Liste les ``<year>.zarr`` sous ``root`` (local OU s3://).

    Retourne des URIs complètes (préserve le schéma ``s3://``) ou des chemins
    locaux selon la nature de ``root``. Tri ascendant par nom.
    """
    if _is_remote(root):
        fs, _ = fsspec.url_to_fs(root, **_storage_options())
        # fs.glob renvoie sans le schéma → on le rajoute.
        protocol = root.split("://", 1)[0]
        pattern = root.rstrip("/") + "/*.zarr"
        matches = sorted(fs.glob(pattern))
        return [m if "://" in m else f"{protocol}://{m}" for m in matches]
    return [str(p) for p in sorted(Path(root).glob("*.zarr"))]


def _zarr_stem(zarr_url: str) -> str:
    """Year stem depuis une URI ou un path local : '.../2022.zarr' -> '2022'."""
    name = zarr_url.rstrip("/").rsplit("/", 1)[-1]
    return name.removesuffix(".zarr")


def _open_zarr(zarr_url: str) -> xr.Dataset:
    """``xr.open_zarr`` qui marche local ou s3:// (avec endpoint custom)."""
    if _is_remote(zarr_url):
        return xr.open_zarr(zarr_url, storage_options=_storage_options())
    return xr.open_zarr(zarr_url)


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


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Validation hors-station (LOO / leave-one-cluster-out) — issue #33
# ---------------------------------------------------------------------------
def _cluster_groups(station_meta: pd.DataFrame, cluster_km: float) -> dict[int, int]:
    """Regroupe les stations à moins de ``cluster_km`` km en clusters (union-find).

    Deux stations plus proches que ``cluster_km`` portent quasi le même résidu RBF
    (σ=7 km) : les laisser dehors une par une sous-estime l'erreur hors-station
    (cf. les 2 stations Plaisians). Le leave-one-cluster-out exclut le groupe entier.

    Retourne ``{bucket_id: group_id}``. Avec ``cluster_km <= 0`` chaque station est
    son propre groupe (= LOO station-par-station classique).
    """
    bids = [int(b) for b in station_meta.index]
    parent = {b: b for b in bids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if cluster_km > 0:
        for i, bi in enumerate(bids):
            lat_i = float(station_meta.at[bi, "latitude"])
            lon_i = float(station_meta.at[bi, "longitude"])
            for bj in bids[i + 1 :]:
                lat_j = float(station_meta.at[bj, "latitude"])
                lon_j = float(station_meta.at[bj, "longitude"])
                dlat = (lat_i - lat_j) * 111.0
                dlon = (lon_i - lon_j) * 111.0 * np.cos(np.deg2rad(0.5 * (lat_i + lat_j)))
                if np.hypot(dlat, dlon) <= cluster_km:
                    union(bi, bj)
    return {b: find(b) for b in bids}


def _loo_predict(
    da_pre: xr.DataArray,
    station_meta: pd.DataFrame,
    obs_per_night: pd.DataFrame,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    sigma_km: float,
    groups: dict[int, int],
) -> dict[int, dict[str, list[float]]]:
    """Rejoue le RBF résiduel de prod en laissant dehors le groupe de la station cible.

    Reproduit fidèlement ``recalibrate_statistical._residual_correction`` :
    - résidu donneur = ``obs_j - t2m_prerbf(cell_j)`` (grille AVANT RBF)
    - poids gaussien ``exp(-d²/2σ²)`` entre la maille de S et la station donneuse j,
      ``cos`` évalué à la latitude du donneur (comme en prod)
    - garde de prod : correction appliquée seulement si ≥ 5 stations présentes la nuit
      ET ≥ 3 donneurs valides ; sinon la grille servie = ``t2m_prerbf`` (non corrigée).

    Retourne ``{bucket_id: {"obs": [...], "pred": [...]}}`` (paires station-nuit).
    """
    # Maille la plus proche de chaque station (séparable, identique à la prod).
    cell = {
        int(b): (
            int(np.argmin(np.abs(lat_grid - float(station_meta.at[b, "latitude"])))),
            int(np.argmin(np.abs(lon_grid - float(station_meta.at[b, "longitude"])))),
        )
        for b in station_meta.index
    }

    out: dict[int, dict[str, list[float]]] = {}
    for d in da_pre["time"].values:
        d_py = pd.Timestamp(d).date()
        nights = obs_per_night[obs_per_night["night_date"] == d_py]
        if nights.empty:
            continue
        slab = da_pre.sel(time=d).values  # (lat, lon), pré-RBF

        # Stations présentes cette nuit avec obs + prerbf valides.
        present: list[dict] = []
        for _, row in nights.iterrows():
            bid = int(row["station_id"])
            if bid not in cell:
                continue
            ii, jj = cell[bid]
            obs_v = float(row["temperature"])
            pre_v = float(slab[ii, jj])
            if np.isnan(obs_v) or np.isnan(pre_v):
                continue
            present.append(
                {
                    "bid": bid,
                    "obs": obs_v,
                    "pre": pre_v,
                    "lat": float(station_meta.at[bid, "latitude"]),
                    "lon": float(station_meta.at[bid, "longitude"]),
                    "clat": float(lat_grid[ii]),
                    "clon": float(lon_grid[jj]),
                }
            )

        corrected = len(present) >= 5  # garde prod (recalibrate:462)
        for s in present:
            if not corrected:
                pred = s["pre"]
            else:
                gid = groups[s["bid"]]
                donors = [o for o in present if groups[o["bid"]] != gid]
                res = np.array([o["obs"] - o["pre"] for o in donors])
                valid = ~np.isnan(res)
                if valid.sum() < 3:  # garde prod (_residual_correction:173)
                    pred = s["pre"]
                else:
                    dla = np.array([o["lat"] for o in donors])[valid]
                    dlo = np.array([o["lon"] for o in donors])[valid]
                    dlat = (s["clat"] - dla) * 111.0
                    dlon = (s["clon"] - dlo) * 111.0 * np.cos(np.deg2rad(dla))
                    w = np.exp(-(dlat**2 + dlon**2) / (2.0 * sigma_km**2))
                    tot = float(w.sum())
                    corr = float((w * res[valid]).sum() / tot) if tot > 1e-6 else 0.0
                    pred = s["pre"] + corr
            rec = out.setdefault(s["bid"], {"obs": [], "pred": []})
            rec["obs"].append(s["obs"])
            rec["pred"].append(pred)
    return out


def _load_regimes(regimes_csv: Path | None) -> dict[str, str]:
    """Load regime labels keyed by ISO date string. Empty dict if not provided."""
    if regimes_csv is None or not regimes_csv.exists():
        return {}
    df = pd.read_csv(regimes_csv)
    if "date" not in df.columns or "regime" not in df.columns:
        log.warning("--regimes-csv : colonnes 'date' et 'regime' attendues, skip")
        return {}
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return dict(zip(df["date"], df["regime"]))


def _analyze_year(
    year_zarr: str,
    sencrop_root: str,
    threshold_c: float,
    regimes: dict[str, str] | None = None,
) -> dict:
    """Charge un Zarr annuel (local ou s3://), extrait résidus station, calcule métriques."""
    ds = _open_zarr(year_zarr)
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
    year = int(_zarr_stem(year_zarr))

    # Tolère time stocké en int (bug Stage 2 antérieur) : synthétise les dates
    # à partir de l'année + fenêtre frost-flo (fév 1 → ...). 90 nuits attendues.
    if da["time"].dtype.kind != "M":
        n = da.sizes.get("time", 0)
        start = pd.Timestamp(f"{year}-02-01")
        synth_times = start + pd.to_timedelta(da["time"].values, unit="D")
        da = da.assign_coords(time=synth_times)
        log.info(
            "Synthesized time coord from int index → %s … %s",
            synth_times[0],
            synth_times[-1] if n > 0 else "n/a",
        )

    # Stations & timeseries Sencrop dans la bbox
    stations_df = load_stations_catalog(sencrop_root, bbox=bbox)
    bucket_ids = stations_df["bucket_id"].astype(int).tolist()
    ts = load_timeseries(years=[year], root=sencrop_root, station_only=True, bucket_ids=bucket_ids)
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True)
    ts["night_date"] = (ts["timestamp"] - pd.Timedelta("9h")).dt.date
    obs_per_night = ts.groupby(["night_date", "station_id"])["temperature"].min().reset_index()

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
    all_regimes: list[str] = []  # régime synoptique de la nuit, repeated par pair
    regimes = regimes or {}
    for d in da["time"].values:
        d_py = pd.Timestamp(d).date()
        regime_d = regimes.get(d_py.isoformat(), "R?")  # R? si pas de label
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
            all_regimes.append(regime_d)
        if not rec_obs:
            continue
        rec_obs_a = np.array(rec_obs)
        rec_pred_a = np.array(rec_pred)
        residuals = rec_obs_a - rec_pred_a
        per_night_records.append(
            {
                "date": str(d_py),
                "regime": regime_d,
                "n_stations": len(rec_obs),
                "tmin_obs_min": float(np.min(rec_obs_a)),
                "tmin_obs_mean": float(np.mean(rec_obs_a)),
                "tmin_pred_min": float(np.min(rec_pred_a)),
                "residual_mean": float(np.mean(residuals)),
                "residual_rmse": float(np.sqrt(np.mean(residuals**2))),
                "residual_abs_mean": float(np.mean(np.abs(residuals))),
            }
        )

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

    # Stratification par régime synoptique (C5.3). Calcule POD/FAR/CSI au seuil
    # principal seulement, pour chaque régime ayant ≥ 30 paires.
    contingency_by_regime: dict[str, dict] = {}
    arr_regimes = np.array(all_regimes)
    if regimes:
        for r in ("R1", "R2", "R3", "R4", "R4a", "R4b", "R0", "R?"):
            mask = arr_regimes == r
            n = int(mask.sum())
            if n < 30:
                continue
            contingency_by_regime[r] = {
                "n_pairs": n,
                f"thr_{int(threshold_c * 10) / 10}": _contingency(
                    arr_obs[mask], arr_pred[mask], threshold_c
                ),
                "rmse": float(np.sqrt(np.mean((arr_obs[mask] - arr_pred[mask]) ** 2))),
                "bias": float(np.mean(arr_obs[mask] - arr_pred[mask])),
            }

    summary = {
        "year": year,
        "n_nights": len(per_night_records),
        "n_pairs_station_night": int(len(arr_obs)),
        "n_stations_bbox": int(len(stations_df)),
        "residual_mean_year": float(np.mean(residuals_all)),
        "residual_rmse_year": float(np.sqrt(np.mean(residuals_all**2))),
        "residual_abs_mean_year": float(np.mean(np.abs(residuals_all))),
        "residual_bias_p10": float(np.percentile(residuals_all, 10)),
        "residual_bias_p90": float(np.percentile(residuals_all, 90)),
        "contingency": contingency,
        "contingency_by_regime": contingency_by_regime,
        "per_night": per_night_records,
        "bbox": bbox,
    }
    return summary


def _analyze_year_loo(
    year_zarr: str,
    sencrop_root: str,
    threshold_c: float,
    sigma_km: float,
    cluster_km: float,
    station_bbox: tuple[float, float, float, float] | None = None,
) -> dict:
    """Métriques POD/FAR/CSI HORS-STATION (out-of-sample) pour une année.

    Exige la variable ``t2m_prerbf`` dans le Zarr (recalibrate --emit-prerbf).
    Calcule deux jeux : mode ``station`` (LOO classique, #33) et mode ``cluster``
    (leave-one-cluster-out, ``cluster_km``, défendable hors-station réel).
    """
    ds = _open_zarr(year_zarr)
    if "t2m_prerbf" not in ds.data_vars:
        raise ValueError(
            f"{year_zarr}: mode LOO requiert la variable 't2m_prerbf'. "
            "Régénère le Zarr avec `recalibrate_statistical.py --emit-prerbf` (#33)."
        )
    da = ds["t2m_prerbf"]
    # Détection Kelvin robuste (médiane globale) — cf. _analyze_year, bug #17.
    src_units = str(da.attrs.get("units", "")).lower()
    sample_global = float(np.nanmedian(da.values))
    if src_units in ("k", "kelvin") or sample_global > 100:
        log.info("t2m_prerbf en Kelvin, conversion → Celsius")
        da = da - 273.15
    bbox = _bbox_from_grid(da)
    year = int(_zarr_stem(year_zarr))

    # Time int → dates synthétiques (cf. _analyze_year).
    if da["time"].dtype.kind != "M":
        start = pd.Timestamp(f"{year}-02-01")
        da = da.assign_coords(time=start + pd.to_timedelta(da["time"].values, unit="D"))

    stations_df = load_stations_catalog(sencrop_root, bbox=bbox)
    # Sous-périmètre optionnel (ex. cut Baronnies/Nyons cold-pool) : restreint les
    # stations d'évaluation ET donneuses du RBF à la sous-bbox, sans changer la grille.
    if station_bbox is not None:
        la0, la1, lo0, lo1 = station_bbox
        n_before = len(stations_df)
        stations_df = stations_df[
            (stations_df["latitude"] >= la0)
            & (stations_df["latitude"] <= la1)
            & (stations_df["longitude"] >= lo0)
            & (stations_df["longitude"] <= lo1)
        ]
        log.info(
            "station-bbox lat[%.3f,%.3f] lon[%.3f,%.3f] : %d → %d stations",
            la0, la1, lo0, lo1, n_before, len(stations_df),
        )
    bucket_ids = stations_df["bucket_id"].astype(int).tolist()
    ts = load_timeseries(years=[year], root=sencrop_root, station_only=True, bucket_ids=bucket_ids)
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True)
    ts["night_date"] = (ts["timestamp"] - pd.Timedelta("9h")).dt.date
    obs_per_night = ts.groupby(["night_date", "station_id"])["temperature"].min().reset_index()

    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lat_grid = da[lat_name].values
    lon_grid = da[lon_name].values

    station_meta = (
        stations_df[["bucket_id", "latitude", "longitude"]]
        .drop_duplicates(subset=["bucket_id"], keep="first")
        .set_index("bucket_id")
    )

    groups_by_mode = {
        "station": _cluster_groups(station_meta, cluster_km=0.0),
        "cluster": _cluster_groups(station_meta, cluster_km=cluster_km),
    }
    n_clusters = len(set(groups_by_mode["cluster"].values()))
    log.info(
        "LOO %s : %d stations, %d clusters à %.1f km",
        year,
        len(station_meta),
        n_clusters,
        cluster_km,
    )

    modes: dict[str, dict] = {}
    for mode, groups in groups_by_mode.items():
        per_station = _loo_predict(
            da, station_meta, obs_per_night, lat_grid, lon_grid, sigma_km, groups
        )
        all_obs: list[float] = []
        all_pred: list[float] = []
        station_rows: list[dict] = []
        for bid, rec in per_station.items():
            o = np.array(rec["obs"])
            pv = np.array(rec["pred"])
            all_obs.extend(rec["obs"])
            all_pred.extend(rec["pred"])
            cont = _contingency(o, pv, threshold_c)
            station_rows.append(
                {
                    "station_id": int(bid),
                    "group_id": int(groups[bid]),
                    "n_pairs": int(len(o)),
                    "POD": cont["POD"],
                    "FAR": cont["FAR"],
                    "CSI": cont["CSI"],
                    "rmse": float(np.sqrt(np.mean((o - pv) ** 2))) if len(o) else float("nan"),
                    "bias": float(np.mean(o - pv)) if len(o) else float("nan"),
                }
            )
        arr_obs = np.array(all_obs)
        arr_pred = np.array(all_pred)
        agg = _contingency(arr_obs, arr_pred, threshold_c) if len(arr_obs) else {}
        modes[mode] = {
            "n_pairs": int(len(arr_obs)),
            "n_groups": len(set(groups.values())),
            "aggregate": {
                **agg,
                "rmse": float(np.sqrt(np.mean((arr_obs - arr_pred) ** 2)))
                if len(arr_obs)
                else float("nan"),
                "bias": float(np.mean(arr_obs - arr_pred)) if len(arr_obs) else float("nan"),
            },
            "per_station": station_rows,
        }

    return {
        "year": year,
        "threshold_c": threshold_c,
        "sigma_km": sigma_km,
        "cluster_km": cluster_km,
        "n_stations": int(len(station_meta)),
        "bbox": bbox,
        "modes": modes,
    }


def _run_loo(zarrs: list[str], args: argparse.Namespace) -> int:
    """Exécute la validation hors-station sur toutes les années → CSV + JSON.

    Sortie (dans --out-dir, ou CWD) :
    - ``lot_b_loo_<variant>.csv`` : une ligne par (année × mode × station) + agrégats,
      avec audit trail (git_sha, command) — matière du livrable opposable C5.
    - ``<year>.loo.json`` : détail complet par année.
    """
    out_dir = args.out_dir if args.out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()
    command = " ".join(["uv", "run", "python", *sys.argv])

    summaries: list[dict] = []
    for z in zarrs:
        log.info("--- LOO year %s ---", _zarr_stem(z))
        try:
            s = _analyze_year_loo(
                z,
                args.sencrop,
                args.threshold_c,
                args.sigma_km,
                args.cluster_km,
                station_bbox=tuple(args.station_bbox) if args.station_bbox else None,
            )
        except Exception as exc:
            log.exception("LOO year %s failed: %s", _zarr_stem(z), exc)
            continue
        summaries.append(s)
        (out_dir / f"{s['year']}.loo.json").write_text(json.dumps(s, indent=2, default=str))

    if not summaries:
        log.error("Aucune année LOO produite (t2m_prerbf manquant ?)")
        return 2

    csv_path = out_dir / f"lot_b_loo_{args.variant}.csv"
    fields = [
        "variant",
        "grouping",
        "year",
        "scope",
        "station_id",
        "group_id",
        "n_pairs",
        "threshold_c",
        "sigma_km",
        "cluster_km",
        "POD",
        "FAR",
        "CSI",
        "rmse",
        "bias",
        "git_sha",
        "command",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            base = {
                "variant": args.variant,
                "year": s["year"],
                "threshold_c": s["threshold_c"],
                "sigma_km": s["sigma_km"],
                "cluster_km": s["cluster_km"],
                "git_sha": git_sha,
                "command": command,
            }
            for mode, md in s["modes"].items():
                agg = md["aggregate"]
                w.writerow(
                    {
                        **base,
                        "grouping": mode,
                        "scope": "ALL",
                        "station_id": "",
                        "group_id": "",
                        "n_pairs": md["n_pairs"],
                        "POD": agg.get("POD"),
                        "FAR": agg.get("FAR"),
                        "CSI": agg.get("CSI"),
                        "rmse": agg.get("rmse"),
                        "bias": agg.get("bias"),
                    }
                )
                for row in md["per_station"]:
                    w.writerow(
                        {
                            **base,
                            "grouping": mode,
                            "scope": "station",
                            "station_id": row["station_id"],
                            "group_id": row["group_id"],
                            "n_pairs": row["n_pairs"],
                            "POD": row["POD"],
                            "FAR": row["FAR"],
                            "CSI": row["CSI"],
                            "rmse": row["rmse"],
                            "bias": row["bias"],
                        }
                    )
    log.info("Wrote %s", csv_path)

    # Synthèse console : agrégat hors-station par année × mode.
    for s in summaries:
        for mode, md in s["modes"].items():
            a = md["aggregate"]
            log.info(
                "LOO %s [%s] : POD=%.2f FAR=%.2f CSI=%.2f RMSE=%.2f (n=%d, %d groupes)",
                s["year"],
                mode,
                a.get("POD", float("nan")),
                a.get("FAR", float("nan")),
                a.get("CSI", float("nan")),
                a.get("rmse", float("nan")),
                md["n_pairs"],
                md["n_groups"],
            )

    # W&B : logge les agrégats hors-station (le livrable métrique du #33). Le chemin
    # LOO ne passait pas par l'init W&B de main() — d'où l'absence de ces métriques
    # dans le projet. Guardé par WANDB_API_KEY + --wandb-disabled.
    if not getattr(args, "wandb_disabled", False) and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                name=f"loo-{args.variant}",
                config={
                    "stage": "loo_out_of_station",
                    "variant": args.variant,
                    "threshold_c": args.threshold_c,
                    "sigma_km": args.sigma_km,
                    "cluster_km": args.cluster_km,
                    "git_sha": git_sha,
                    "years": [s["year"] for s in summaries],
                },
                reinit=True,
            )
            tbl = wandb.Table(
                columns=["year", "grouping", "n_pairs", "POD", "FAR", "CSI", "rmse", "bias"]
            )
            for s in summaries:
                for mode, md in s["modes"].items():
                    a = md["aggregate"]
                    tbl.add_data(
                        s["year"], mode, md["n_pairs"],
                        a.get("POD"), a.get("FAR"), a.get("CSI"), a.get("rmse"), a.get("bias"),
                    )
                    wandb.log(
                        {
                            f"{mode}/{s['year']}/POD": a.get("POD"),
                            f"{mode}/{s['year']}/FAR": a.get("FAR"),
                            f"{mode}/{s['year']}/CSI": a.get("CSI"),
                            f"{mode}/{s['year']}/rmse": a.get("rmse"),
                        }
                    )
            wandb.log({"loo_summary": tbl})
            log.info("W&B LOO run: %s", run.url)
            run.finish()
        except Exception as exc:
            log.warning("W&B LOO logging failed (continuing): %s", exc)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dir avec <year>.zarr (local OU s3:// — cf. AWS_ENDPOINT_URL pour Scaleway)",
    )
    p.add_argument("--sencrop", type=str, required=True, help="Sencrop root (local ou s3://)")
    p.add_argument("--threshold-c", type=float, default=-2.2)
    p.add_argument(
        "--loo",
        action="store_true",
        help="Validation HORS-STATION (out-of-sample) : rejoue le RBF en leave-one-out "
        "station-par-station ET leave-one-cluster-out. Exige la var t2m_prerbf dans le "
        "Zarr (recalibrate --emit-prerbf). Produit lot_b_loo_<variant>.csv (#33).",
    )
    p.add_argument(
        "--sigma-km",
        type=float,
        default=7.0,
        help="σ du RBF résiduel — DOIT matcher la valeur de prod (recalibrate --sigma-km).",
    )
    p.add_argument(
        "--cluster-km",
        type=float,
        default=7.0,
        help="Distance de regroupement pour le leave-one-cluster-out (défaut = σ). "
        "Les stations plus proches sont exclues ensemble.",
    )
    p.add_argument(
        "--variant",
        type=str,
        default="nu",
        help="Étiquette du run LOO (ex. 'nu' ou 'qdm') → nom de fichier + colonne CSV.",
    )
    p.add_argument(
        "--station-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Restreint le LOO à un sous-périmètre de stations (ex. cut Baronnies/Nyons "
        "cold-pool) sans changer la grille. Évite la dilution par les plaines Rhône.",
    )
    p.add_argument(
        "--regimes-csv",
        type=Path,
        default=None,
        help="CSV avec colonnes 'date' et 'regime' (cf. flag_regimes.py). "
        "Active la stratification POD/FAR/CSI par régime synoptique (C5.3).",
    )
    p.add_argument("--wandb-project", default="karpos-recalibrate-statistical")
    p.add_argument("--wandb-disabled", action="store_true")
    p.add_argument(
        "--years", type=int, nargs="*", default=None, help="Subset, défaut: tous les *.zarr trouvés"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Dossier local pour les sidecars <year>.posthoc.json. "
        "Défaut : à côté du Zarr si --root local, CWD si --root s3://.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    zarrs = _list_zarrs(args.root)
    if args.years:
        zarrs = [z for z in zarrs if int(_zarr_stem(z)) in args.years]
    if not zarrs:
        log.error("No <year>.zarr under %s", args.root)
        return 2
    log.info("Analyzing %d Zarr(s): %s", len(zarrs), [_zarr_stem(z) + ".zarr" for z in zarrs])

    # -----------------------------------------------------------------------
    # Mode LOO / leave-one-cluster-out (#33) — court-circuite l'analyse in-sample.
    # -----------------------------------------------------------------------
    if args.loo:
        return _run_loo(zarrs, args)

    # W&B init (optionnel)
    wandb_run = None
    if not args.wandb_disabled and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"analyze-stat-{'_'.join([_zarr_stem(z) for z in zarrs])}",
                config={
                    "stage": "statistical_posthoc",
                    "threshold_c": args.threshold_c,
                    "sencrop_root": args.sencrop,
                    "years": [int(_zarr_stem(z)) for z in zarrs],
                },
                reinit=True,
            )
            log.info("W&B run: %s", wandb_run.url)
        except Exception as exc:
            log.warning("W&B init failed: %s", exc)

    regimes = _load_regimes(args.regimes_csv)
    if regimes:
        log.info("Régimes synoptiques chargés : %d nuits dans %s", len(regimes), args.regimes_csv)

    all_summaries: list[dict] = []
    for z in zarrs:
        log.info("--- year %s ---", _zarr_stem(z))
        try:
            s = _analyze_year(z, args.sencrop, args.threshold_c, regimes=regimes)
        except Exception as exc:
            log.exception("Year %s failed: %s", _zarr_stem(z), exc)
            continue
        all_summaries.append(s)
        # Write metadata.posthoc.json sidecar.
        # - local --root : à côté du Zarr (rétro-compat)
        # - s3 --root    : impossible d'écrire à côté sans creds write, donc
        #                  on tombe en CWD (ou --out-dir si fourni).
        stem = _zarr_stem(z)
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_json = args.out_dir / f"{stem}.posthoc.json"
        elif _is_remote(z):
            out_json = Path.cwd() / f"{stem}.posthoc.json"
        else:
            out_json = Path(z).parent / f"{stem}.posthoc.json"
        out_json.write_text(json.dumps(s, indent=2, default=str))
        log.info("Wrote %s", out_json)
        if wandb_run is not None:
            try:
                import wandb

                # Flat log par année (préfixé)
                yr = s["year"]
                wandb.log(
                    {
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
                    }
                )
            except Exception as exc:
                log.warning("W&B log year %s failed: %s", s["year"], exc)

    # Synthèse multi-années (POD/FAR/CSI agrégés)
    valid = [
        s for s in all_summaries if "contingency" in s and s.get("n_pairs_station_night", 0) > 0
    ]
    if valid:
        # Moyenne pondérée des résiduels
        weights = np.array([s["n_pairs_station_night"] for s in valid], dtype=float)
        rmse_w = float(
            np.sqrt(np.average([s["residual_rmse_year"] ** 2 for s in valid], weights=weights))
        )
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
