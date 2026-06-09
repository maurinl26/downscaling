#!/usr/bin/env python
"""
Évaluation POD/FAR du U-Net downscalé vs stations Sencrop (ground truth).

Le *vrai* protocole métier : sur des nuits **held-out** (hors entraînement), on
descend la réanalyse (ERA5-Land → champ 1 km) avec le U-Net entraîné, on échantillonne
le Tmin prédit à la maille de chaque station (correction d'altitude lapse·dz), et on
compare au Tmin observé Sencrop → POD / FAR / RMSE au seuil de gel.

Distinct du `val/pod` de l'entraînement (qui compare à la CIBLE CERRA, pas aux stations).

Réutilise la machinerie étage C (provider + station targets) sans entraîner.

Usage :
  uv run python downscaling/scripts/eval_unet_sencrop.py \
      --checkpoint checkpoints/residual_run/best_model.ckpt \
      --stats      checkpoints/residual_run/normalization_stats.json \
      --cerra-fine-dir data/cerra_fine --sencrop-dir data/sencrop \
      --dem data/training/dem_attributes.nc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from downscaling.deep_learning.cerra_provider import CERRACoarseProvider
from downscaling.deep_learning.model import build_model
from downscaling.deep_learning.sparse_calibration import (
    UNetSparseCalibrationModule, UNetStationDataset, unet_sparse_collate,
)
from downscaling.scripts.run_calibration import grids_from_dem, _load_unet_weights


def operating_points(pred: np.ndarray, obs: np.ndarray, thr: float) -> dict:
    """POD/FAR/CSI @ seuil + AUC(ROC) + meilleur POD à FAR<0,20 et meilleur FAR à POD>0,75.

    ROC vectorisé : on déclare gel si Tmin prédit ≤ τ ; en triant par Tmin prédit
    croissant, TP/FP cumulés donnent toute la courbe en O(n log n).
    """
    label = obs < thr
    P, N = int(label.sum()), int((~label).sum())
    # Point de fonctionnement au seuil physique (pred < thr)
    sf = pred < thr
    hits = int((label & sf).sum()); misses = P - hits
    fa = int((~label & sf).sum())
    pod = hits / (hits + misses) if (hits + misses) else float("nan")
    far = fa / (hits + fa) if (hits + fa) else float("nan")
    csi = hits / (hits + misses + fa) if (hits + misses + fa) else float("nan")
    # Courbe ROC (tri croissant sur pred = déclare gel pour les plus froids d'abord)
    order = np.argsort(pred, kind="mergesort")
    lab = label[order].astype(np.int64)
    cum_tp = np.cumsum(lab)
    cum_fp = np.cumsum(1 - lab)
    tpr = cum_tp / P if P else np.zeros_like(cum_tp, dtype=float)
    fpr = cum_fp / N if N else np.zeros_like(cum_fp, dtype=float)
    far_b = cum_fp / np.maximum(cum_tp + cum_fp, 1)
    auc = float(np.trapz(np.concatenate([[0.0], tpr]), np.concatenate([[0.0], fpr])))
    pod_at = float(tpr[far_b < 0.20].max()) if (far_b < 0.20).any() else float("nan")
    far_at = float(far_b[tpr > 0.75].min()) if (tpr > 0.75).any() else float("nan")
    return {"pod": pod, "far": far, "csi": csi, "hits": hits, "misses": misses, "fa": fa,
            "auc": auc, "pod_at_far20": pod_at, "far_at_pod75": far_at}


def main() -> None:
    ap = argparse.ArgumentParser(description="Éval POD/FAR U-Net vs Sencrop")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--cerra-fine-dir", default="data/cerra_fine")
    ap.add_argument("--sencrop-dir", default="data/sencrop")
    ap.add_argument("--dem", default="data/training/dem_attributes.nc")
    ap.add_argument("--file-template-cerra", default="era5land_{date}.nc")
    ap.add_argument("--met-vars", default="t2m")
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--no-residual", action="store_true", help="désactive le résiduel (ablation)")
    ap.add_argument("--lapse-rate", type=float, default=-4.0e-3)
    ap.add_argument("--threshold", type=float, default=0.0, help="seuil gel (°C)")
    ap.add_argument("--min-stations", type=int, default=5)
    args = ap.parse_args()

    met_vars = args.met_vars.split(",")
    stats = {k: tuple(v) for k, v in json.load(open(args.stats)).items()}

    model = build_model(architecture="unet", met_in_ch=len(met_vars), dem_in_ch=4,
                        base_ch=args.base_ch, n_levels=4, use_film=True,
                        residual=not args.no_residual)
    _load_unet_weights(model, args.checkpoint)
    model.eval()

    provider = CERRACoarseProvider(
        args.cerra_fine_dir, args.dem, met_vars=met_vars, stats=stats,
        file_template=args.file_template_cerra, hourly=True, reduce="min",
    )
    lat_grid, lon_grid, elev_grid = grids_from_dem(xr.open_dataset(args.dem))
    dataset = UNetStationDataset(
        provider.dates(), provider, args.sencrop_dir, lat_grid, lon_grid,
        file_template="sencrop_{date}.csv", elevation_grid=elev_grid,
        min_stations=args.min_stations, lapse_rate=args.lapse_rate,
    )
    lit = UNetSparseCalibrationModule(
        model, denorm=stats.get("t2m"), kelvin_to_celsius=False,
        lapse_rate=args.lapse_rate, elevation_aware=True, hourly=True, reduce="min",
    )
    lit.eval()

    thr = args.threshold
    all_pred, all_obs, all_key = [], [], []
    n_nights = 0
    for i in range(len(dataset)):
        batch = unet_sparse_collate([dataset[i]])
        with torch.no_grad():
            pred = lit._predict_target(batch)              # (1,1,H,W) °C
        row, col = batch["obs_row"][0], batch["obs_col"][0]
        obs = batch["obs_tmin"][0].numpy()
        dz = batch["obs_dz"][0]
        pa = pred[0, 0, row, col]
        if dz is not None:
            pa = pa + args.lapse_rate * dz                 # correction d'altitude station/maille
        pa = pa.numpy()
        key = (row.numpy().astype(np.int64) * 100000 + col.numpy().astype(np.int64))  # maille = proxy station
        m = ~np.isnan(obs) & ~np.isnan(pa)
        if not m.any():
            continue
        n_nights += 1
        all_pred.append(pa[m]); all_obs.append(obs[m]); all_key.append(key[m])

    pred = np.concatenate(all_pred); obs = np.concatenate(all_obs)
    key = np.concatenate(all_key)
    rmse = float(np.sqrt(((pred - obs) ** 2).mean()))

    raw = operating_points(pred, obs, thr)

    # Oracle calibration par station : retire le biais médian par maille (≈ étage C).
    # In-sample → borne SUPÉRIEURE de ce que la calibration station pourrait débloquer.
    pred_db = pred.copy()
    for k in np.unique(key):
        sel = key == k
        if sel.sum() >= 5:
            pred_db[sel] = pred[sel] - np.median(pred[sel] - obs[sel])
    db = operating_points(pred_db, obs, thr)

    tag = "ABSOLU" if args.no_residual else "RÉSIDUEL"
    print(f"\n=== Éval U-Net ({tag}) vs Sencrop — {n_nights} nuits, {len(obs)} obs-station, seuil {thr}°C ===")
    print(f"  Point @{thr}°C : POD={raw['pod']:.3f}  FAR={raw['far']:.3f}  CSI={raw['csi']:.3f}  "
          f"(hits {raw['hits']}/misses {raw['misses']}/FA {raw['fa']})")
    print(f"  RMSE Tmin = {rmse:.2f} °C   |   AUC(ROC) = {raw['auc']:.3f}")
    print(f"  Seuil optimisé   : POD={raw['pod_at_far20']:.3f} @FAR<0,20  ·  FAR={raw['far_at_pod75']:.3f} @POD>0,75")
    print(f"  + débiais/station: POD={db['pod_at_far20']:.3f} @FAR<0,20  ·  FAR={db['far_at_pod75']:.3f} @POD>0,75"
          f"  (AUC {db['auc']:.3f}, borne sup. calibration)")
    print(f"\n  Rappel baselines @0°C (médiane 48 st.) : RAW 0,77/0,53 · EQM 0,62/0,40 · KF 0,58/0,25")
    print(f"  Cible : POD>0,75  FAR<0,20")


if __name__ == "__main__":
    main()
