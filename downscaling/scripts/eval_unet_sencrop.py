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
    all_pred, all_obs = [], []
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
        m = ~np.isnan(obs) & ~np.isnan(pa)
        if not m.any():
            continue
        n_nights += 1
        all_pred.append(pa[m]); all_obs.append(obs[m])

    pred = np.concatenate(all_pred); obs = np.concatenate(all_obs)
    of, sf = obs < thr, pred < thr                          # gel observé / prédit @ seuil
    hits = int((of & sf).sum()); misses = int((of & ~sf).sum())
    fa = int((~of & sf).sum()); cn = int((~of & ~sf).sum())
    pod = hits / (hits + misses) if (hits + misses) else float("nan")
    far = fa / (hits + fa) if (hits + fa) else float("nan")
    csi = hits / (hits + misses + fa) if (hits + misses + fa) else float("nan")
    rmse = float(np.sqrt(((pred - obs) ** 2).mean()))

    # --- ROC / AUC : pouvoir discriminant (sweep du seuil de DÉCISION sur Tmin prédit) ---
    label = of                                              # vrai gel (obs < thr)
    P, N = int(label.sum()), int((~label).sum())
    roc = []   # (decision_tau, POD=TPR, FPR, FAR_business)
    for tau in np.unique(pred):
        decl = pred <= tau                                  # déclare gel si Tmin prédit ≤ tau
        tp = int((decl & label).sum()); fp = int((decl & ~label).sum())
        fn = P - tp
        tpr = tp / P if P else np.nan
        fpr = fp / N if N else np.nan
        far_b = fp / (tp + fp) if (tp + fp) else np.nan
        roc.append((tau, tpr, fpr, far_b))
    roc = np.array(roc)
    order = np.argsort(roc[:, 2])                            # par FPR croissant
    auc = float(np.trapz(roc[order, 1], roc[order, 2]))
    # Point de fonctionnement métier : parmi les seuils avec FAR<0,20, POD max
    ok = roc[(roc[:, 3] < 0.20)]
    best_pod = float(ok[:, 1].max()) if len(ok) else float("nan")
    # et : parmi les seuils avec POD>0,75, FAR min
    ok2 = roc[(roc[:, 1] > 0.75)]
    best_far = float(ok2[:, 3].min()) if len(ok2) else float("nan")

    tag = "ABSOLU" if args.no_residual else "RÉSIDUEL"
    print(f"\n=== Éval U-Net ({tag}) vs Sencrop — {n_nights} nuits, {len(obs)} obs-station, seuil {thr}°C ===")
    print(f"  Point @{thr}°C : POD={pod:.3f}  FAR={far:.3f}  CSI={csi:.3f}  (hits {hits}/misses {misses}/FA {fa})")
    print(f"  RMSE Tmin = {rmse:.2f} °C")
    print(f"\n  Discrimination — AUC(ROC) = {auc:.3f}")
    print(f"  Meilleur POD atteignable à FAR<0,20 : {best_pod:.3f}")
    print(f"  Meilleur FAR atteignable à POD>0,75 : {best_far:.3f}")
    print(f"\n  Rappel baselines @0°C (médiane 48 st.) : RAW 0,77/0,53 · EQM 0,62/0,40 · KF 0,58/0,25")
    print(f"  Cible : POD>0,75  FAR<0,20")


if __name__ == "__main__":
    main()
