#!/usr/bin/env python
"""
Comparaison ROC multi-classifieurs (détection du gel) vs stations Sencrop.

Pour chaque classifieur (réanalyse brute, U-Net résiduel, étages C…), on échantillonne
le Tmin prédit aux stations sur un jeu held-out, on balaie le seuil de décision (= les
"paramètres" du classifieur côté opération), et on trace :
  - panneau gauche : ROC standard (TPR vs FPR) + AUC en légende ;
  - panneau droit  : POD vs FAR (métier) + zone cible FAR<0,20 / POD>0,75.

Classifieurs passés en `--clf "label:checkpoint:met_vars:cerra_fine_dir"`
(checkpoint='raw' = réanalyse brute, pas de modèle).

Usage :
  uv run python downscaling/scripts/roc_compare.py \
    --dem data/training/dem_attributes.nc --sencrop-dir data/sencrop --out roc_gel_2025.png \
    --clf "RAW ERA5-Land:raw:t2m,td2m,u10:data/cerra_fine_mv_test" \
    --clf "U-Net résiduel:checkpoints/mv_run/best_model.ckpt:t2m,td2m,u10:data/cerra_fine_mv_test" \
    --clf "Étage C MSE:checkpoints/mv_run/calibrated.pt:t2m,td2m,u10:data/cerra_fine_mv_test" \
    --clf "Étage C tail-aware:checkpoints/mv_run/calibrated_taw.pt:t2m,td2m,u10:data/cerra_fine_mv_test"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from downscaling.deep_learning.cerra_provider import CERRACoarseProvider
from downscaling.deep_learning.model import build_model
from downscaling.deep_learning.sparse_calibration import (
    UNetSparseCalibrationModule, UNetStationDataset, unet_sparse_collate,
)
from downscaling.scripts.run_calibration import grids_from_dem, _load_unet_weights


def collect_scores(checkpoint, met_vars, cerra_dir, sencrop_dir, dem, stats,
                   lat_g, lon_g, elev_g, lapse=-4.0e-3) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (pred_Tmin °C aux stations, obs_Tmin) sur toutes les nuits du dir."""
    provider = CERRACoarseProvider(cerra_dir, dem, met_vars=met_vars, stats=stats,
                                   file_template="era5land_{date}.nc", hourly=True, reduce="min")
    ds = UNetStationDataset(provider.dates(), provider, sencrop_dir, lat_g, lon_g,
                            file_template="sencrop_{date}.csv", elevation_grid=elev_g,
                            min_stations=5, lapse_rate=lapse)
    mu, sg = stats["t2m"]
    lit = None
    if checkpoint != "raw":
        model = build_model(architecture="unet", met_in_ch=len(met_vars), met_out_ch=1,
                            dem_in_ch=4, base_ch=64, n_levels=4, use_film=True, residual=True)
        _load_unet_weights(model, checkpoint); model.eval()
        lit = UNetSparseCalibrationModule(model, denorm=stats["t2m"], kelvin_to_celsius=False,
                                          lapse_rate=lapse, elevation_aware=True, hourly=True, reduce="min")
        lit.eval()
    preds, obss = [], []
    for i in range(len(ds)):
        b = unet_sparse_collate([ds[i]])
        row, col, obs, dz = b["obs_row"][0], b["obs_col"][0], b["obs_tmin"][0].numpy(), b["obs_dz"][0]
        if checkpoint == "raw":
            # Tmin de la maille d'entrée (t2m, canal 0) : descente horaire → min, dénormalisé.
            xm = b["x_met"][0]                      # (T, C, H, W) normalisé
            field = (xm[:, 0].min(dim=0).values * sg + mu)   # (H,W) °C
            pa = field[row, col]
        else:
            with torch.no_grad():
                field = lit._predict_target(b)[0, 0]
            pa = field[row, col]
        if dz is not None:
            pa = pa + lapse * dz
        pa = pa.numpy()
        m = ~np.isnan(obs) & ~np.isnan(pa)
        preds.append(pa[m]); obss.append(obs[m])
    return np.concatenate(preds), np.concatenate(obss)


def roc(pred, obs, thr=0.0):
    """ROC vectorisé : déclare gel si Tmin prédit ≤ τ. Retourne tpr, fpr, far, auc."""
    label = obs < thr
    P, N = int(label.sum()), int((~label).sum())
    o = np.argsort(pred, kind="mergesort")
    lab = label[o].astype(np.int64)
    tp, fp = np.cumsum(lab), np.cumsum(1 - lab)
    tpr = np.concatenate([[0.0], tp / max(P, 1)])
    fpr = np.concatenate([[0.0], fp / max(N, 1)])
    far = fp / np.maximum(tp + fp, 1)
    auc = float(np.trapz(tpr, fpr))
    return tpr, fpr, np.concatenate([[1.0], far]), auc


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparaison ROC multi-classifieurs gel")
    ap.add_argument("--dem", default="data/training/dem_attributes.nc")
    ap.add_argument("--sencrop-dir", default="data/sencrop")
    ap.add_argument("--out", default="roc_gel.png")
    ap.add_argument("--clf", action="append", required=True,
                    help="'label:checkpoint:met_vars:cerra_fine_dir' (checkpoint=raw pour réanalyse brute)")
    args = ap.parse_args()

    lat_g, lon_g, elev_g = grids_from_dem(xr.open_dataset(args.dem))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    rows = []
    for spec in args.clf:
        label, ckpt, mv, cdir = spec.split(":")
        met_vars = mv.split(",")
        stats = {k: tuple(v) for k, v in json.load(open(
            Path(ckpt).parent / "normalization_stats.json" if ckpt != "raw"
            else "checkpoints/mv_run/normalization_stats.json")).items()}
        print(f"[roc] {label}…")
        pred, obs = collect_scores(ckpt, met_vars, cdir, args.sencrop_dir, args.dem, stats,
                                   lat_g, lon_g, elev_g)
        tpr, fpr, far, auc = roc(pred, obs)
        pod_at = float(tpr[1:][far[1:] < 0.20].max()) if (far[1:] < 0.20).any() else float("nan")
        axL.plot(fpr, tpr, lw=2, label=f"{label} (AUC {auc:.3f})")
        axR.plot(far[1:], tpr[1:], lw=2, label=f"{label} (POD@FAR<.20={pod_at:.2f})")
        rows.append({"label": label, "auc": round(auc, 3), "pod_at_far20": round(pod_at, 3),
                     "n_obs": int(len(obs)), "n_frost": int((obs < 0).sum())})

    axL.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axL.set(xlabel="FPR", ylabel="TPR (POD)", title="ROC (détection gel 0°C)")
    axL.legend(fontsize=8); axL.grid(alpha=0.3)
    # zone cible FAR<0,20 / POD>0,75
    axR.axvspan(0, 0.20, color="green", alpha=0.07)
    axR.axhline(0.75, color="green", ls=":", lw=1)
    axR.axvline(0.20, color="green", ls=":", lw=1)
    axR.set(xlabel="FAR (fausses alertes)", ylabel="POD (détection)", xlim=(0, 1), ylim=(0, 1),
            title="POD vs FAR (métier) — zone cible verte")
    axR.legend(fontsize=8); axR.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    import pandas as pd
    pd.DataFrame(rows).to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(f"\n✓ {args.out}  +  {Path(args.out).with_suffix('.csv')}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
