#!/usr/bin/env python
"""
Figure produit définitive : ROC + POD/FAR avec la **courbe calibrée par station**.

Trois classifieurs sur le jeu TEST (held-out), pour montrer le chemin produit :
  1. RAW réanalyse brute (maille, sans modèle) ;
  2. U-Net résiduel + FiLM·DEM (descente d'échelle, non calibrée) ;
  3. U-Net résiduel + **calibration station out-of-sample** : biais médian par station
     ajusté sur CALIB (2022-2024), appliqué sur TEST (2025) → le chiffre opérationnel.

Le point OPÉRATIONNEL (seuil τ* transféré de calib) est marqué d'une étoile dans la
zone cible verte (FAR<0,20 / POD>0,75).

Usage :
  uv run python downscaling/scripts/roc_report.py \
    --checkpoint checkpoints/mv_run/best_model.ckpt --met-vars t2m,td2m,u10 \
    --calib-dir data/cerra_fine_mv_cal --test-dir data/cerra_fine_mv_test \
    --sencrop-dir data/sencrop --dem data/training/dem_attributes.nc \
    --out roc_produit_2025.png
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


def collect(checkpoint, met_vars, cdir, sencrop_dir, dem, stats,
            lat_g, lon_g, elev_g, lapse=-4.0e-3):
    """(pred_Tmin °C, obs_Tmin, station_key) sur toutes les nuits du dir.
    checkpoint='raw' → maille brute (canal t2m), sinon inférence U-Net résiduel."""
    provider = CERRACoarseProvider(cdir, dem, met_vars=met_vars, stats=stats,
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
    P, O, K = [], [], []
    for i in range(len(ds)):
        b = unet_sparse_collate([ds[i]])
        row, col, obs, dz = b["obs_row"][0], b["obs_col"][0], b["obs_tmin"][0].numpy(), b["obs_dz"][0]
        if checkpoint == "raw":
            field = (b["x_met"][0][:, 0].min(dim=0).values * sg + mu)
            pa = field[row, col]
        else:
            with torch.no_grad():
                pa = lit._predict_target(b)[0, 0][row, col]
        if dz is not None:
            pa = pa + lapse * dz
        pa = pa.numpy()
        key = row.numpy().astype(np.int64) * 100000 + col.numpy().astype(np.int64)
        m = ~np.isnan(obs) & ~np.isnan(pa)
        P.append(pa[m]); O.append(obs[m]); K.append(key[m])
    return np.concatenate(P), np.concatenate(O), np.concatenate(K)


def roc(pred, obs, thr=0.0):
    """ROC vectorisé : déclare gel si Tmin prédit ≤ τ. Retourne tpr, fpr, far, auc."""
    label = obs < thr
    P, N = int(label.sum()), int((~label).sum())
    o = np.argsort(pred, kind="mergesort")
    lab = label[o].astype(np.int64)
    tp, fp = np.cumsum(lab), np.cumsum(1 - lab)
    tpr = np.concatenate([[0.0], tp / max(P, 1)])
    fpr = np.concatenate([[0.0], fp / max(N, 1)])
    far = np.concatenate([[1.0], fp / np.maximum(tp + fp, 1)])
    auc = float(np.trapz(tpr, fpr))
    return tpr, fpr, far, auc


def per_station_bias(pred, obs, key, min_n=5):
    """Biais médian (pred-obs) par station sur le jeu fourni + repli global."""
    bias = {}
    for k in np.unique(key):
        sel = key == k
        if sel.sum() >= min_n:
            bias[int(k)] = float(np.median(pred[sel] - obs[sel]))
    return bias, float(np.median(pred - obs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure produit ROC + calibration station")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--met-vars", default="t2m,td2m,u10")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--sencrop-dir", default="data/sencrop")
    ap.add_argument("--dem", default="data/training/dem_attributes.nc")
    ap.add_argument("--out", default="roc_produit_2025.png")
    ap.add_argument("--thr", type=float, default=0.0)
    args = ap.parse_args()

    met_vars = args.met_vars.split(",")
    stats = {k: tuple(v) for k, v in json.load(
        open(Path(args.checkpoint).parent / "normalization_stats.json")).items()}
    lat_g, lon_g, elev_g = grids_from_dem(xr.open_dataset(args.dem))
    g = (stats, lat_g, lon_g, elev_g)

    print("[report] inférence RAW test…")
    raw_p, raw_o, _ = collect("raw", met_vars, args.test_dir, args.sencrop_dir, args.dem, *g)
    print("[report] inférence U-Net test…")
    un_pt, un_ot, un_kt = collect(args.checkpoint, met_vars, args.test_dir, args.sencrop_dir, args.dem, *g)
    print("[report] inférence U-Net calib (pour biais station)…")
    un_pc, un_oc, un_kc = collect(args.checkpoint, met_vars, args.calib_dir, args.sencrop_dir, args.dem, *g)

    # calibration station OOS : biais fit sur calib → appliqué test
    bias, gbias = per_station_bias(un_pc, un_oc, un_kc)
    cal_pt = un_pt - np.array([bias.get(int(k), gbias) for k in un_kt])

    # seuil opérationnel τ* : meilleur POD@FAR<0,20 sur calib (débiaisé), transféré test
    cal_pc = un_pc - np.array([bias.get(int(k), gbias) for k in un_kc])
    lab_c = un_oc < args.thr
    order = np.argsort(cal_pc); ls = lab_c[order].astype(int)
    tp, fp = np.cumsum(ls), np.cumsum(1 - ls)
    far_c = fp / np.maximum(tp + fp, 1)
    ok = far_c < 0.20
    tau_star = float(cal_pc[order][ok][np.argmax((tp / max(lab_c.sum(), 1))[ok])]) if ok.any() else args.thr
    declT = cal_pt <= tau_star; labT = un_ot < args.thr
    pod_op = int((declT & labT).sum()) / max(int(labT.sum()), 1)
    far_op = int((declT & ~labT).sum()) / max(int(declT.sum()), 1)

    series = [
        ("RAW réanalyse brute", raw_p, raw_o, "#888888"),
        ("U-Net résiduel + FiLM·DEM", un_pt, un_ot, "#1f77b4"),
        ("U-Net + calibration station (OOS)", cal_pt, un_ot, "#d62728"),
    ]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    rows = []
    for label, p, o, c in series:
        tpr, fpr, far, auc = roc(p, o, args.thr)
        pod_at = float(tpr[1:][far[1:] < 0.20].max()) if (far[1:] < 0.20).any() else float("nan")
        axL.plot(fpr, tpr, lw=2.2, color=c, label=f"{label} (AUC {auc:.3f})")
        axR.plot(far[1:], tpr[1:], lw=2.2, color=c, label=f"{label} (POD@FAR<.20={pod_at:.2f})")
        rows.append({"label": label, "auc": round(auc, 3), "pod_at_far20": round(pod_at, 3),
                     "n_obs": int(len(o)), "n_frost": int((o < args.thr).sum())})

    # point opérationnel (seuil τ* transféré de calib)
    axR.plot([far_op], [pod_op], marker="*", ms=20, color="#d62728", mec="k", mew=0.8, zorder=5,
             label=f"Point opérationnel τ*={tau_star:.2f}°C (POD {pod_op:.2f} / FAR {far_op:.2f})")

    axL.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axL.set(xlabel="FPR (fausses alertes / non-gels)", ylabel="TPR — POD (détection du gel)",
            title="ROC — détection du gel (seuil 0°C), test 2025")
    axL.legend(fontsize=8, loc="lower right"); axL.grid(alpha=0.3)
    axR.axvspan(0, 0.20, color="green", alpha=0.07)
    axR.axhline(0.75, color="green", ls=":", lw=1); axR.axvline(0.20, color="green", ls=":", lw=1)
    axR.set(xlabel="FAR (taux de fausses alertes)", ylabel="POD (taux de détection)",
            xlim=(0, 1), ylim=(0, 1), title="POD vs FAR (métier) — zone cible verte")
    axR.legend(fontsize=8, loc="lower right"); axR.grid(alpha=0.3)
    fig.suptitle("Performance produit — descente d'échelle gel · Sencrop Drôme-Ardèche",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(args.out, dpi=130)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(f"\n✓ {args.out}  +  {Path(args.out).with_suffix('.csv')}")
    print(df.to_string(index=False))
    print(f"\nPoint OPÉRATIONNEL (τ*={tau_star:.2f}°C transféré de calib) : "
          f"POD {pod_op:.3f}  FAR {far_op:.3f}  (n_test={len(un_ot)}, n_frost={int((un_ot<0).sum())})")


if __name__ == "__main__":
    main()
