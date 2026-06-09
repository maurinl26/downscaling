#!/usr/bin/env python
"""
Calibration par station OUT-OF-SAMPLE (le chiffre opérationnel honnête).

Au lieu de l'oracle in-sample (biais ajusté et noté sur le même jeu → optimiste), on :
  1. ajuste le **biais médian par station** + le **seuil de décision** sur un jeu CALIB ;
  2. les **applique** sur un jeu TEST séparé (jamais vu pour l'ajustement) ;
  3. reporte POD/FAR/AUC + meilleur POD@FAR<0,20.

C'est la calibration « propre » : fit dev → applique held-out. À reporter in fine sur 2026.

Usage :
  uv run python downscaling/scripts/calib_oos.py \
    --checkpoint checkpoints/mv_run/best_model.ckpt --met-vars t2m,td2m,u10 \
    --calib-dir data/cerra_fine_mv_cal --test-dir data/cerra_fine_mv_test \
    --sencrop-dir data/sencrop --dem data/training/dem_attributes.nc
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
from downscaling.scripts.eval_unet_sencrop import operating_points


def collect(checkpoint, met_vars, cdir, sencrop_dir, dem, stats, lat_g, lon_g, elev_g, lapse=-4.0e-3):
    """(pred_Tmin °C, obs_Tmin, station_key) sur toutes les nuits du dir."""
    provider = CERRACoarseProvider(cdir, dem, met_vars=met_vars, stats=stats,
                                   file_template="era5land_{date}.nc", hourly=True, reduce="min")
    ds = UNetStationDataset(provider.dates(), provider, sencrop_dir, lat_g, lon_g,
                            file_template="sencrop_{date}.csv", elevation_grid=elev_g,
                            min_stations=5, lapse_rate=lapse)
    model = build_model(architecture="unet", met_in_ch=len(met_vars), met_out_ch=1, dem_in_ch=4,
                        base_ch=64, n_levels=4, use_film=True, residual=True)
    _load_unet_weights(model, checkpoint); model.eval()
    lit = UNetSparseCalibrationModule(model, denorm=stats["t2m"], kelvin_to_celsius=False,
                                      lapse_rate=lapse, elevation_aware=True, hourly=True, reduce="min")
    lit.eval()
    P, O, K = [], [], []
    for i in range(len(ds)):
        b = unet_sparse_collate([ds[i]])
        row, col, obs, dz = b["obs_row"][0], b["obs_col"][0], b["obs_tmin"][0].numpy(), b["obs_dz"][0]
        with torch.no_grad():
            pa = lit._predict_target(b)[0, 0][row, col]
        if dz is not None:
            pa = pa + lapse * dz
        pa = pa.numpy()
        key = row.numpy().astype(np.int64) * 100000 + col.numpy().astype(np.int64)
        m = ~np.isnan(obs) & ~np.isnan(pa)
        P.append(pa[m]); O.append(obs[m]); K.append(key[m])
    return np.concatenate(P), np.concatenate(O), np.concatenate(K)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration station out-of-sample")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--met-vars", default="t2m,td2m,u10")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--sencrop-dir", default="data/sencrop")
    ap.add_argument("--dem", default="data/training/dem_attributes.nc")
    ap.add_argument("--thr", type=float, default=0.0)
    args = ap.parse_args()

    met_vars = args.met_vars.split(",")
    stats = {k: tuple(v) for k, v in json.load(open(Path(args.checkpoint).parent / "normalization_stats.json")).items()}
    lat_g, lon_g, elev_g = grids_from_dem(xr.open_dataset(args.dem))

    print("[calib OOS] inférence CALIB…")
    pc, oc, kc = collect(args.checkpoint, met_vars, args.calib_dir, args.sencrop_dir, args.dem, stats, lat_g, lon_g, elev_g)
    print("[calib OOS] inférence TEST…")
    pt, ot, kt = collect(args.checkpoint, met_vars, args.test_dir, args.sencrop_dir, args.dem, stats, lat_g, lon_g, elev_g)

    # 1) biais médian par station, AJUSTÉ sur CALIB
    bias = {}
    for k in np.unique(kc):
        sel = kc == k
        if sel.sum() >= 5:
            bias[int(k)] = float(np.median(pc[sel] - oc[sel]))
    gbias = float(np.median(pc - oc))   # repli stations non vues
    # 2) APPLIQUÉ sur TEST
    corr = np.array([bias.get(int(k), gbias) for k in kt])
    pt_cal = pt - corr

    # 3) seuil de décision optimisé sur CALIB (best POD@FAR<0,20), appliqué TEST
    #    (le débiais s'applique aussi au calib pour cohérence)
    cc = pc - np.array([bias.get(int(k), gbias) for k in kc])
    op_cal = operating_points(cc, oc, args.thr)
    # tau optimal sur calib : balayage
    lab = oc < args.thr
    order = np.argsort(cc); ls = lab[order].astype(int)
    tp, fp = np.cumsum(ls), np.cumsum(1 - ls)
    far = fp / np.maximum(tp + fp, 1)
    taus = cc[order]
    ok = far < 0.20
    tau_star = float(taus[ok][np.argmax((tp / max(lab.sum(),1))[ok])]) if ok.any() else args.thr

    raw_test = operating_points(pt, ot, args.thr)          # test sans calibration
    cal_test = operating_points(pt_cal, ot, args.thr)       # test + calibration OOS
    # POD/FAR au seuil tau* (transféré de calib)
    declT = pt_cal <= tau_star; labT = ot < args.thr
    podT = int((declT & labT).sum()) / max(int(labT.sum()), 1)
    farT = int((declT & ~labT).sum()) / max(int((declT).sum()), 1)

    print(f"\n=== Calibration OUT-OF-SAMPLE (fit {args.calib_dir} → test {args.test_dir}) ===")
    print(f"  Test SANS calibration  : POD@0° {raw_test['pod']:.3f} FAR {raw_test['far']:.3f} | "
          f"AUC {raw_test['auc']:.3f} | POD@FAR<0,20 {raw_test['pod_at_far20']:.3f}")
    print(f"  Test + calibration OOS : POD@0° {cal_test['pod']:.3f} FAR {cal_test['far']:.3f} | "
          f"AUC {cal_test['auc']:.3f} | POD@FAR<0,20 {cal_test['pod_at_far20']:.3f}")
    print(f"  Point OPÉRATIONNEL (seuil τ*={tau_star:.2f}°C transféré de calib) : "
          f"POD {podT:.3f}  FAR {farT:.3f}")
    print(f"  (oracle in-sample calib pour référence : POD@FAR<0,20 {op_cal['pod_at_far20']:.3f})")


if __name__ == "__main__":
    main()
