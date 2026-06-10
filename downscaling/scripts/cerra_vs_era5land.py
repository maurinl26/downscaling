#!/usr/bin/env python
"""
RAW CERRA 5,5 km vs RAW ERA5-Land 9 km — à calibration station ÉGALE.

Question : la vraie cible CERRA 5,5 km bat-elle ERA5-Land 9 km au point de
fonctionnement métier (POD@FAR<0,20) une fois la calibration capteurs appliquée ?

Procédure identique pour les deux sources :
  1. échantillonnage nearest-cell du Tmin nocturne à chaque station Sencrop ;
  2. biais médian par station ajusté sur CALIB (2022-2024) ;
  3. appliqué sur TEST (2025) + seuil τ* transféré ;
  4. ROC / POD@FAR<0,20 / point opérationnel.

Nuit = D 20:00 → D+1 07:00. CERRA (3-horaire) : min sur {D21, D+1 00/03/06}.

Usage :
  uv run python downscaling/scripts/cerra_vs_era5land.py \
    --cerra-glob '/Users/loicmaurin/kDrive/karpos_datasets/data/raw/cerra/2m_temperature/*.nc' \
    --era5land-dir data/cerra_fine_mv --sencrop-dir data/sencrop \
    --dem data/training/dem_attributes.nc --out cerra_vs_era5land_2025.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from downscaling.scripts.frost_eval_core import (
    LAPSE, grids_from_dem, to_c, stations, sample_cerra, sample_era5land,
    roc, oos_calibrate, collect,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="RAW CERRA vs ERA5-Land, calibration égale")
    ap.add_argument("--cerra-glob", required=True)
    ap.add_argument("--era5land-dir", required=True)
    ap.add_argument("--sencrop-dir", default="data/sencrop")
    ap.add_argument("--dem", default="data/training/dem_attributes.nc")
    ap.add_argument("--out", default="cerra_vs_era5land_2025.png")
    ap.add_argument("--thr", type=float, default=0.0)
    args = ap.parse_args()

    lat_g, lon_g, _ = grids_from_dem(xr.open_dataset(args.dem))

    print("[cmp] chargement CERRA…")
    parts = []
    for p in sorted(glob.glob(args.cerra_glob)):
        d = xr.open_dataset(p)["t2m"]
        tdim = "valid_time" if "valid_time" in d.dims else "time"
        parts.append(d.rename({tdim: "time"}))
    da = xr.concat(parts, dim="time").sortby("time")
    if "expver" in da.coords:
        da = da.drop_vars("expver")

    all_dates = sorted(Path(p).stem.replace("sencrop_", "")
                       for p in glob.glob(f"{args.sencrop_dir}/sencrop_*.csv"))
    e5_dates = set(Path(p).stem.replace("era5land_", "")
                   for p in glob.glob(f"{args.era5land_dir}/era5land_*.nc"))
    dates = [d for d in all_dates if d in e5_dates]      # nuits communes
    calib = [d for d in dates if d[:4] in ("2022", "2023", "2024")]
    test = [d for d in dates if d[:4] == "2025"]
    print(f"[cmp] nuits : calib {len(calib)} | test {len(test)}")

    cerra_fn = lambda d, la, lo: sample_cerra(da, d, la, lo)
    e5_fn = lambda d, la, lo: sample_era5land(
        f"{args.era5land_dir}/era5land_{d}.nc", lat_g, lon_g, la, lo)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    rows = []
    cfg = [("CERRA 5,5 km", cerra_fn, "#d62728"), ("ERA5-Land 9 km", e5_fn, "#1f77b4")]
    for name, fn, col in cfg:
        print(f"[cmp] {name} : échantillonnage calib + test…")
        pc, oc, kc = collect(fn, calib, args.sencrop_dir)
        pt, ot, kt = collect(fn, test, args.sencrop_dir)
        pt_cal, tau = oos_calibrate(pc, oc, kc, pt, kt, args.thr)
        for tag, p, ls in [("RAW", pt, "--"), ("+ calib station", pt_cal, "-")]:
            tpr, fpr, far, auc = roc(p, ot, args.thr)
            pod = float(tpr[1:][far[1:] < 0.20].max()) if (far[1:] < 0.20).any() else float("nan")
            lw = 2.4 if tag.startswith("+") else 1.4
            axL.plot(fpr, tpr, ls, lw=lw, color=col, alpha=0.95 if ls == "-" else 0.6,
                     label=f"{name} {tag} (AUC {auc:.3f})")
            axR.plot(far[1:], tpr[1:], ls, lw=lw, color=col, alpha=0.95 if ls == "-" else 0.6,
                     label=f"{name} {tag} (POD@FAR<.20={pod:.2f})")
            rows.append({"source": name, "variant": tag, "auc": round(auc, 3),
                         "pod_at_far20": round(pod, 3), "n_obs": int(len(ot)),
                         "n_frost": int((ot < args.thr).sum())})
        # point opérationnel (calibré, τ* transféré)
        decl = pt_cal <= tau; lab = ot < args.thr
        pod_op = int((decl & lab).sum()) / max(int(lab.sum()), 1)
        far_op = int((decl & ~lab).sum()) / max(int(decl.sum()), 1)
        axR.plot([far_op], [pod_op], "*", ms=18, color=col, mec="k", mew=0.8, zorder=5,
                 label=f"{name} opérationnel τ*={tau:.2f}°C (POD {pod_op:.2f}/FAR {far_op:.2f})")
        rows.append({"source": name, "variant": "opérationnel", "auc": "",
                     "pod_at_far20": round(pod_op, 3), "n_obs": f"FAR={far_op:.3f}",
                     "n_frost": f"tau={tau:.2f}"})

    axL.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
    axL.set(xlabel="FPR", ylabel="POD", title="ROC — détection gel 0°C (test 2025)")
    axL.legend(fontsize=7, loc="lower right"); axL.grid(alpha=0.3)
    axR.axvspan(0, 0.20, color="green", alpha=0.07)
    axR.axhline(0.75, color="green", ls=":", lw=1); axR.axvline(0.20, color="green", ls=":", lw=1)
    axR.set(xlabel="FAR", ylabel="POD", xlim=(0, 1), ylim=(0, 1),
            title="POD vs FAR (métier) — zone cible verte")
    axR.legend(fontsize=7, loc="lower right"); axR.grid(alpha=0.3)
    fig.suptitle("CERRA 5,5 km vs ERA5-Land 9 km — à calibration station égale",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(args.out, dpi=130)

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(f"\n✓ {args.out}  +  {Path(args.out).with_suffix('.csv')}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
