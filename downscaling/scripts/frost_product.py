#!/usr/bin/env python
"""
Évaluation PRODUIT gel — archi figée : CERRA 5,5 km + calibration station (OOS).

Le chiffre défendable du 15 juin, reproductible en une commande. Calibration station
out-of-sample (biais médian/station + seuil τ* ajustés sur 2022-2024, appliqués sur le
test pur 2025), aucune fuite. Émet un artefact métriques JSON + une figure ROC propre.

Archi : CERRA 5,5 km en entrée (maille native, nearest-cell aux stations) ; correction =
biais médian par station ; décision = Tmin ≤ τ*. Pas de U-Net (la maille CERRA + calib
atteint déjà la cible métier ; le U-Net station-supervisé reste un levier "parcelles non
instrumentées", hors périmètre du livrable figé).

Usage :
  uv run frost-product \
    --cerra-glob '/Users/loicmaurin/kDrive/karpos_datasets/data/raw/cerra/2m_temperature/*.nc' \
    --sencrop-dir data/sencrop --dem data/training/dem_attributes.nc --out-dir reports_product
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from downscaling.scripts.frost_eval_core import (
    grids_from_dem, stations, sample_cerra, roc, oos_calibrate, collect, to_c,
)

# Protocole FIGÉ (borné par la dispo Sencrop)
FIT_YEARS = ("2022", "2023", "2024")
TEST_YEAR = "2025"
FAR_MAX, POD_MIN = 0.20, 0.75


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _load_cerra(cerra_glob: str) -> xr.DataArray:
    import glob
    parts = []
    for p in sorted(glob.glob(cerra_glob)):
        d = xr.open_dataset(p)["t2m"]
        tdim = "valid_time" if "valid_time" in d.dims else "time"
        parts.append(d.rename({tdim: "time"}))
    da = xr.concat(parts, dim="time").sortby("time")
    return da.drop_vars("expver") if "expver" in da.coords else da


def main() -> None:
    ap = argparse.ArgumentParser(description="Évaluation produit gel (CERRA + calibration station)")
    ap.add_argument("--cerra-glob", required=True)
    ap.add_argument("--sencrop-dir", default="data/sencrop")
    ap.add_argument("--dem", default="data/training/dem_attributes.nc")
    ap.add_argument("--out-dir", default="reports_product")
    ap.add_argument("--thr", type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lat_g, lon_g, _ = grids_from_dem(xr.open_dataset(args.dem))
    print("[produit] chargement CERRA…")
    da = _load_cerra(args.cerra_glob).load()

    import glob
    sencrop_dates = sorted(Path(p).stem.replace("sencrop_", "")
                           for p in glob.glob(f"{args.sencrop_dir}/sencrop_*.csv"))
    # nuits CERRA dispo = celles couvertes par le champ chargé
    t0 = str(da.time.values.min())[:10]; t1 = str(da.time.values.max())[:10]
    dates = [d for d in sencrop_dates if t0 <= d <= t1]
    fit = [d for d in dates if d[:4] in FIT_YEARS]
    test = [d for d in dates if d[:4] == TEST_YEAR]
    print(f"[produit] nuits : fit {len(fit)} ({'/'.join(FIT_YEARS)}) | test {len(test)} ({TEST_YEAR})")

    fn = lambda d, la, lo: sample_cerra(da, d, la, lo)
    pc, oc, kc = collect(fn, fit, args.sencrop_dir)
    pt, ot, kt = collect(fn, test, args.sencrop_dir)
    pt_cal, tau = oos_calibrate(pc, oc, kc, pt, kt, args.thr)

    def summarize(pred):
        tpr, fpr, far, auc = roc(pred, ot, args.thr)
        pod_at = float(tpr[1:][far[1:] < FAR_MAX].max()) if (far[1:] < FAR_MAX).any() else float("nan")
        return dict(auc=round(auc, 3), pod_at_far20=round(pod_at, 3)), (tpr, fpr, far)

    raw_m, raw_c = summarize(pt)
    cal_m, cal_c = summarize(pt_cal)
    decl = pt_cal <= tau; lab = ot < args.thr
    pod_op = int((decl & lab).sum()) / max(int(lab.sum()), 1)
    far_op = int((decl & ~lab).sum()) / max(int(decl.sum()), 1)

    metrics = {
        "product": "frost-downscaling-cerra",
        "version": "v0.3",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "architecture": "CERRA 5,5 km (maille native) + calibration station OOS (biais médian/station + seuil τ*)",
        "protocol": {"fit": "+".join(FIT_YEARS), "test": TEST_YEAR, "reserved": "2026"},
        "n_test_obs": int(len(ot)), "n_test_frost": int((ot < args.thr).sum()),
        "metrics": {
            "raw": raw_m,
            "calibrated": cal_m,
            "operational": {"tau_star_C": round(tau, 3), "pod": round(pod_op, 3), "far": round(far_op, 3)},
        },
        "targets": {"far_max": FAR_MAX, "pod_min": POD_MIN},
        "target_reached_envelope": bool(cal_m["pod_at_far20"] >= POD_MIN),
    }
    (out / "frost_product_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # figure produit (CERRA seul : RAW vs +calibration + point opérationnel)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    for (tpr, fpr, far), m, lab_, col, ls in [
        (raw_c, raw_m, "CERRA 5,5 km RAW", "#999999", "--"),
        (cal_c, cal_m, "CERRA 5,5 km + calibration station", "#d62728", "-"),
    ]:
        axL.plot(fpr, tpr, ls, lw=2.2, color=col, label=f"{lab_} (AUC {m['auc']:.3f})")
        axR.plot(far[1:], tpr[1:], ls, lw=2.2, color=col,
                 label=f"{lab_} (POD@FAR<.20={m['pod_at_far20']:.2f})")
    axR.plot([far_op], [pod_op], "*", ms=20, color="#d62728", mec="k", mew=0.8, zorder=5,
             label=f"Point opérationnel τ*={tau:.2f}°C (POD {pod_op:.2f}/FAR {far_op:.2f})")
    axL.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
    axL.set(xlabel="FPR", ylabel="POD", title=f"ROC — détection gel 0°C (test pur {TEST_YEAR})")
    axL.legend(fontsize=8, loc="lower right"); axL.grid(alpha=0.3)
    axR.axvspan(0, FAR_MAX, color="green", alpha=0.07)
    axR.axhline(POD_MIN, color="green", ls=":", lw=1); axR.axvline(FAR_MAX, color="green", ls=":", lw=1)
    axR.set(xlabel="FAR", ylabel="POD", xlim=(0, 1), ylim=(0, 1),
            title="POD vs FAR (métier) — zone cible verte")
    axR.legend(fontsize=8, loc="lower right"); axR.grid(alpha=0.3)
    fig.suptitle("Produit gel — CERRA 5,5 km + calibration station · Drôme-Ardèche",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / "frost_product_roc.png", dpi=130)

    print(f"\n✓ {out}/frost_product_metrics.json")
    print(f"✓ {out}/frost_product_roc.png")
    print(json.dumps(metrics["metrics"], indent=2, ensure_ascii=False))
    print(f"Cible métier (enveloppe ROC) atteinte : {metrics['target_reached_envelope']}")


if __name__ == "__main__":
    main()
