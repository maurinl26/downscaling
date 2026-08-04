#!/usr/bin/env python3
"""QDM AROME-native: calibrate + MEASURED lift vs raw AROME (#105).

Predictor: AROME-HD (Open-Meteo) at Sencrop stations (backfill_arome_openmeteo.py).
Target:    Sencrop nightly Tmin (= the assimilated analysis at the station point).
Threshold: frost -2.2 C. Three levels, same station-nights, honest protocol:

  1. AROME raw               : the operational model, uncalibrated.
  2. QDM station-anchored    : per-station transfer trained on that station's OTHER
     (LOYO)                    years, applied to the held-out year. Regime "station
                               history present" -> detection lift.
  3. QDM generalized         : domain-wide transfer trained WITHOUT the test station
     (LOYO + LOSO)             -> ungauged parcel, obs-free. Calibration + (measured)
                               a real detection lift too, because quantile mapping
                               corrects the cold TAIL, not just the mean.

Empirical quantile mapping (rank of AROME in its CDF -> obs quantile). Leave-one-
year-out (forecast temporality) x leave-one-station-out (parcel generalization);
no leakage: the (AROME,obs) pair of the test station/year never enters its transfer.

Persists a metrics JSON (relates #88 / #89). Note the short-lead caveat of the
Open-Meteo historical series (#91): the lift is optimistic vs a strict J-1 24 h.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEF_SENC = "/Users/loicmaurin/kDrive/karpos_datasets/data/raw/sencrop"
DEF_AR = "/Users/loicmaurin/kDrive/karpos_datasets/output/arome_openmeteo_backfill/arome_hd_stations_2023-2025.parquet"
THR = -2.2


def night(d):
    return pd.Timestamp(d) + pd.Timedelta("20h"), pd.Timestamp(d) + pd.Timedelta(
        "1D"
    ) + pd.Timedelta("8h")


def qdm(train_p, train_o, test_p):
    tp, to = np.sort(train_p), np.sort(train_o)
    q = np.interp(test_p, tp, np.linspace(0, 1, len(tp)))
    return np.interp(q, np.linspace(0, 1, len(to)), to)


def scores(o, p):
    o, p = np.asarray(o), np.asarray(p)
    m = np.isfinite(p)
    o, p = o[m], p[m]
    op, pp = o <= THR, p <= THR
    vp, fp, fn = int(np.sum(op & pp)), int(np.sum(~op & pp)), int(np.sum(op & ~pp))
    return {
        "n": int(len(o)),
        "rmse": float(np.sqrt(np.mean((p - o) ** 2))),
        "bias": float(np.mean(p - o)),
        "pod": vp / (vp + fn) if vp + fn else None,
        "far": fp / (vp + fp) if vp + fp else None,
        "csi": vp / (vp + fp + fn) if vp + fp + fn else None,
    }


def load_obs(senc, years):
    cat = pd.read_csv(f"{senc}/stations_integrated.csv")  # noqa: F841 (kept for parity)
    rows = []
    for yr in years:
        sdf = pd.read_csv(sorted(glob.glob(f"{senc}/{yr}.csv/part-*.csv"))[0])
        sdf["timestamp"] = pd.to_datetime(
            sdf["timestamp"], utc=True, errors="coerce"
        ).dt.tz_localize(None)
        sdf = sdf[sdf["temperature_source"] == "station"]
        for d in pd.date_range(f"{yr}-01-01", f"{yr}-04-30", freq="D"):
            a, b = night(d)
            w = sdf[(sdf.timestamp >= a) & (sdf.timestamp <= b)]
            if w.empty:
                continue
            g = w.groupby("station_id")["temperature"].agg(["min", "count"])
            g = g[g["count"] >= 6]
            for st, r in g.iterrows():
                rows.append((str(st), pd.Timestamp(d), float(r["min"])))
    return pd.DataFrame(rows, columns=["station_id", "night", "obs_tmin"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sencrop", default=DEF_SENC)
    p.add_argument("--arome", default=DEF_AR)
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument(
        "--out-json",
        default=str(
            Path(__file__).resolve().parents[2] / "reports" / "qdm_arome_native" / "metrics.json"
        ),
    )
    a = p.parse_args()

    obs = load_obs(a.sencrop, a.years)
    ar = pd.read_parquet(a.arome)
    ar["station_id"] = ar.station_id.astype(str)
    df = ar.merge(obs, on=["station_id", "night"], how="inner")
    df["year"] = df.night.dt.year
    print(
        f"{len(df)} (AROME,obs) pairs · {df.station_id.nunique()} stations · years {sorted(df.year.unique())}"
    )

    anc = np.full(len(df), np.nan)
    gen = np.full(len(df), np.nan)
    for Y in sorted(df.year.unique()):
        for st in df.station_id.unique():
            te = df[(df.year == Y) & (df.station_id == st)]
            if te.empty:
                continue
            tra = df[(df.year != Y) & (df.station_id == st)]
            if len(tra) >= 30:
                anc[te.index] = qdm(
                    tra.arome_tmin.values, tra.obs_tmin.values, te.arome_tmin.values
                )
            trg = df[(df.year != Y) & (df.station_id != st)]
            if len(trg) >= 100:
                gen[te.index] = qdm(
                    trg.arome_tmin.values, trg.obs_tmin.values, te.arome_tmin.values
                )
    df["qdm_anchored"], df["qdm_general"] = anc, gen
    E = df.dropna(subset=["qdm_anchored", "qdm_general"])

    levels = {
        "arome_raw": scores(E.obs_tmin, E.arome_tmin),
        "qdm_anchored": scores(E.obs_tmin, E.qdm_anchored),
        "qdm_generalized": scores(E.obs_tmin, E.qdm_general),
    }
    b = levels["arome_raw"]
    lift = {
        k: {
            "pod_ratio": levels[k]["pod"] / b["pod"],
            "csi_from": b["csi"],
            "csi_to": levels[k]["csi"],
        }
        for k in ("qdm_anchored", "qdm_generalized")
    }
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = None
    out = {
        "issue": "maurinl26/karpos-downscaling#105",
        "threshold_C": THR,
        "protocol": "leave-one-year-out x leave-one-station-out (no leakage)",
        "predictor": "AROME-HD Open-Meteo at stations (short-lead, #91 caveat)",
        "target": "Sencrop nightly Tmin (assimilated analysis at station point)",
        "years": sorted(int(y) for y in E.year.unique()),
        "n_station_nights": int(len(E)),
        "n_stations": int(E.station_id.nunique()),
        "levels": levels,
        "lift_vs_arome_raw": lift,
        "git_sha": sha,
        "command": "uv run python -m downscaling.scripts.qdm_arome_native",
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(levels, indent=2))
    print(f"wrote {a.out_json}")
    for k in ("qdm_anchored", "qdm_generalized"):
        print(f"{k}: POD x{lift[k]['pod_ratio']:.1f}, CSI {b['csi']:.2f} -> {levels[k]['csi']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
