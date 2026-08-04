#!/usr/bin/env python3
"""Backfill AROME (Open-Meteo Historical Forecast API) at Sencrop stations.

Issue maurinl26/karpos-downscaling#105 (QDM AROME-native, voie B of #70).

The Meteo-France real-time bucket (OVH `meteofrance-pnt`, used by karpos-engine
`scripts/ingest_arome.py`) keeps only ~2 weeks of runs: past frost seasons are
NOT retrievable there. Open-Meteo's Historical Forecast API archives past
operational AROME runs and is the only public source reaching 2023-2025.

Source retained: ``models=arome_france_hd`` (~1.5 km, closest public proxy to the
native 1.3 km grid). Coverage probed 2026-08: arome_france_hd starts ~2022-12
(2023, 2024, 2025 OK; 2022 unavailable everywhere).

Caveat (documented on #105/#91): the historical-forecast series is assembled at
SHORT lead (freshest run per hour), not a controlled +12 h / J-1 at 24 h. The
QDM lift measured on it is therefore optimistic vs a strict day-ahead forecast.

Output: parquet ``(station_id, night, arome_tmin, lat, lon)``, the QDM predictor.
Night-min = min of temperature_2m over 20h(d) -> 08h(d+1), UTC.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
DEF_SENC = "/Users/loicmaurin/kDrive/karpos_datasets/data/raw/sencrop"
DEF_OUT = "/Users/loicmaurin/kDrive/karpos_datasets/output/arome_openmeteo_backfill/arome_hd_stations_2023-2025.parquet"


def night(d: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(d) + pd.Timedelta("20h"), pd.Timestamp(d) + pd.Timedelta(
        "1D"
    ) + pd.Timedelta("8h")


def fetch(la, lo, d0, d1, model):
    q = urllib.parse.urlencode(
        {
            "latitude": ",".join(f"{x:.4f}" for x in la),
            "longitude": ",".join(f"{x:.4f}" for x in lo),
            "hourly": "temperature_2m",
            "models": model,
            "start_date": d0,
            "end_date": d1,
            "timezone": "UTC",
        }
    )
    with urllib.request.urlopen(f"{API}?{q}", timeout=180) as r:
        return json.load(r)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sencrop", default=DEF_SENC)
    p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--model", default="arome_france_hd")
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[44.0, 45.5, 4.0, 5.5],
        metavar=("LAT0", "LAT1", "LON0", "LON1"),
    )
    p.add_argument("--min-hours", type=int, default=6, help="min covered hours per night")
    a = p.parse_args()

    cat = pd.read_csv(f"{a.sencrop}/stations_integrated.csv").dropna(
        subset=["latitude", "longitude"]
    )
    la0, la1, lo0, lo1 = a.bbox
    cat = cat[
        (cat.latitude >= la0)
        & (cat.latitude <= la1)
        & (cat.longitude >= lo0)
        & (cat.longitude <= lo1)
    ]
    cat = cat.drop_duplicates(subset=["bucket_id"]).reset_index(drop=True)
    sid = cat["bucket_id"].astype(str).values
    lat, lon = cat["latitude"].values, cat["longitude"].values
    print(f"{len(sid)} stations in bbox")

    rows = []
    for yr in a.years:
        res = fetch(lat, lon, f"{yr}-01-01", f"{yr}-04-30", a.model)
        res = res if isinstance(res, list) else [res]
        for k, loc in enumerate(res):
            h = loc.get("hourly", {})
            t = pd.to_datetime(h.get("time", []))
            v = np.array(
                [np.nan if x is None else x for x in h.get("temperature_2m", [])], dtype=float
            )
            if len(t) == 0 or not np.isfinite(v).any():
                continue
            df = pd.DataFrame({"t": t, "v": v})
            hr = df.t.dt.hour
            df["night"] = df.t.dt.normalize()
            df.loc[hr <= 8, "night"] = df.t.dt.normalize() - pd.Timedelta("1D")
            df = df[(hr >= 20) | (hr <= 8)].dropna(subset=["v"])
            g = df.groupby("night")["v"].agg(["min", "count"])
            g = g[g["count"] >= a.min_hours]
            for nn, r in g.iterrows():
                rows.append((sid[k], pd.Timestamp(nn), float(r["min"]), lat[k], lon[k]))
        print(f"{yr}: {len(rows)} pairs cumulated")

    out = pd.DataFrame(rows, columns=["station_id", "night", "arome_tmin", "lat", "lon"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(
        f"wrote {a.out} · {len(out)} pairs · {out.station_id.nunique()} stations · "
        f"{out.night.min().date()}..{out.night.max().date()}"
    )


if __name__ == "__main__":
    main()
