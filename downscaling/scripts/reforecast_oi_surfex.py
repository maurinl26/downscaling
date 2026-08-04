#!/usr/bin/env python3
"""Reforecast OI-on-SURFEX (max-perf, obs-in-loop) — serving demonstrator (#105).

The best-perf Karpos field: physical SURFEX T2m background + terrain-aware OI
assimilation of Sencrop + smoothing, on the 1 km grid. Obs-in-loop -> this is the
RAPPORT / suivi regime (not a live obs-free alert). Served on /alerte as a
"suivi assimilé" demonstrator over the reference nights SURFEX actually covers.

Reforecast mode: runs on available SURFEX output (offline runs), currently the
February 2025 frost episode (SURF_ATM_DIAGNOSTICS.OUT.nc, 2025-02-13 -> 02-17).

Pipeline per night (d 20h -> d+1 08h):
  SURFEX T2M hourly -> night-min on SURFEX grid -> regrid to 1 km target grid
  -> terrain_aware_oi(background, Sencrop) -> Gaussian smooth (#103)
Output: canonical serving zarr with t2m_karpos_sr (+ background, n_stations, elevation).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter

from downscaling.prtihvi_wxc.optimal_interpolation import terrain_aware_oi

DEF_SURFEX = "/Users/loicmaurin/open-SURFEX-V9-1-0/domains/drome"
DEF_DATA = "/Users/loicmaurin/kDrive/karpos_datasets/output/regen_cerra_2023"
DEF_SENC = "/Users/loicmaurin/kDrive/karpos_datasets/data/raw/sencrop"
DEF_KZ = "/Users/loicmaurin/kDrive/karpos_datasets/output/karpos_sr_multiyear/2025.zarr"
DEF_OUT = "/Users/loicmaurin/kDrive/karpos_datasets/output/karpos_sr_serving/reforecast_surfex_feb2025.zarr"


def night(d):
    return pd.Timestamp(d) + pd.Timedelta("20h"), pd.Timestamp(d) + pd.Timedelta(
        "1D"
    ) + pd.Timedelta("8h")


def smooth(field, glat, sigma_km):
    res_km = float(np.mean(np.abs(np.diff(glat)))) * 111.0
    return gaussian_filter(field, sigma=max(sigma_km / res_km, 1e-6), mode="nearest")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surfex-dir", default=DEF_SURFEX)
    p.add_argument("--diag", default="SURF_ATM_DIAGNOSTICS.OUT.nc")
    p.add_argument("--data", default=DEF_DATA)
    p.add_argument("--sencrop", default=DEF_SENC)
    p.add_argument("--grid-zarr", default=DEF_KZ)
    p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--smooth-km", type=float, default=1.5)
    p.add_argument("--min-stations", type=int, default=3)
    a = p.parse_args()

    # target 1 km grid + terrain (for the OI decorrelation)
    kz = xr.open_zarr(a.grid_zarr)
    klat, klon = kz["latitude"].values, kz["longitude"].values
    KLON, KLAT = np.meshgrid(klon, klat)
    elev = np.asarray(xr.open_dataset(f"{a.data}/dem_attributes_svf.nc")["elevation"].values)
    if elev.shape != (klat.size, klon.size):
        elev = elev.T
    ei = RegularGridInterpolator((klat, klon), elev, bounds_error=False, fill_value=None)

    # SURFEX background
    sfx = xr.open_dataset(f"{a.surfex_dir}/{a.diag}")
    pgd = xr.open_dataset(f"{a.surfex_dir}/PGD.nc")
    slat = np.asarray(pgd["LAT"].values)
    slon = np.asarray(pgd["LON"].values)
    st = pd.DatetimeIndex(pd.to_datetime(sfx["time"].values))
    t2 = np.asarray(sfx["T2M"].values)  # (time, yy, xx), Kelvin or Celsius?
    if np.nanmedian(t2) > 100:  # Kelvin -> Celsius
        t2 = t2 - 273.15
    gpts = np.column_stack([slon.ravel(), slat.ravel()])
    ok = np.isfinite(gpts).all(1)

    cat = pd.read_csv(f"{a.sencrop}/stations_integrated.csv")
    alt = "altitude_m" if "altitude_m" in cat.columns else "altitude"
    yr = int(st[0].year)
    sdf = pd.read_csv(sorted(glob.glob(f"{a.sencrop}/{yr}.csv/part-*.csv"))[0])
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], utc=True, errors="coerce").dt.tz_localize(
        None
    )
    sdf = sdf[sdf["temperature_source"] == "station"]

    # nights fully covered by SURFEX
    d0, d1 = st.min().normalize(), st.max().normalize()
    times, A_an, A_bg, A_ns = [], [], [], []
    for d in pd.date_range(d0, d1, freq="D"):
        aa, bb = night(d)
        if aa < st.min() or bb > st.max():
            continue
        m = (st >= aa) & (st <= bb)
        if m.sum() < 6:
            continue
        snm = np.nanmin(t2[m], axis=0).ravel()  # SURFEX night-min on SURFEX grid
        bg = griddata(gpts[ok], snm[ok], (KLON, KLAT), method="linear")
        bad = ~np.isfinite(bg)
        if bad.any():
            bg[bad] = griddata(
                np.column_stack([KLON[~bad], KLAT[~bad]]),
                bg[~bad],
                (KLON[bad], KLAT[bad]),
                method="nearest",
            )
        w = sdf[(sdf.timestamp >= aa) & (sdf.timestamp <= bb)]
        s = (
            w.groupby("station_id")["temperature"]
            .min()
            .reset_index()
            .merge(cat, left_on="station_id", right_on="bucket_id")
            .dropna(subset=["latitude", "longitude", "temperature"])
        )
        s = s[(s.latitude >= 44) & (s.latitude <= 45.5) & (s.longitude >= 4) & (s.longitude <= 5.5)]
        n = len(s)
        if n >= a.min_stations:
            sla, slo_, sv = s.latitude.values, s.longitude.values, s.temperature.values
            sa = np.where(
                np.isfinite(s[alt].values), s[alt].values, ei(np.column_stack([sla, slo_]))
            )
            an = smooth(
                terrain_aware_oi(bg, klat, klon, elev, sla, slo_, sa, sv), klat, a.smooth_km
            )
        else:
            an = bg.copy()
        times.append(pd.Timestamp(d))
        A_an.append(an.astype("float32"))
        A_bg.append(bg.astype("float32"))
        A_ns.append(n)

    ds = xr.Dataset(
        {
            "t2m_karpos_sr": (("time", "y", "x"), np.stack(A_an)),
            "t2m_background": (("time", "y", "x"), np.stack(A_bg)),
            "n_stations": (("time",), np.array(A_ns, dtype="int16")),
            "elevation": (("y", "x"), elev.astype("float32")),
        },
        coords={
            "time": pd.DatetimeIndex(times),
            "latitude": (("y",), klat),
            "longitude": (("x",), klon),
        },
        attrs={
            "title": "Karpos OI-on-SURFEX reforecast (max-perf, obs-in-loop) — serving demo #105",
            "regime": "RAPPORT / suivi assimilé (obs-in-loop, NOT a live obs-free alert)",
            "method": "SURFEX T2M night-min background + terrain-aware OI (Sencrop, HxV) + smooth #103",
            "night": "d 20h -> d+1 08h",
            "threshold_frost_C": -2.2,
            "surfex_source": a.diag,
        },
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(a.out, mode="w")
    print(
        f"wrote {a.out} · {ds.sizes['time']} nights {[str(t.date()) for t in times]} "
        f"· stations/night={A_ns} · t2m_karpos_sr min={float(np.stack(A_an).min()):.1f}"
    )


if __name__ == "__main__":
    main()
