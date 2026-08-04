#!/usr/bin/env python3
"""Generate the Sencrop-assimilated analysis surface (QDM target, #105).

For each night of the frost seasons (Jan-Apr), produces on the 1 km grid:
  t2m_analysis   = OI terrain-aware(CERRA night-min background, Sencrop), smoothed
  t2m_background = CERRA night-min regridded to 1 km (reference)
  n_stations     = number of Sencrop stations assimilated that night (QC)

This is the TRUTH (analysis) surface that the AROME-native QDM (#105) targets. It
is independent of the forecast model: CERRA is the background here, AROME is the
SOURCE on the QDM side (not the target). The complete field carries cold-pool
structure between stations, which the terrain-generalized QDM learns from.

Method: OI terrain-aware (horizontal L_h=15 km AND elevation L_z=150 m
decorrelation), then Gaussian post-smoothing (#103, sigma~1.5 km) for field
coherence without skill loss. Night = d 20h -> d+1 08h. Threshold -2.2 C.

Output: canonical zarr (time, y, x).
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

DEF_DATA = "/Users/loicmaurin/kDrive/karpos_datasets/output/regen_cerra_2023"
DEF_SENC = "/Users/loicmaurin/kDrive/karpos_datasets/data/raw/sencrop"
DEF_KZ = "/Users/loicmaurin/kDrive/karpos_datasets/output/karpos_sr_multiyear/2025.zarr"
DEF_OUT = "/Users/loicmaurin/kDrive/karpos_datasets/output/karpos_analysis_assimilee/analysis_2022-2025.zarr"


def night(d):
    return pd.Timestamp(d) + pd.Timedelta("20h"), pd.Timestamp(d) + pd.Timedelta(
        "1D"
    ) + pd.Timedelta("8h")


def smooth(field, glat, sigma_km):
    """Local inline of #103 Gaussian smoothing (keeps #105 independent of PR #104)."""
    res_km = float(np.mean(np.abs(np.diff(glat)))) * 111.0
    return gaussian_filter(field, sigma=max(sigma_km / res_km, 1e-6), mode="nearest")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEF_DATA)
    p.add_argument("--sencrop", default=DEF_SENC)
    p.add_argument("--grid-zarr", default=DEF_KZ)
    p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    p.add_argument("--smooth-km", type=float, default=1.5)
    p.add_argument("--min-stations", type=int, default=3)
    a = p.parse_args()

    kz = xr.open_zarr(a.grid_zarr)
    klat, klon = kz["latitude"].values, kz["longitude"].values
    KLON, KLAT = np.meshgrid(klon, klat)
    elev = np.asarray(xr.open_dataset(f"{a.data}/dem_attributes_svf.nc")["elevation"].values)
    if elev.shape != (klat.size, klon.size):
        elev = elev.T
    ei = RegularGridInterpolator((klat, klon), elev, bounds_error=False, fill_value=None)
    cat = pd.read_csv(f"{a.sencrop}/stations_integrated.csv")
    alt = "altitude_m" if "altitude_m" in cat.columns else "altitude"

    times, A_an, A_bg, A_ns = [], [], [], []
    for yr in a.years:
        c = xr.open_dataset(f"{a.data}/cerra_atm_{yr}.nc")
        ct = pd.DatetimeIndex(pd.to_datetime(c["valid_time"].values, utc=True)).tz_localize(None)
        clat, clon = c["latitude"].values, c["longitude"].values
        casc = clat[0] < clat[-1]
        CLON, CLAT = np.meshgrid(clon, clat)
        cv = c["t2m"].values
        sdf = pd.read_csv(sorted(glob.glob(f"{a.sencrop}/{yr}.csv/part-*.csv"))[0])
        sdf["timestamp"] = pd.to_datetime(
            sdf["timestamp"], utc=True, errors="coerce"
        ).dt.tz_localize(None)
        sdf = sdf[sdf["temperature_source"] == "station"]
        for d in pd.date_range(f"{yr}-01-01", f"{yr}-04-30", freq="D"):
            aa, bb = night(d)
            m = (ct >= aa) & (ct <= bb)
            if m.sum() == 0:
                continue
            cnm = np.nanmin(cv[m], axis=0) - 273.15
            bg = griddata(
                np.column_stack([CLON.ravel(), CLAT.ravel()]),
                (cnm if casc else cnm[::-1]).ravel(),
                (KLON, KLAT),
                method="linear",
            )
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
            s = s[
                (s.latitude >= 44)
                & (s.latitude <= 45.5)
                & (s.longitude >= 4)
                & (s.longitude <= 5.5)
            ]
            n = len(s)
            if n >= a.min_stations:
                sla, slo, sv = s.latitude.values, s.longitude.values, s.temperature.values
                sa = np.where(
                    np.isfinite(s[alt].values), s[alt].values, ei(np.column_stack([sla, slo]))
                )
                an = smooth(
                    terrain_aware_oi(bg, klat, klon, elev, sla, slo, sa, sv), klat, a.smooth_km
                )
            else:
                an = bg.copy()
            times.append(pd.Timestamp(d))
            A_an.append(an.astype("float32"))
            A_bg.append(bg.astype("float32"))
            A_ns.append(n)

    ds = xr.Dataset(
        {
            "t2m_analysis": (("time", "y", "x"), np.stack(A_an)),
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
            "title": "Karpos Sencrop-assimilated analysis (QDM target #105)",
            "method": "OI terrain-aware (HxV, L_h=15km L_z=150m) on CERRA night-min background, "
            f"Gaussian smoothed sigma={a.smooth_km}km (#103)",
            "night": "d 20h -> d+1 08h",
            "threshold_frost_C": -2.2,
        },
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(a.out, mode="w")
    frost = (ds.t2m_analysis <= -2.2).any(("y", "x")).sum().item()
    print(
        f"wrote {a.out} · {ds.sizes['time']} nights · stations/night med={int(np.median(A_ns))} "
        f"· frost nights in analysis={frost}"
    )


if __name__ == "__main__":
    main()
