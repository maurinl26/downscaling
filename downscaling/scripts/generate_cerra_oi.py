"""CERRA-OI : surface assimilée = background CERRA t2m (min-nuit) + incrément OI
terrain-aware vers Sencrop (issue #99). PAS de SURFEX : background = réanalyse
CERRA téléchargée, incrément = OI Python (downscaling.prtihvi_wxc.optimal_interpolation).

Produit cerra_oi_<year>.nc (t2m, grille DEM 1 km), convention de nuit alignée sur
l'entraînement (d 20h -> d+1 8h). --eval-loo sort le RMSE honnête (leave-station-out,
non-circulaire) sur les nuits de gel, vs in-sample.

    uv run python -m downscaling.scripts.generate_cerra_oi \
        --cerra-atm cerra_atm_2025.nc --dem dem_attributes_svf.nc \
        --sencrop <root> --out cerra_oi_2025.nc --year 2025 --eval-loo 2025-02-13,14,15,16
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from downscaling.prtihvi_wxc.optimal_interpolation import terrain_aware_oi, leave_one_out_stations

BBOX = dict(lat_min=44.0, lat_max=45.5, lon_min=4.0, lon_max=5.5)


def _night(d):  # convention entraînement : d 20h -> d+1 8h
    return pd.Timestamp(d) + pd.Timedelta("20h"), pd.Timestamp(d) + pd.Timedelta("1D") + pd.Timedelta("8h")


def _sencrop_night(sdf, cat, alt, d):
    a, b = _night(d)
    w = sdf[(sdf.timestamp >= a) & (sdf.timestamp <= b)]
    s = (w.groupby("station_id")["temperature"].min().reset_index()
         .merge(cat, left_on="station_id", right_on="bucket_id", how="inner")
         .dropna(subset=["latitude", "longitude", "temperature"]))
    s = s[(s.latitude >= BBOX["lat_min"]) & (s.latitude <= BBOX["lat_max"]) &
          (s.longitude >= BBOX["lon_min"]) & (s.longitude <= BBOX["lon_max"])]
    return s.latitude.values, s.longitude.values, s.temperature.values, (s[alt].values if alt else None)


def main():
    p = argparse.ArgumentParser(description="Génère la surface CERRA-OI (assimilation Python, pas SURFEX)")
    p.add_argument("--cerra-atm", type=Path, required=True)
    p.add_argument("--dem", type=Path, required=True)
    p.add_argument("--sencrop", type=str, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--months", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--eval-loo", type=str, default=None, help="dates gel CSV pour le RMSE LOO honnête")
    a = p.parse_args()

    dem = xr.open_dataset(a.dem)
    klat = (dem["lat"] if "lat" in dem else dem["latitude"]).values
    klon = (dem["lon"] if "lon" in dem else dem["longitude"]).values
    elev = np.asarray(dem["elevation"].values)
    if elev.shape != (klat.size, klon.size):
        elev = elev.T
    from scipy.interpolate import RegularGridInterpolator, griddata
    elev_i = RegularGridInterpolator((klat, klon), elev, bounds_error=False, fill_value=None)
    KLON, KLAT = np.meshgrid(klon, klat)

    c = xr.open_dataset(a.cerra_atm)
    ct = pd.DatetimeIndex(pd.to_datetime(c["valid_time"].values, utc=True)).tz_localize(None)
    clat, clon = c["latitude"].values, c["longitude"].values
    casc = clat[0] < clat[-1]
    CLON, CLAT = np.meshgrid(clon, clat)

    part = sorted(glob.glob(f"{a.sencrop}/{a.year}.csv/part-*.csv"))[0]
    sdf = pd.read_csv(part)
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    sdf = sdf[sdf["temperature_source"] == "station"]
    cat = pd.read_csv(f"{a.sencrop}/stations_integrated.csv")
    alt = "altitude_m" if "altitude_m" in cat.columns else ("altitude" if "altitude" in cat.columns else None)

    def bg_nightmin(d):
        s, e = _night(d)
        m = (ct >= s) & (ct <= e)
        if m.sum() == 0:
            return None
        fld = np.nanmin(c["t2m"].values[m], axis=0) - 273.15
        g = griddata(np.column_stack([CLON.ravel(), CLAT.ravel()]),
                     (fld if casc else fld[::-1]).ravel(), (KLON, KLAT), method="linear")
        bad = ~np.isfinite(g)
        if bad.any():
            g[bad] = griddata(np.column_stack([KLON[~bad], KLAT[~bad]]), g[~bad], (KLON[bad], KLAT[bad]), method="nearest")
        return g

    def stations(d):
        sla, slo, sv, sa = _sencrop_night(sdf, cat, alt, d)
        if sa is None or not np.isfinite(sa).all():
            filled = elev_i(np.column_stack([sla, slo]))
            sa = filled if sa is None else np.where(np.isfinite(sa), sa, filled)
        return sla, slo, sv, sa

    # --- eval LOO honnête ---
    if a.eval_loo:
        allm_in, allm_loo, allo = [], [], []
        for d in a.eval_loo.split(","):
            d = d.strip()
            bg = bg_nightmin(d)
            sla, slo, sv, sa = stations(d)
            if bg is None or len(sv) < 5:
                continue
            ana = terrain_aware_oi(bg, klat, klon, elev, sla, slo, sa, sv)
            m_in = RegularGridInterpolator((klat, klon), ana, bounds_error=False, fill_value=None)(np.column_stack([sla, slo]))
            m_loo = leave_one_out_stations(bg, klat, klon, elev, sla, slo, sa, sv)
            allm_in.append(m_in); allm_loo.append(m_loo); allo.append(sv)
        o = np.concatenate(allo); mi = np.concatenate(allm_in); ml = np.concatenate(allm_loo)
        r = lambda m: float(np.sqrt(np.nanmean((m - o) ** 2)))
        print(f"[CERRA-OI eval] {len(o)} station-nuits | in-sample RMSE={r(mi):.2f} (circulaire) | "
              f"LOO RMSE={r(ml):.2f} (honnête, non-circulaire) | bias_loo={np.nanmean(ml-o):+.2f}")

    # --- génération surface pleine année ---
    dates = pd.date_range(f"{a.year}-{a.months[0]:02d}-01", periods=120, freq="D")
    dates = [d for d in dates if d.month in a.months]
    fields, kept = [], []
    for d in dates:
        bg = bg_nightmin(d)
        if bg is None:
            continue
        sla, slo, sv, sa = stations(d)
        surf = terrain_aware_oi(bg, klat, klon, elev, sla, slo, sa, sv) if len(sv) >= 5 else bg
        fields.append(surf.astype("f4")); kept.append(d)
    xr.Dataset({"t2m": (("time", "latitude", "longitude"), np.stack(fields))},
               coords={"time": pd.DatetimeIndex(kept), "latitude": klat, "longitude": klon}).to_netcdf(a.out)
    print(f"[CERRA-OI] wrote {a.out} : {len(kept)} nights")


if __name__ == "__main__":
    main()
