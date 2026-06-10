#!/usr/bin/env python
"""
Cœur d'évaluation gel — helpers SANS torch (numpy/pandas/xarray seulement).

Échantillonnage maille→station, ROC vectorisée, calibration station out-of-sample.
Partagé par `frost_product.py` (produit CERRA) et `cerra_vs_era5land.py` (comparaison).
Aucune dépendance deep-learning : le produit CERRA + calibration station n'utilise pas
de réseau, donc son évaluation doit tourner sans la stack torch/Lightning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

LAPSE = -4.0e-3  # °C/m, gradient adiabatique sec


def grids_from_dem(dem: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lat_grid 1D, lon_grid 1D, elevation 2D)`` du MNT, tolérant aux noms/dims."""
    def _axis(names, axis):
        for n in names:
            if n in dem.coords or n in dem.variables:
                arr = np.asarray(dem[n].values)
                return arr if arr.ndim == 1 else (arr[:, 0] if axis == 0 else arr[0, :])
        return None

    elevation = np.asarray(dem["elevation"].values)
    ny, nx = elevation.shape[-2:]
    lat_grid = _axis(("lat", "latitude", "y"), axis=0)
    lon_grid = _axis(("lon", "longitude", "x"), axis=1)
    if lat_grid is None:
        lat_grid = np.arange(ny)
    if lon_grid is None:
        lon_grid = np.arange(nx)
    return lat_grid, lon_grid, elevation


def to_c(a):
    """Kelvin→°C si la moyenne trahit des Kelvin, sinon identité."""
    return a - 273.15 if np.nanmean(a) > 100 else a


def stations(sencrop_dir, date):
    """Agrège une nuit Sencrop par station : lat/lon/alt + Tmin nocturne observé."""
    df = pd.read_csv(f"{sencrop_dir}/sencrop_{date}.csv")
    return df.groupby("device_id").agg(
        lat=("latitude", "first"), lon=("longitude", "first"),
        alt=("altitude", "first"), tmin=("temperature", "min")).reset_index()


def sample_cerra(da, date, lat, lon):
    """Tmin CERRA nocturne (3-horaire) au nearest-cell. Nuit D 20:00 → D+1 07:00."""
    d = np.datetime64(date)
    sub = da.sel(time=slice(d + np.timedelta64(20, "h"), d + np.timedelta64(31, "h")))
    if sub.time.size == 0:
        return np.full(len(lat), np.nan)
    field = to_c(sub.min("time").values)
    latv, lonv = da.latitude.values, da.longitude.values
    i = np.abs(latv[None, :] - lat[:, None]).argmin(1)
    j = np.abs(lonv[None, :] - lon[:, None]).argmin(1)
    return field[i, j]


def sample_era5land(path, lat_g, lon_g, lat, lon):
    """Tmin ERA5-Land (tuile 1 km horaire) au nearest-cell, grille régulière 1D."""
    ds = xr.open_dataset(path)
    field = to_c(ds["t2m"].min("time").values)
    i = np.abs(lat_g[None, :] - lat[:, None]).argmin(1)
    j = np.abs(lon_g[None, :] - lon[:, None]).argmin(1)
    return field[i, j]


def roc(pred, obs, thr=0.0):
    """ROC vectorisée : déclare gel si Tmin prédit ≤ τ. Retourne tpr, fpr, far, auc."""
    label = obs < thr
    P, N = int(label.sum()), int((~label).sum())
    o = np.argsort(pred, kind="mergesort")
    lab = label[o].astype(np.int64)
    tp, fp = np.cumsum(lab), np.cumsum(1 - lab)
    tpr = np.concatenate([[0.0], tp / max(P, 1)])
    fpr = np.concatenate([[0.0], fp / max(N, 1)])
    far = np.concatenate([[1.0], fp / np.maximum(tp + fp, 1)])
    return tpr, fpr, far, float(np.trapz(tpr, fpr))


def oos_calibrate(pc, oc, kc, pt, kt, thr=0.0, far_max=0.20):
    """Biais médian/station fit calib → applique test ; seuil τ* (best POD@FAR<far_max) transféré."""
    bias, gb = {}, float(np.median(pc - oc))
    for k in np.unique(kc):
        s = kc == k
        if s.sum() >= 5:
            bias[k] = float(np.median(pc[s] - oc[s]))
    pt_cal = pt - np.array([bias.get(k, gb) for k in kt])
    cc = pc - np.array([bias.get(k, gb) for k in kc])
    lab = oc < thr
    order = np.argsort(cc); ls = lab[order].astype(int)
    tp, fp = np.cumsum(ls), np.cumsum(1 - ls)
    far = fp / np.maximum(tp + fp, 1)
    ok = far < far_max
    tau = float(cc[order][ok][np.argmax((tp / max(lab.sum(), 1))[ok])]) if ok.any() else thr
    return pt_cal, tau


def collect(source_fn, dates, sencrop_dir):
    """(pred Tmin °C, obs Tmin, clé station) sur les nuits demandées."""
    P, O, K = [], [], []
    for d in dates:
        st = stations(sencrop_dir, d)
        pa = source_fn(d, st.lat.values, st.lon.values)
        obs, key = st.tmin.values, st.device_id.values
        m = ~np.isnan(pa) & ~np.isnan(obs)
        P.append(pa[m]); O.append(obs[m]); K.append(key[m])
    return np.concatenate(P), np.concatenate(O), np.concatenate(K)
