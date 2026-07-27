"""Outils de calibration sur capteurs in situ (étage C) — **sans torch**.

Indépendant de la source (Netatmo, Sencrop) et de la méthode amont (U-Net CERRA,
Prithvi). Deux briques :

- ``assign_to_grid`` : plus-proche-voisin station → indices (ligne, colonne) sur
  la grille haute résolution.
- ``elevation_offset`` : ``dz = altitude_station − altitude_maille`` (m), pour la
  **correction lapse-rate** au point de calibration. C'est par ce ``dz`` que le
  MNT est valorisé dans le fine-tuning : une station en fond de vallée, 200 m
  sous l'altitude moyenne de sa maille 1 km, est comparée à la bonne altitude.

Voir ``docs/architecture.md`` (§4, étage C). Module léger (numpy/pandas) : il
n'importe pas torch, pour rester utilisable côté CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .netatmo_qc import NetatmoObs, tmin_nocturnal

StationObs = NetatmoObs  # conteneur générique (Netatmo / Sencrop / …)


def night_station_targets(
    obs_qc: StationObs,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    elevation_grid: np.ndarray | None = None,
    *,
    holdout_bbox: tuple[float, float, float, float] | None = None,
    role: str = "all",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Observations QC'd d'une nuit → cibles sparse ``(tmin, row, col, dz)``.

    Mutualise l'extraction entre les datasets de calibration (Prithvi / U-Net) :
    Tmin par station valide, leur position sur la grille HR, et le décalage
    d'altitude ``dz`` (m) pour la correction lapse-rate.

    Leave-station-out (parité avec le LOO Lot B #33) : si ``holdout_bbox``
    ``(lat_min, lat_max, lon_min, lon_max)`` est fourni et ``role != "all"`` :
    - ``role="train"`` : ne garde que les stations HORS de la bbox (fit sans elles) ;
    - ``role="val"``   : ne garde que les stations DEDANS (éval out-of-station).
    Défaut ``role="all"`` → toutes les stations (rétro-compat). Peut retourner des
    tableaux vides (nuit sans station dans le rôle) — l'appelant doit gérer.
    """
    tmin = tmin_nocturnal(obs_qc)
    valid = ~np.isnan(tmin.values)
    lat, lon = obs_qc.lat[valid], obs_qc.lon[valid]
    elev = obs_qc.elevation_m[valid]
    vals = tmin.values[valid].astype(np.float32)
    if holdout_bbox is not None and role != "all":
        la0, la1, lo0, lo1 = holdout_bbox
        inside = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        keep = inside if role == "val" else ~inside
        lat, lon, elev, vals = lat[keep], lon[keep], elev[keep], vals[keep]
    row, col = assign_to_grid(lat, lon, lat_grid, lon_grid)
    dz = elevation_offset(elev, row, col, elevation_grid)
    return vals, row, col, dz


def assign_to_grid(
    lat: np.ndarray, lon: np.ndarray, lat_grid: np.ndarray, lon_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Indices (row, col) de la maille la plus proche pour chaque station."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    row = np.argmin(np.abs(lat_grid[:, None] - lat[None, :]), axis=0)
    col = np.argmin(np.abs(lon_grid[:, None] - lon[None, :]), axis=0)
    return row, col


def elevation_offset(
    elevation_station: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    elevation_grid: np.ndarray | None,
) -> np.ndarray:
    """``dz = altitude_station − altitude_maille`` (m) par station.

    Si ``elevation_grid`` est ``None`` → zéros (pas de correction d'altitude).
    """
    elevation_station = np.asarray(elevation_station, dtype=np.float32)
    if elevation_grid is None:
        return np.zeros(len(elevation_station), dtype=np.float32)
    grid_elev = np.asarray(elevation_grid)[row, col]
    return (elevation_station - grid_elev).astype(np.float32)


def dataframe_to_station_obs(
    df: pd.DataFrame, date: str, bbox: dict[str, float] | None = None
) -> StationObs:
    """Construit un :class:`StationObs` à partir d'un DataFrame normalisé.

    Colonnes attendues : ``station_id, lat, lon, elevation_m, timestamp,
    t_celsius``. Filtre la nuit (20h → 08h+1) et pivote stations × heures.
    Mutualise la logique entre les loaders Netatmo et Sencrop.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    night_start = pd.Timestamp(date) + pd.Timedelta("20h")
    next_morning = pd.Timestamp(date) + pd.Timedelta("1D") + pd.Timedelta("8h")
    df = df[(df["timestamp"] >= night_start) & (df["timestamp"] < next_morning)]

    if bbox:
        df = df[
            (df["lat"] >= bbox["lat_min"])
            & (df["lat"] <= bbox["lat_max"])
            & (df["lon"] >= bbox["lon_min"])
            & (df["lon"] <= bbox["lon_max"])
        ]

    if df.empty:
        raise ValueError(f"Aucune observation station pour la nuit du {date}")

    pivot = df.pivot_table(
        index="station_id", columns="timestamp", values="t_celsius", aggfunc="mean"
    )
    meta = df.drop_duplicates("station_id").set_index("station_id")

    return StationObs(
        station_id=pivot.index.values,
        lat=meta.loc[pivot.index, "lat"].values,
        lon=meta.loc[pivot.index, "lon"].values,
        elevation_m=meta.loc[pivot.index, "elevation_m"].values,
        t_raw=pivot.values.astype(np.float32),
        times=pd.DatetimeIndex(pivot.columns),
    )
