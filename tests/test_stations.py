"""Tests de la calibration capteurs (étage C) — sans torch ni réseau.

Couvre l'assignation station→grille, le décalage d'altitude (``obs_dz``, qui
valorise le MNT au point de calibration) et le loader Sencrop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from downscaling.prtihvi_wxc.sencrop import SENCROP_COLUMNS, load_sencrop, tmin_nocturnal
from downscaling.prtihvi_wxc.stations import assign_to_grid, elevation_offset


def test_assign_to_grid_nearest_neighbour():
    lat_grid = np.array([44.0, 44.5, 45.0])
    lon_grid = np.array([4.0, 4.5, 5.0])
    # stations proches de (44.5, 5.0) et (44.0, 4.0)
    row, col = assign_to_grid(np.array([44.48, 44.02]), np.array([4.97, 4.01]), lat_grid, lon_grid)
    assert list(row) == [1, 0]
    assert list(col) == [2, 0]


def test_elevation_offset_meters():
    elevation_grid = np.array([[100.0, 200.0], [300.0, 400.0]])
    row = np.array([0, 1])
    col = np.array([1, 0])
    # station1 à 250 m sur maille 200 m → +50 ; station2 à 280 m sur maille 300 m → −20
    dz = elevation_offset(np.array([250.0, 280.0]), row, col, elevation_grid)
    np.testing.assert_allclose(dz, [50.0, -20.0])


def test_elevation_offset_none_is_zero():
    dz = elevation_offset(np.array([250.0, 280.0]), np.array([0, 1]), np.array([0, 1]), None)
    np.testing.assert_array_equal(dz, [0.0, 0.0])


def _sencrop_csv(tmp_path):
    """Écrit un export Sencrop synthétique (schéma natif) pour une nuit."""
    night = pd.Timestamp("2021-04-27")
    rows = []
    for sid, (lat, lon, alt, t) in {
        "dev-A": (44.6, 4.9, 320.0, -1.5),
        "dev-B": (44.7, 5.0, 180.0, 2.0),
    }.items():
        for h in (21, 23, 2, 5):  # heures nocturnes
            ts = night + pd.Timedelta(hours=h if h >= 20 else 24 + h)
            rows.append(
                {
                    SENCROP_COLUMNS["station_id"]: sid,
                    SENCROP_COLUMNS["lat"]: lat,
                    SENCROP_COLUMNS["lon"]: lon,
                    SENCROP_COLUMNS["elevation_m"]: alt,
                    SENCROP_COLUMNS["timestamp"]: ts,
                    SENCROP_COLUMNS["t_celsius"]: t + 0.1 * h,  # variation horaire
                }
            )
    path = tmp_path / "sencrop_2021-04-27.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_sencrop_maps_native_schema(tmp_path):
    obs = load_sencrop(_sencrop_csv(tmp_path), "2021-04-27")
    assert set(obs.station_id) == {"dev-A", "dev-B"}
    # élévation correctement remontée (sert au dz de calibration)
    elev = dict(zip(obs.station_id, obs.elevation_m, strict=True))
    assert elev["dev-A"] == 320.0 and elev["dev-B"] == 180.0
    # Tmin nocturne par station (agrégat partagé avec Netatmo)
    tmin = tmin_nocturnal(obs)
    assert tmin["dev-A"] < tmin["dev-B"]  # station froide en altitude
