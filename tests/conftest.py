"""Fixtures et utilitaires partagés pour la suite de tests.

Les helpers construisent des ``xr.DataArray`` synthétiques avec un axe ``time``
réaliste (pandas ``date_range``), ce dont dépendent indices et transfos
statistiques pour leurs ``resample`` / ``groupby``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

K0 = 273.15


def daily_series(values, start="2020-01-01", name="t", extra_dims=None):
    """DataArray 1-D (time) ou (time, y, x) à pas journalier.

    ``values`` peut être une liste/array 1-D (time) ; ``extra_dims`` ajoute des
    dimensions spatiales en répétant la série (utile pour vérifier le broadcast).
    """
    values = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=values.shape[0], freq="D")
    if extra_dims is None:
        return xr.DataArray(values, dims=["time"], coords={"time": time}, name=name)
    ny, nx = extra_dims
    arr = np.repeat(values[:, None, None], ny, axis=1).repeat(nx, axis=2)
    return xr.DataArray(
        arr,
        dims=["time", "y", "x"],
        coords={"time": time, "y": np.arange(ny), "x": np.arange(nx)},
        name=name,
    )


def hourly_series(values, start="2020-01-01", name="t"):
    """DataArray 1-D (time) à pas horaire."""
    values = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=values.shape[0], freq="h")
    return xr.DataArray(values, dims=["time"], coords={"time": time}, name=name)


def c2k(celsius):
    """°C → K."""
    return np.asarray(celsius, dtype=float) + K0
