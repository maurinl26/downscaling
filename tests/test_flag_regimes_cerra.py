"""Tests du chemin CERRA de flag_regimes (mapping variables, Td-depuis-RH, R1)."""

import numpy as np
import pytest
import xarray as xr

from downscaling.scripts.flag_regimes import (
    _classify,
    _night_features_cerra,
    _resolve_var,
    _td_from_rh,
)


def test_td_from_rh_saturation():
    # RH = 1.0 ⇒ Td = T (saturation), exact par construction Magnus.
    t = np.array([10.0, -3.0, 0.0])
    td = _td_from_rh(t, np.ones_like(t))
    assert np.allclose(td, t, atol=1e-6)


def test_td_from_rh_subsaturated_below_temp():
    # RH < 1 ⇒ Td < T, et dépression positive.
    t = np.array([10.0])
    td = _td_from_rh(t, np.array([0.5]))
    assert td[0] < t[0]
    assert (t[0] - td[0]) > 3.0  # air à 50% RH nettement sous-saturé


def _cerra_night(si10, tcc_pct, msl_pa, r2_pct, t2m_k):
    lat = np.array([45.5, 45.0, 44.5, 44.0])
    lon = np.array([4.0, 4.5, 5.0, 5.5])
    tt = np.array([0, 1, 2], dtype="int64")
    shape = (3, 4, 4)

    def full(v):
        return (("time", "lat", "lon"), np.full(shape, v, dtype=float))

    return xr.Dataset(
        {
            "si10": full(si10),
            "wdir10": full(20.0),
            "msl": full(msl_pa),
            "tcc": full(tcc_pct),
            "r2": full(r2_pct),
            "t2m": full(t2m_k),
        },
        coords={"time": tt, "lat": lat, "lon": lon},
    )


def test_resolve_var_finds_and_raises():
    ds = _cerra_night(1.0, 10.0, 102000.0, 60.0, 270.0)
    assert _resolve_var(ds, "wind_speed").name == "si10"
    with pytest.raises(ValueError, match="mslp"):
        _resolve_var(ds.drop_vars("msl"), "mslp")


def test_cerra_features_radiative_is_R1():
    # Nuit radiative : vent faible, ciel clair (tcc 10%), anticyclone (1020 hPa).
    ds = _cerra_night(si10=1.0, tcc_pct=10.0, msl_pa=102000.0, r2_pct=60.0, t2m_k=270.0)
    bbox = {"lat_min": 44.0, "lat_max": 45.5, "lon_min": 4.0, "lon_max": 5.5}
    f = _night_features_cerra(ds, bbox)
    # Unités auto-détectées : tcc % → fraction, msl Pa → hPa, t2m K → °C.
    assert f["tcc_med"] == pytest.approx(0.10, abs=1e-6)
    assert f["mslp_med"] == pytest.approx(1020.0, abs=0.1)
    assert f["t2m_med"] == pytest.approx(-3.15, abs=0.1)
    assert f["wind_med"] == pytest.approx(1.0, abs=1e-6)
    assert f["dewpoint_dep_med"] > 0.0
    assert _classify(f) == "R1"
