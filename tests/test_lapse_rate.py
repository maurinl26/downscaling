"""Tests de la correction lapse-rate (``downscaling.karpos_slr.lapse_rate``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import daily_series

from downscaling.karpos_slr.lapse_rate import (
    STANDARD_LAPSE_RATE,
    LapseRateCorrector,
    MonthlyLapseRate,
    correct_surface_pressure,
)


def _grid(value, ny=2, nx=2, name="z"):
    return xr.DataArray(
        np.full((ny, nx), float(value)),
        dims=["y", "x"],
        coords={"y": np.arange(ny), "x": np.arange(nx)},
        name=name,
    )


def test_scalar_correction_known_value():
    # MNT 200 m plus haut que la réanalyse, gamma = -6.5e-3 K/m
    t_coarse = daily_series([288.0, 290.0], extra_dims=(2, 2), name="t2m")
    z_coarse = _grid(800.0)
    z_fine = _grid(1000.0)
    corr = LapseRateCorrector(lapse_rate=STANDARD_LAPSE_RATE)
    out = corr.correct(t_coarse, z_coarse, z_fine)
    # dz = +200, correction = -6.5e-3 * 200 = -1.3 K
    np.testing.assert_allclose(out.isel(time=0).values, 288.0 - 1.3)
    np.testing.assert_allclose(out.isel(time=1).values, 290.0 - 1.3)
    assert out.attrs["lapse_rate_correction"] == "applied"


def test_monthly_gamma_aligns_on_calendar_month():
    # 12 valeurs mensuelles distinctes ; on vérifie l'alignement temporel.
    gamma_monthly = np.linspace(-0.01, -0.001, 12)  # K/m, par mois
    # Une valeur par mois (1er de chaque mois 2020)
    time = pd.date_range("2020-01-01", periods=12, freq="MS")
    t_coarse = xr.DataArray(
        np.full((12, 1, 1), 280.0),
        dims=["time", "y", "x"],
        coords={"time": time, "y": [0], "x": [0]},
        name="t2m",
    )
    z_coarse = _grid(0.0, ny=1, nx=1)
    z_fine = _grid(100.0, ny=1, nx=1)  # dz = +100
    out = LapseRateCorrector(lapse_rate=gamma_monthly).correct(t_coarse, z_coarse, z_fine)
    expected = 280.0 + gamma_monthly * 100.0
    np.testing.assert_allclose(out.squeeze().values, expected)


def test_monthly_lapse_rate_recovers_known_slope():
    # T = a + gamma*z exactement → la régression doit retrouver gamma.
    z = np.array([100.0, 500.0, 1000.0, 1500.0, 2000.0])
    true_gamma = -6.0e-3
    intercept = 290.0
    temps_one_month = intercept + true_gamma * z
    # 12 mois identiques
    station_temps = np.tile(temps_one_month, (12, 1))
    fitter = MonthlyLapseRate(station_altitudes=z, station_temps=station_temps)
    gamma = fitter.fit()
    assert gamma.shape == (12,)
    np.testing.assert_allclose(gamma, true_gamma, rtol=1e-6)
    # R² parfait
    np.testing.assert_allclose(fitter.r2_, 1.0, atol=1e-9)


def test_monthly_lapse_rate_fallback_when_too_few_stations():
    z = np.array([100.0, 500.0])  # seulement 2 stations
    station_temps = np.full((12, 2), 285.0)
    fitter = MonthlyLapseRate(station_altitudes=z, station_temps=station_temps)
    with pytest.warns(UserWarning):
        gamma = fitter.fit()
    np.testing.assert_allclose(gamma, STANDARD_LAPSE_RATE)


def test_monthly_lapse_rate_validates_shape():
    with pytest.raises(ValueError):
        MonthlyLapseRate(
            station_altitudes=np.array([100.0, 200.0, 300.0]),
            station_temps=np.full((12, 2), 285.0),  # 2 colonnes ≠ 3 altitudes
        )


def test_to_corrector_roundtrip():
    z = np.array([0.0, 1000.0, 2000.0])
    station_temps = np.tile(290.0 - 6.5e-3 * z, (12, 1))
    corrector = MonthlyLapseRate(z, station_temps).to_corrector()
    assert isinstance(corrector, LapseRateCorrector)
    assert isinstance(corrector.lapse_rate, np.ndarray)
    assert corrector.lapse_rate.shape == (12,)


def test_correct_surface_pressure_decreases_with_altitude():
    sp = daily_series([101325.0], extra_dims=(1, 1), name="sp")
    z_coarse = _grid(0.0, ny=1, nx=1)
    z_fine = _grid(1000.0, ny=1, nx=1)  # 1000 m plus haut → pression plus basse
    out = correct_surface_pressure(sp, z_coarse, z_fine, t_mean_k=288.15)
    g, Rd = 9.80665, 287.05
    expected = 101325.0 * np.exp(-g * 1000.0 / (Rd * 288.15))
    np.testing.assert_allclose(out.squeeze().values, expected)
    assert float(out.squeeze()) < 101325.0
    assert out.attrs["hypsometric_correction"] == "applied"
