"""Tests des indices d'assurance paramétrique (``downscaling.shared.indices``).

Cœur de valeur du produit : on valide chaque indice sur des valeurs connues
construites à la main, sans dépendance lourde (numpy/xarray uniquement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from downscaling.shared import indices as idx

from conftest import K0, c2k, daily_series, hourly_series


# ---------------------------------------------------------------------------
# Gel
# ---------------------------------------------------------------------------

def test_frost_days_counts_subzero_minima():
    # Tmin (°C) : 3 jours sous 0, 2 jours au-dessus
    tmin = daily_series(c2k([-5.0, -0.1, 2.0, 0.0, -3.0]))
    out = idx.frost_days(tmin, threshold_c=0.0, freq="YS")
    assert out.name == "frost_days"
    # -5, -0.1, -3 sont < 0 ; 2.0 et 0.0 ne le sont pas (seuil strict)
    assert int(out.sum()) == 3


def test_frost_hours_hourly_threshold():
    t2m = hourly_series(c2k([-2.0, -1.0, 0.5, 1.0, -0.5, 3.0]))
    out = idx.frost_hours(t2m, threshold_c=0.0, freq="YS")
    assert out.name == "frost_hours"
    assert int(out.sum()) == 3  # -2, -1, -0.5


def test_frost_days_custom_threshold():
    tmin = daily_series(c2k([-3.0, -1.5, -0.5, 1.0]))
    # seuil -2 °C : seul -3 passe
    out = idx.frost_days(tmin, threshold_c=-2.0, freq="YS")
    assert int(out.sum()) == 1


def test_spring_frost_index_zero_without_budburst():
    # GDD jamais atteint (températures sous la base) ⇒ aucun débourrement ⇒ indice 0
    n_hours = 24 * 30
    time_h = pd.date_range("2020-03-01", periods=n_hours, freq="h")
    t2m = xr.DataArray(
        np.full(n_hours, c2k(-10.0)), dims=["time"], coords={"time": time_h}, name="t2m"
    )
    tmin = daily_series(c2k(np.full(30, -10.0)), start="2020-03-01", name="tmin")
    out = idx.spring_frost_index(t2m, tmin, gdd_threshold=50.0, base_temp_c=5.0)
    assert out.name == "spring_frost_index"
    assert float(out.sum()) == 0.0


# ---------------------------------------------------------------------------
# Thermique / agronomie
# ---------------------------------------------------------------------------

def test_growing_degree_days_known_values():
    # tmean = (tmax+tmin)/2. base=10, cap=30 → contribution plafonnée à 20.
    tmax = daily_series(c2k([20.0, 40.0, 12.0]), name="tmax")
    tmin = daily_series(c2k([10.0, 20.0, 8.0]), name="tmin")
    out = idx.growing_degree_days(tmax, tmin, base_c=10.0, cap_c=30.0, freq="YS")
    # jour1 tmean=15 → 5 ; jour2 tmean=30 → 20 (cap) ; jour3 tmean=10 → 0
    assert int(out.sum()) == 25


def test_growing_degree_days_no_cap():
    tmax = daily_series(c2k([40.0]), name="tmax")
    tmin = daily_series(c2k([20.0]), name="tmin")
    out = idx.growing_degree_days(tmax, tmin, base_c=10.0, cap_c=None, freq="YS")
    # tmean=30, base=10 → 20 (pas de plafond)
    assert int(out.sum()) == 20


def test_heat_stress_days():
    tmax = daily_series(c2k([36.0, 35.0, 40.0, 20.0]), name="tmax")
    out = idx.heat_stress_days(tmax, threshold_c=35.0, freq="YS")
    # > 35 strict : 36 et 40
    assert int(out.sum()) == 2


def test_heatwave_index_zero_when_no_hot_days():
    tmax = daily_series(c2k(np.full(10, 20.0)), name="tmax")
    out = idx.heatwave_index(tmax, threshold_c=35.0, min_consecutive_days=3, freq="YS")
    assert int(out.sum()) == 0


def test_heatwave_index_detects_long_wave():
    # 7 jours consécutifs très chauds entourés de jours frais
    vals = [20.0, 20.0] + [38.0] * 7 + [20.0]
    tmax = daily_series(c2k(vals), name="tmax")
    out = idx.heatwave_index(tmax, threshold_c=35.0, min_consecutive_days=3, freq="YS")
    # Tous les jours de la vague sont comptés, pas seulement les fenêtres centrées.
    assert int(out.sum()) == 7


def test_heatwave_index_short_run_below_threshold():
    # Une séquence de 2 jours chauds (< N=3) ne compte pas ; une de 3 compte 3.
    vals = [38.0, 38.0, 20.0, 38.0, 38.0, 38.0, 20.0]
    tmax = daily_series(c2k(vals), name="tmax")
    out = idx.heatwave_index(tmax, threshold_c=35.0, min_consecutive_days=3, freq="YS")
    assert int(out.sum()) == 3


# ---------------------------------------------------------------------------
# Précipitations
# ---------------------------------------------------------------------------

def test_to_mm_conversion():
    tp_m = daily_series([0.001, 0.0, 0.02], name="tp")  # mètres
    out_mm = idx._to_mm(tp_m, "m")
    np.testing.assert_allclose(out_mm.values, [1.0, 0.0, 20.0])
    # déjà en mm → inchangé
    np.testing.assert_allclose(idx._to_mm(tp_m, "mm").values, tp_m.values)


def test_accumulated_precipitation_mm():
    tp = daily_series([0.001, 0.002, 0.003], name="tp")  # m → 1+2+3 = 6 mm
    out = idx.accumulated_precipitation(tp, unit="m", freq="YS")
    assert out.name == "accumulated_precip_mm"
    np.testing.assert_allclose(float(out.sum()), 6.0)


def test_extreme_precip_days():
    tp = daily_series([0.025, 0.010, 0.030, 0.020], name="tp")  # mm: 25,10,30,20
    out = idx.extreme_precip_days(tp, threshold_mm=20.0, unit="m", freq="YS")
    # > 20 strict : 25 et 30
    assert int(out.sum()) == 2


def test_dry_spell_days():
    tp = daily_series([0.0, 0.0005, 0.002, 0.0], name="tp")  # mm: 0, 0.5, 2, 0
    out = idx.dry_spell_days(tp, threshold_mm=1.0, unit="m", freq="YS")
    # < 1 mm : 0, 0.5, 0 → 3
    assert int(out.sum()) == 3


def test_max_consecutive_helper():
    arr = np.array([[1], [1], [0], [1], [1], [1], [0]])  # (time, 1)
    out = idx._max_consecutive_along_axis0(arr)
    assert out.shape == (1,)
    assert int(out[0]) == 3


def test_dry_spell_max_length():
    # mm : 0,0,2,0,0,0 → plus longue séquence sèche = 3 (les trois derniers)
    tp = daily_series([0.0, 0.0, 0.002, 0.0, 0.0, 0.0], name="tp")
    out = idx.dry_spell_max_length(tp, threshold_mm=1.0, unit="m", freq="YS")
    assert out.name == "dry_spell_max_length"
    assert int(out.max()) == 3


def test_r95p_sums_top_tail():
    # Distribution avec quelques extrêmes ; le p95 isole la queue haute.
    vals = ([0.0] * 90) + [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
    tp = daily_series(vals, name="tp")
    out = idx.r95p(tp, unit="m", freq="YS")
    assert out.name == "r95p_mm"
    # somme strictement positive et bornée par le cumul total
    assert float(out.sum()) > 0.0
    assert float(out.sum()) <= idx.accumulated_precipitation(tp).sum()


# ---------------------------------------------------------------------------
# Vent
# ---------------------------------------------------------------------------

def test_wind_speed_from_components_pythagore():
    u = daily_series([3.0, 0.0], name="u")
    v = daily_series([4.0, 5.0], name="v")
    ws = idx.wind_speed_from_components(u, v)
    np.testing.assert_allclose(ws.values, [5.0, 5.0])


def test_wind_storm_hours():
    ws = hourly_series([10.0, 16.0, 20.0, 14.0], name="ws")
    out = idx.wind_storm_hours(ws, threshold_ms=15.0, freq="YS")
    assert int(out.sum()) == 2  # 16, 20


# ---------------------------------------------------------------------------
# Neige (proxy)
# ---------------------------------------------------------------------------

def test_snowfall_proxy_days():
    tp = daily_series([0.002, 0.002, 0.0], name="tp")  # mm: 2, 2, 0
    t2m = daily_series(c2k([1.0, 5.0, 1.0]), name="t2m")  # °C: 1, 5, 1
    out = idx.snowfall_proxy_days(
        tp, t2m, temp_threshold_c=2.0, precip_threshold_mm=1.0, unit="m", freq="YS"
    )
    # précip>1 ET T<2 : jour1 (2mm, 1°C) ok ; jour2 (5°C) non ; jour3 (0mm) non
    assert int(out.sum()) == 1


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def test_compute_all_indices_produces_expected_variables():
    n = 24 * 10  # 10 jours horaires
    time = pd.date_range("2020-01-01", periods=n, freq="h")
    rng = np.arange(n, dtype=float)
    ds = xr.Dataset(
        {
            "t2m": ("time", c2k(5.0 + 5.0 * np.sin(rng / 12.0))),
            "tp": ("time", np.abs(np.sin(rng / 5.0)) * 0.001),
            "u10": ("time", np.full(n, 3.0)),
            "v10": ("time", np.full(n, 4.0)),
        },
        coords={"time": time},
    )
    out = idx.compute_all_indices(ds, unit_tp="m", freq="YS")
    expected = {
        "frost_days",
        "frost_hours",
        "heat_stress_days",
        "heatwave_days",
        "gdd",
        "accumulated_precip_mm",
        "extreme_precip_days",
        "dry_spell_days",
        "r95p_mm",
        "wind_storm_hours",
        "snowfall_proxy_days",
    }
    assert expected.issubset(set(out.data_vars))
