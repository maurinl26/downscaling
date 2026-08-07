"""Tests des primitives cible stress thermique (Tmax diurne, #110).

Fenêtre diurne, agrégateur tmax_daytime, dispatcher reduce, plafond QC. Rétro-
compatibilité : défaut = nuit/min (gel inchangé).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from downscaling.prtihvi_wxc.netatmo_qc import (
    NetatmoObs,
    reduce_station_target,
    time_window_bounds,
    tmax_daytime,
    tmin_nocturnal,
)
from downscaling.prtihvi_wxc.stations import dataframe_to_station_obs, night_station_targets


def _obs() -> NetatmoObs:
    # 2 stations × 3 heures ; station 0 [10,-2,4] station 1 [20,25,15].
    return NetatmoObs(
        station_id=np.array(["A", "B"]),
        lat=np.array([44.5, 44.6]),
        lon=np.array([5.0, 5.1]),
        elevation_m=np.array([300.0, 200.0]),
        t_raw=np.array([[10.0, -2.0, 4.0], [20.0, 25.0, 15.0]], dtype=np.float32),
        times=pd.DatetimeIndex(pd.date_range("2022-07-18 12:00", periods=3, freq="1h")),
    )


class TestTimeWindow:
    def test_night_default(self):
        start, end = time_window_bounds("2021-04-27")
        assert start == pd.Timestamp("2021-04-27 20:00")
        assert end == pd.Timestamp("2021-04-28 08:00")

    def test_day_window(self):
        start, end = time_window_bounds("2022-07-18", "day")
        assert start == pd.Timestamp("2022-07-18 06:00")
        assert end == pd.Timestamp("2022-07-18 20:00")

    def test_unknown_raises(self):
        try:
            time_window_bounds("2022-07-18", "afternoon")
        except ValueError:
            return
        raise AssertionError("fenêtre inconnue devrait lever")


class TestAggregators:
    def test_tmin_vs_tmax(self):
        obs = _obs()
        np.testing.assert_allclose(tmin_nocturnal(obs).values, [-2.0, 15.0])
        np.testing.assert_allclose(tmax_daytime(obs).values, [10.0, 25.0])

    def test_reduce_dispatch(self):
        obs = _obs()
        np.testing.assert_allclose(reduce_station_target(obs, "min").values, [-2.0, 15.0])
        np.testing.assert_allclose(reduce_station_target(obs, "max").values, [10.0, 25.0])

    def test_reduce_default_is_min(self):
        np.testing.assert_allclose(reduce_station_target(_obs()).values, [-2.0, 15.0])


class TestNightStationTargets:
    def test_reduce_max_selects_tmax(self):
        obs = _obs()
        lat_grid = np.array([44.4, 44.5, 44.6, 44.7])
        lon_grid = np.array([4.9, 5.0, 5.1, 5.2])
        vals_min, *_ = night_station_targets(obs, lat_grid, lon_grid, reduce="min")
        vals_max, *_ = night_station_targets(obs, lat_grid, lon_grid, reduce="max")
        np.testing.assert_allclose(sorted(vals_min), [-2.0, 15.0])
        np.testing.assert_allclose(sorted(vals_max), [10.0, 25.0])


class TestDataframeWindow:
    def _df(self) -> pd.DataFrame:
        # Une station, points à 03h (nuit) et 15h (jour) le 2022-07-18.
        base = pd.Timestamp("2022-07-18")
        return pd.DataFrame(
            {
                "station_id": ["S", "S"],
                "lat": [44.5, 44.5],
                "lon": [5.0, 5.0],
                "elevation_m": [300.0, 300.0],
                "timestamp": [base + pd.Timedelta("3h"), base + pd.Timedelta("15h")],
                "t_celsius": [12.0, 34.0],
            }
        )

    def test_day_window_keeps_afternoon(self):
        # Fenêtre jour (06h-20h) du 18 → garde 15h (34°C), exclut 03h.
        obs = dataframe_to_station_obs(self._df(), "2022-07-18", window="day")
        np.testing.assert_allclose(tmax_daytime(obs).values, [34.0])

    def test_night_window_excludes_afternoon(self):
        # Fenêtre nuit (20h D -> 08h D+1) : ni 03h du 18 ni 15h du 18 ne tombent
        # dedans → aucune obs → ValueError (comportement gel inchangé).
        try:
            dataframe_to_station_obs(self._df(), "2022-07-18", window="night")
        except ValueError:
            return
        raise AssertionError("la fenêtre nuit ne devrait garder aucune obs ici")
