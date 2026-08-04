"""Tests du loader RADOME + de son merge holdout-aware dans le dataset KarposSR.

Valide : (1) parse décimale virgule FR + mapping DATE(YYYYMMDD)→nuit + fallback
TNSOL ; (2) le merge RADOME dans BulkSencropDataset respecte le holdout
(role=train exclut les RADOME dans la bbox ; role=val n'ajoute aucun RADOME).
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from downscaling.deep_learning.radome_loader import load_radome_targets

BBOX = (44.2, 44.3, 5.2, 5.3)  # holdout Baronnies synthétique


def _write_radome_fixture(tmp_path):
    obs_root = tmp_path / "quotidienne" / "2023"
    sdir = obs_root / "station=26050001"
    sdir.mkdir(parents=True)
    # CSV MF : séparateur ';', décimales virgule ; TN vide le 05 → fallback TNSOL.
    (sdir / "2023_frost.csv").write_text(
        "POSTE;DATE;TN;QTN;TNSOL\n26050001;20230404;-2,2;1;-4,0\n26050001;20230405;;1;-3,1\n"
    )
    cat = tmp_path / "catalogue_2023.csv"
    pd.DataFrame(
        {
            "station_id": [26050001],
            "name": ["BESIGNAN"],
            "lat": [44.32],
            "lon": [5.31],
            "alt_m": [1445],
        }
    ).to_csv(cat, index=False)
    return str(obs_root), str(cat)


def test_load_radome_french_decimal_date_and_fallback(tmp_path):
    obs_root, cat = _write_radome_fixture(tmp_path)
    targets = load_radome_targets(obs_root, cat)

    assert set(targets) == {"2023-04-04", "2023-04-05"}
    lat, lon, alt, tn = targets["2023-04-04"][0]
    assert (lat, lon, alt) == (44.32, 5.31, 1445.0)
    assert tn == pytest.approx(-2.2)  # décimale FR "-2,2" → -2.2
    # 05 : TN vide → fallback TNSOL "-3,1"
    assert targets["2023-04-05"][0][3] == pytest.approx(-3.1)


# ---------------------------------------------------------------------------
# Merge holdout-aware dans le dataset (torch requis pour BulkSencropDataset)
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")

from downscaling.prtihvi_wxc.netatmo_qc import NetatmoObs  # noqa: E402
from downscaling.scripts import recalibrate_dl_film as R  # noqa: E402

_LAT_GRID = np.linspace(43.9, 44.7, 24)
_LON_GRID = np.linspace(4.8, 5.5, 24)


def _sencrop_obs(*_a, **_k) -> NetatmoObs:
    # 2 stations Sencrop, toutes deux HORS de la bbox holdout.
    return NetatmoObs(
        station_id=np.arange(2),
        lat=np.array([44.00, 44.50]),
        lon=np.array([5.00, 5.00]),
        elevation_m=np.full(2, 200.0),
        t_raw=np.array([[-3.0, -3.0], [0.0, 0.0]], dtype=np.float32),
        times=pd.date_range("2023-04-04 20:00", periods=2, freq="h"),
    )


def _make_dataset(monkeypatch, role):
    monkeypatch.setattr(R, "load_sencrop", _sencrop_obs)
    radome_map = {
        "2023-04-04": [
            (44.25, 5.25, 1400.0, -5.0),  # DEDANS la bbox → exclu en train
            (44.60, 5.40, 300.0, -1.0),  # dehors → gardé en train
        ]
    }
    ds = R.BulkSencropDataset(
        dates=["2023-04-04"],
        coarse_provider=lambda d: (torch.zeros(5, 8, 8), torch.zeros(4, 8, 8)),
        sencrop_root="dummy",
        lat_grid=_LAT_GRID,
        lon_grid=_LON_GRID,
        elevation_grid=np.zeros((24, 24), dtype=np.float32),
        min_stations=1,
        holdout_bbox=BBOX,
        role=role,
        radome_map=radome_map,
    )
    ds.qc = types.SimpleNamespace(run=lambda o: o)  # bypass QC : obs synthétique tel quel
    return ds


def test_radome_merge_train_excludes_inbbox(monkeypatch):
    ds = _make_dataset(monkeypatch, role="train")
    n = ds[0]["obs_tmin"].numel()
    # sencrop hors-bbox (2) + RADOME hors-bbox (1) ; RADOME in-bbox exclu.
    assert n == 3


def test_radome_merge_val_adds_no_radome(monkeypatch):
    ds = _make_dataset(monkeypatch, role="val")
    # val = Sencrop DANS la bbox (0 ici) ; aucun RADOME ajouté.
    assert ds[0]["obs_tmin"].numel() == 0


def test_radome_merge_all_takes_everything(monkeypatch):
    ds = _make_dataset(monkeypatch, role="all")
    # sencrop (2) + RADOME (2), aucun filtrage.
    assert ds[0]["obs_tmin"].numel() == 4
