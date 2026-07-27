"""Smoke tests du leave-station-out Lot C (#33) — sans GPU.

Valide : (1) le filtre holdout de ``night_station_targets`` (train = hors bbox,
val = dans bbox, all = tout) ; (2) le role-split du ``UNetSparseDataModule``
(train/val = mêmes nuits, stations disjointes, pas de fuite) ; (3) la garde
"nuit sans station dans le rôle" dans les steps ; (4) un fast_dev_run holdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from downscaling.prtihvi_wxc.netatmo_qc import NetatmoObs
from downscaling.prtihvi_wxc.stations import night_station_targets

# bbox holdout : lat[44.2,44.3] lon[5.2,5.3] → stations 0 et 1 dedans, 2-5 dehors.
BBOX = (44.2, 44.3, 5.2, 5.3)
_LAT = np.array([44.25, 44.25, 44.00, 44.50, 44.60, 44.10])
_LON = np.array([5.25, 5.22, 5.00, 5.00, 5.40, 4.90])
_LAT_GRID = np.linspace(43.9, 44.7, 24)
_LON_GRID = np.linspace(4.8, 5.5, 24)


def _obs_night() -> NetatmoObs:
    # t_raw (6 stations, 3 heures nocturnes) : Tmin par station = -3..+2 °C.
    tmin_per_station = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    t_raw = np.repeat(tmin_per_station[:, None], 3, axis=1)
    return NetatmoObs(
        station_id=np.arange(6),
        lat=_LAT,
        lon=_LON,
        elevation_m=np.full(6, 200.0),
        t_raw=t_raw,
        times=pd.date_range("2023-04-04 20:00", periods=3, freq="h"),
    )


def test_night_station_targets_holdout_partitions_stations():
    obs = _obs_night()
    v_all, *_ = night_station_targets(obs, _LAT_GRID, _LON_GRID)
    v_val, *_ = night_station_targets(obs, _LAT_GRID, _LON_GRID, holdout_bbox=BBOX, role="val")
    v_train, *_ = night_station_targets(obs, _LAT_GRID, _LON_GRID, holdout_bbox=BBOX, role="train")
    assert len(v_all) == 6
    assert len(v_val) == 2  # stations 0,1 dans la bbox
    assert len(v_train) == 4  # stations 2,3,4,5 hors bbox
    assert len(v_val) + len(v_train) == len(v_all)
    # val = les deux stations froides (-3, -2), disjointes du train
    assert set(np.round(v_val, 1)) == {-3.0, -2.0}
    assert set(np.round(v_val, 1)).isdisjoint(set(np.round(v_train, 1)))


def test_role_all_is_backward_compatible():
    obs = _obs_night()
    a, *_ = night_station_targets(obs, _LAT_GRID, _LON_GRID)
    b, *_ = night_station_targets(obs, _LAT_GRID, _LON_GRID, holdout_bbox=None, role="train")
    assert len(a) == len(b) == 6  # sans bbox → aucun filtrage quel que soit le rôle


# ---------------------------------------------------------------------------
# DataModule + steps (torch requis)
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")
pytest.importorskip("lightning.pytorch")

import lightning.pytorch as pl  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from downscaling.deep_learning.model import build_model  # noqa: E402
from downscaling.deep_learning.sparse_calibration import (  # noqa: E402
    UNetSparseCalibrationModule,
    UNetSparseDataModule,
    unet_sparse_collate,
)

MET_CH, DEM_CH, SIZE = 5, 4, 16
_N_BY_ROLE = {"train": 4, "val": 2, "all": 6}


class _HoldoutDataset(Dataset):
    """Nuits synthétiques role-aware : train=4 / val=2 / all=6 stations disjointes."""

    def __init__(self, role: str = "all"):
        self.holdout_bbox = BBOX
        self.role = role
        self._gen = torch.Generator().manual_seed(0)

    def with_role(self, role: str) -> _HoldoutDataset:
        import copy

        clone = copy.copy(self)
        clone.role = role
        return clone

    def __len__(self) -> int:
        return 4

    def __getitem__(self, i: int) -> dict:
        n = _N_BY_ROLE[self.role]
        return {
            "x_met": torch.randn(MET_CH, SIZE, SIZE, generator=self._gen),
            "x_dem": torch.randn(DEM_CH, SIZE, SIZE, generator=self._gen),
            "obs_tmin": torch.randn(n, generator=self._gen),
            "obs_row": torch.randint(0, SIZE, (n,), generator=self._gen),
            "obs_col": torch.randint(0, SIZE, (n,), generator=self._gen),
            "obs_dz": torch.randn(n, generator=self._gen) * 100.0,
            "date": "2023-04-04",
        }


def _toy_unet():
    return build_model(architecture="srcnn", met_in_ch=MET_CH, dem_in_ch=DEM_CH)


def test_datamodule_role_split_no_leak():
    dm = UNetSparseDataModule(_HoldoutDataset(), num_workers=0)
    dm.setup()
    # role-split (pas random_split) : les deux vues portent un rôle explicite.
    assert dm.train_ds.role == "train"
    assert dm.val_ds.role == "val"
    train_obs = dm.train_ds[0]["obs_tmin"].numel()
    val_obs = dm.val_ds[0]["obs_tmin"].numel()
    assert train_obs == 4 and val_obs == 2  # stations disjointes, aucune fuite


def test_empty_sample_guard_skips_without_nan():
    lit = UNetSparseCalibrationModule(_toy_unet(), kelvin_to_celsius=False, elevation_aware=False)
    empty = {
        "x_met": torch.randn(1, MET_CH, SIZE, SIZE),
        "x_dem": torch.randn(1, DEM_CH, SIZE, SIZE),
        "obs_tmin": [torch.zeros(0)],
        "obs_row": [torch.zeros(0, dtype=torch.long)],
        "obs_col": [torch.zeros(0, dtype=torch.long)],
        "obs_dz": [None],
        "date": ["2023-04-04"],
    }
    assert lit.training_step(empty, 0) is None  # nuit vide → sautée
    assert lit.validation_step(empty, 0) is None


def test_fast_dev_run_holdout():
    lit = UNetSparseCalibrationModule(
        _toy_unet(), lr=1e-3, max_epochs=2, kelvin_to_celsius=False
    )
    dm = UNetSparseDataModule(_HoldoutDataset(), num_workers=0)
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit, datamodule=dm)
    assert "val/rmse" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["val/rmse"])


def test_collate_handles_empty_obs():
    """Le collate ne casse pas sur une nuit sans station (obs vides)."""
    sample = {
        "x_met": torch.randn(MET_CH, SIZE, SIZE),
        "x_dem": torch.randn(DEM_CH, SIZE, SIZE),
        "obs_tmin": torch.zeros(0),
        "obs_row": torch.zeros(0, dtype=torch.long),
        "obs_col": torch.zeros(0, dtype=torch.long),
        "obs_dz": torch.zeros(0),
        "date": "2023-04-04",
    }
    batch = unet_sparse_collate([sample])
    assert batch["obs_tmin"][0].numel() == 0
