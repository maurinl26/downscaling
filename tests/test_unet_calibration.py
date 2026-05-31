"""Smoke tests de la calibration sparse du U-Net (chemin B, étage C) — sans GPU.

Valide le câblage déployable « CERRA → U-Net 1 km → capteurs Sencrop » : le
``UNetSparseCalibrationModule`` appelle le U-Net ``(x_met, x_dem)``, sélectionne
le canal cible, et supervise sur stations sparse avec correction d'altitude.

Sauté sans l'extra ``dl`` (torch + lightning).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning.pytorch")

import lightning.pytorch as pl
from torch.utils.data import Dataset

from downscaling.deep_learning.model import build_model
from downscaling.deep_learning.sparse_calibration import (
    UNetSparseCalibrationModule,
    UNetSparseDataModule,
    unet_sparse_collate,
)

MET_CH, DEM_CH, SIZE, N_OBS = 5, 4, 16, 6


class _RandomUNetSparseDataset(Dataset):
    """Nuits synthétiques : champ météo coarse + MNT + Tmin sparse aux stations."""

    def __init__(self, n=6):
        gen = torch.Generator().manual_seed(0)
        self.items = []
        for _ in range(n):
            self.items.append({
                "x_met": torch.randn(MET_CH, SIZE, SIZE, generator=gen),
                "x_dem": torch.randn(DEM_CH, SIZE, SIZE, generator=gen),
                "obs_tmin": torch.randn(N_OBS, generator=gen),
                "obs_row": torch.randint(0, SIZE, (N_OBS,), generator=gen),
                "obs_col": torch.randint(0, SIZE, (N_OBS,), generator=gen),
                "obs_dz": torch.randn(N_OBS, generator=gen) * 100.0,  # m
                "date": "2021-04-27",
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _toy_unet():
    # SRCNN : préserve la taille spatiale, sort `met_in_ch` canaux.
    return build_model(architecture="srcnn", met_in_ch=MET_CH, dem_in_ch=DEM_CH)


def test_fast_dev_run_calibration():
    lit = UNetSparseCalibrationModule(_toy_unet(), target_channel=0, lr=1e-3, max_epochs=2)
    dm = UNetSparseDataModule(_RandomUNetSparseDataset(), num_workers=0)
    trainer = pl.Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(lit, datamodule=dm)
    assert "val/rmse" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["val/rmse"])


def test_collate_stacks_dense_lists_sparse():
    batch = unet_sparse_collate([_RandomUNetSparseDataset(n=1)[0]])
    assert batch["x_met"].shape == (1, MET_CH, SIZE, SIZE)
    assert batch["x_dem"].shape == (1, DEM_CH, SIZE, SIZE)
    assert isinstance(batch["obs_tmin"], list) and batch["obs_tmin"][0].shape == (N_OBS,)
    assert batch["obs_dz"][0].shape == (N_OBS,)


def test_target_channel_selected_for_loss():
    """Seul le canal cible alimente la supervision (sortie U-Net multi-canaux)."""
    lit = UNetSparseCalibrationModule(_toy_unet(), target_channel=2)
    ds = _RandomUNetSparseDataset(n=1)
    batch = unet_sparse_collate([ds[0]])
    loss, parts = lit._shared_step(batch)
    assert torch.isfinite(loss) and "loss_obs" in parts


def test_elevation_aware_toggle_changes_loss():
    """Activer/désactiver obs_dz modifie la perte (le MNT est bien pris en compte)."""
    ds = _RandomUNetSparseDataset(n=1)
    batch = unet_sparse_collate([ds[0]])

    on = UNetSparseCalibrationModule(_toy_unet(), elevation_aware=True, lapse_rate=-6.5e-3)
    off = UNetSparseCalibrationModule(on.model, elevation_aware=False)  # même modèle
    _, parts_on = on._shared_step(batch)
    _, parts_off = off._shared_step(batch)
    assert parts_on["loss_obs"] != parts_off["loss_obs"]
