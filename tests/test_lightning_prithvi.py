"""Smoke tests du fine-tuning Prithvi sur Lightning (adapter-only, sans GPU).

« Test dérisquant » : un modèle jouet exposant ``.adapter`` (entraîné) +
``.backbone`` (gelé) et la signature ``(era5_t0, era5_t1, dem_hr) → ŷ`` passe un
``fast_dev_run`` avec supervision sparse. On vérifie les invariants clés du
fine-tuning : backbone gelé en eval, optimiseur restreint à l'adapter,
checkpoint réduit à l'adapter, et logging de ``val/rmse``.

Sautés sans l'extra (torch + lightning) — cohérent avec la CI extra=statistical.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning.pytorch")

import lightning.pytorch as pl
import torch.nn as nn
from torch.utils.data import Dataset

from downscaling.prtihvi_wxc.lightning_finetune import (
    PrithviFinetuneDataModule,
    PrithviFinetuneLitModule,
    SparseSupervisedLoss,
)

H = W = 16
N_OBS = 6


class _ToyPrithvi(nn.Module):
    """Imite ``PrithviWxCDownscaler`` : backbone gelé + adapter entraîné, sortie K."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(2, 2, 1)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.adapter = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 3, padding=1)
        )

    def forward(self, era5_t0, era5_t1, dem_hr):
        return self.adapter(dem_hr) + 273.15  # Kelvin


class _RandomNightDataset(Dataset):
    """Nuits synthétiques : ERA5 t0/t1, DEM HR, et obs sparse aux stations."""

    def __init__(self, n=6):
        gen = torch.Generator().manual_seed(0)
        self.items = []
        for _ in range(n):
            self.items.append(
                {
                    "era5_t0": torch.randn(6, 8, 8, generator=gen),
                    "era5_t1": torch.randn(6, 8, 8, generator=gen),
                    "dem_hr": torch.randn(3, H, W, generator=gen),
                    "obs_tmin": torch.randn(N_OBS, generator=gen),
                    "obs_row": torch.randint(0, H, (N_OBS,), generator=gen),
                    "obs_col": torch.randint(0, W, (N_OBS,), generator=gen),
                    "date": "2021-04-27",
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _module():
    return PrithviFinetuneLitModule(_ToyPrithvi(), lr=1e-3, warmup_epochs=1, max_epochs=2)


def test_fast_dev_run_logs_val_rmse():
    lit = _module()
    dm = PrithviFinetuneDataModule(_RandomNightDataset(), num_workers=0)
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


def test_backbone_stays_frozen_in_eval():
    lit = _module()
    lit.model.train()  # Lightning basculerait tout en train…
    lit.on_train_epoch_start()  # …mais le hook regèle le backbone en eval.
    assert lit.model.backbone.training is False
    assert lit.model.adapter.training is True


def test_optimizer_targets_adapter_only():
    lit = _module()
    optimizer = lit.configure_optimizers()["optimizer"]
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in lit.model.adapter.parameters()}
    # Aucun paramètre du backbone n'est optimisé.
    assert optimized.isdisjoint({id(p) for p in lit.model.backbone.parameters()})


def test_elevation_aware_loss_corrects_altitude():
    """La correction lapse-rate (obs_dz) ramène la prévision à l'altitude station."""
    loss_fn = SparseSupervisedLoss(lambda_tv=0.0, lambda_smooth=0.0)
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 1, 2] = 10.0
    obs_row = torch.tensor([1])
    obs_col = torch.tensor([2])
    # Station 100 m au-dessus de la maille ; lapse −6.5e-3 → −0.65 °C attendu.
    obs_dz = torch.tensor([100.0])
    target = torch.tensor([10.0 + (-6.5e-3) * 100.0])  # 9.35

    # Sans correction : erreur de 0.65 ; avec correction : ~0.
    _, plain = loss_fn(pred, target, obs_row, obs_col)
    _, corrected = loss_fn(pred, target, obs_row, obs_col, obs_dz=obs_dz, lapse_rate=-6.5e-3)
    assert plain["loss_obs"] == pytest.approx(0.65, abs=1e-4)
    assert corrected["loss_obs"] == pytest.approx(0.0, abs=1e-5)


def test_checkpoint_keeps_only_adapter():
    lit = _module()
    ckpt = {"state_dict": {
        "model.adapter.0.weight": torch.zeros(1),
        "model.adapter.0.bias": torch.zeros(1),
        "model.backbone.weight": torch.zeros(1),
    }}
    lit.on_save_checkpoint(ckpt)
    assert set(ckpt["state_dict"]) == {"model.adapter.0.weight", "model.adapter.0.bias"}
