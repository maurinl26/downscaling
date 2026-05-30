"""Smoke tests Lightning — valident la glue d'entraînement sans GPU ni réseau.

« Test dérisquant » de la Phase 3 : un modèle jouet (SRCNN, sans contrainte de
taille) + un dataset aléatoire passent un ``fast_dev_run`` complet (1 batch
train + 1 batch val). On vérifie que le module logge ``val/rmse`` (la métrique
surveillée par ``ModelCheckpoint`` / ``EarlyStopping``) et que le scheduler
warmup→cosine est bien câblé.

Sautés si l'extra ``dl`` (torch + lightning) n'est pas installé — cohérent avec
la matrice CI qui n'installe que l'extra ``statistical``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("lightning.pytorch")

from torch.utils.data import Dataset

from downscaling.deep_learning.lightning_module import (
    DownscalingDataModule,
    DownscalingLitModule,
)
from downscaling.deep_learning.model import build_model


class _RandomDownscalingDataset(Dataset):
    """Renvoie ``(x_met, dem, y_fine)`` aléatoires aux formes attendues."""

    def __init__(self, n=8, met_ch=5, dem_ch=4, size=16):
        gen = torch.Generator().manual_seed(0)
        self.met = torch.randn(n, met_ch, size, size, generator=gen)
        self.dem = torch.randn(n, dem_ch, size, size, generator=gen)
        self.y = torch.randn(n, met_ch, size, size, generator=gen)

    def __len__(self):
        return self.met.shape[0]

    def __getitem__(self, i):
        return self.met[i], self.dem[i], self.y[i]


def _toy_module():
    # SRCNN : préserve la taille spatiale (pas de downsampling) → robuste en smoke.
    model = build_model(architecture="srcnn", met_in_ch=5, dem_in_ch=4)
    return DownscalingLitModule(model, lr=1e-3, warmup_epochs=1, max_epochs=2)


def test_fast_dev_run_logs_val_rmse(tmp_path):
    lit = _toy_module()
    dm = DownscalingDataModule(_RandomDownscalingDataset(), batch_size=4, num_workers=0)
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


def test_configure_optimizers_warmup_then_cosine():
    lit = _toy_module()
    cfg = lit.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.AdamW)

    scheduler = cfg["lr_scheduler"]["scheduler"]
    # Le LR nominal vit dans base_lrs ; param_groups reflète déjà le facteur warmup.
    assert scheduler.base_lrs[0] == pytest.approx(1e-3)
    # warmup_epochs=1 : facteur < 1 au tout premier epoch…
    assert scheduler.lr_lambdas[0](0) < 1.0
    # …puis palier nominal atteint à la fin du warmup.
    assert scheduler.lr_lambdas[0](1) == pytest.approx(1.0)
