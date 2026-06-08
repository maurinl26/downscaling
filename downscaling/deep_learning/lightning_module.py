"""Enveloppe Lightning du U-Net FiLM de descente d'échelle.

Remplace la boucle ``torch`` manuelle (``train.Trainer``) par un
``LightningModule`` + ``LightningDataModule`` : le scheduler warmup→cosine,
le clipping de gradient, l'early stopping, le meilleur checkpoint et le logger
deviennent des briques Lightning natives (cf. ``train.main``).

Pattern « modèle injecté » : le réseau est construit en amont (``build_model``)
puis passé au module ; l'optimiseur et le scheduler sont instanciés dans
``configure_optimizers`` (et non au ``__init__``) pour qu'ils voient bien les
paramètres du modèle.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Réutilise loss composite / métriques / scheduler — source unique de vérité.
from .train import DownscalingLoss, compute_metrics, cosine_with_warmup


class DownscalingLitModule(pl.LightningModule):
    """``LightningModule`` pour le U-Net FiLM (ou tout modèle ``(x_met, x_dem) → ŷ``)."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        loss_weights: dict | None = None,
        frost_threshold_norm: float | None = None,  # seuil gel en espace normalisé = (0−µ)/σ
    ):
        super().__init__()
        self.model = model
        lw = loss_weights or {}
        # Seuil de détection du gel (queue froide) — suivi POD/FAR + pondération de loss.
        self.frost_threshold_norm = frost_threshold_norm
        self.criterion = DownscalingLoss(
            lambda_mse=lw.get("mse", 1.0),
            lambda_spectral=lw.get("spectral", 0.1),
            lambda_gradient=lw.get("gradient", 0.05),
            frost_threshold_norm=frost_threshold_norm,
            frost_alpha=lw.get("frost_alpha", 0.0),
        )
        self._frost = {"hits": 0, "misses": 0, "fa": 0}
        # Le modèle est un objet injecté (non sérialisable proprement) → exclu.
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x_met: torch.Tensor, x_dem: torch.Tensor) -> torch.Tensor:
        return self.model(x_met, x_dem)

    def training_step(self, batch, batch_idx):
        x_coarse, dem, y_fine = batch
        pred = self(x_coarse, dem)
        loss, breakdown = self.criterion(pred, y_fine)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for name, value in breakdown.items():
            self.log(f"train/{name}", value, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._frost = {"hits": 0, "misses": 0, "fa": 0}

    def validation_step(self, batch, batch_idx):
        x_coarse, dem, y_fine = batch
        pred = self(x_coarse, dem)
        loss, _ = self.criterion(pred, y_fine)
        metrics = compute_metrics(pred, y_fine)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse", metrics["rmse"], prog_bar=True)
        self.log("val/mae", metrics["mae"])
        self.log("val/bias", metrics["bias"])

        # Détection du gel (canal 0 = t2m) au seuil normalisé — accumulé sur l'epoch.
        if self.frost_threshold_norm is not None:
            thr = self.frost_threshold_norm
            obs_f = y_fine[:, 0] < thr
            src_f = pred[:, 0] < thr
            self._frost["hits"] += int((obs_f & src_f).sum())
            self._frost["misses"] += int((obs_f & ~src_f).sum())
            self._frost["fa"] += int((~obs_f & src_f).sum())

    def on_validation_epoch_end(self):
        if self.frost_threshold_norm is None:
            return
        h, m, fa = self._frost["hits"], self._frost["misses"], self._frost["fa"]
        pod = h / (h + m) if (h + m) else 0.0
        far = fa / (h + fa) if (h + fa) else 0.0
        self.log("val/pod", pod, prog_bar=True)
        self.log("val/far", far)

    def configure_optimizers(self):
        # Instancié ici (pas au __init__) : l'optimiseur doit voir self.parameters().
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = cosine_with_warmup(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            total_epochs=self.hparams.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class DownscalingDataModule(pl.LightningDataModule):
    """Split train/val déterministe + ``DataLoader`` autour d'un ``Dataset`` unique."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 8,
        num_workers: int = 0,
        val_fraction: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None

    def setup(self, stage: str | None = None):
        n_val = max(1, int(len(self.dataset) * self.val_fraction))
        n_train = len(self.dataset) - n_val
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds = random_split(
            self.dataset, [n_train, n_val], generator=generator
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
