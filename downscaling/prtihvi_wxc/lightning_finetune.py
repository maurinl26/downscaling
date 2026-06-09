"""Fine-tuning Prithvi WxC sur Lightning — adapter entraîné, backbone gelé.

Pendant Prithvi de :mod:`downscaling.deep_learning.lightning_module`. Le réseau
``PrithviWxCDownscaler`` est *injecté* (pattern « modèle injecté ») : seul son
``adapter`` (~2 M params) est optimisé, le ``backbone`` (2,3 B) reste gelé et en
``eval()`` même pendant l'entraînement. La supervision est *sparse* (stations
Netatmo QC'd) via :class:`SparseSupervisedLoss`.

Module auto-suffisant (torch + lightning) : il n'importe ni le loader Prithvi
(qui tire ``huggingface_hub``) ni le dataset Netatmo — c'est l'appelant qui les
construit et passe le modèle + dataset. Reste donc importable et testable sans
l'extra ``prithvi`` complet.
"""

from __future__ import annotations

import math

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

KELVIN = 273.15


# ---------------------------------------------------------------------------
# Loss sparse + collate (déplacés depuis finetune.py — source unique)
# ---------------------------------------------------------------------------

class SparseSupervisedLoss(nn.Module):
    """Loss de supervision sparse aux stations + régularisation spatiale.

    ``L = λ_obs · RMSE(stations) + λ_tv · TV + λ_smooth · ‖Laplacien‖²``
    """

    def __init__(
        self,
        lambda_obs: float = 1.0,
        lambda_tv: float = 0.01,
        lambda_smooth: float = 0.001,
        frost_alpha: float = 0.0,        # >0 = sur-pondère les nuits gélives observées
        frost_threshold: float = 0.0,    # °C (obs_tmin < seuil → poids accru)
    ):
        super().__init__()
        self.lambda_obs = lambda_obs
        self.lambda_tv = lambda_tv
        self.lambda_smooth = lambda_smooth
        self.frost_alpha = frost_alpha
        self.frost_threshold = frost_threshold

    def forward(
        self,
        pred: torch.Tensor,        # (B, 1, H_hr, W_hr)
        obs_tmin: torch.Tensor,    # (n_obs,) — valeurs aux stations
        obs_row: torch.Tensor,     # (n_obs,) — indices ligne grille
        obs_col: torch.Tensor,     # (n_obs,) — indices colonne grille
        batch_idx: int = 0,
        obs_dz: torch.Tensor | None = None,   # (n_obs,) — altitude_station − altitude_maille (m)
        lapse_rate: float = -6.5e-3,          # K/m (gel nocturne : ≈ −4e-3, cf. config)
    ) -> tuple[torch.Tensor, dict]:
        # L_obs : supervision aux stations
        pred_at_obs = pred[batch_idx, 0, obs_row, obs_col]
        # Correction d'altitude (valorise le MNT) : la maille 1 km et la station
        # ne sont pas à la même altitude → on ramène la prévision à l'altitude
        # réelle du capteur avant comparaison (fonds froids de vallée, etc.).
        if obs_dz is not None:
            pred_at_obs = pred_at_obs + lapse_rate * obs_dz
        # Supervision pondérée queue-froide : les nuits gélives observées pèsent davantage
        # → empêche la régression vers la moyenne chaude (sinon POD s'effondre, cf. étage C MSE).
        err2 = (pred_at_obs - obs_tmin) ** 2
        if self.frost_alpha:
            w = 1.0 + self.frost_alpha * torch.relu(self.frost_threshold - obs_tmin)
            l_obs = torch.sqrt((w * err2).sum() / w.sum())
        else:
            l_obs = torch.sqrt(torch.mean(err2))

        # L_TV : variation totale (cohérence spatiale)
        diff_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        diff_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        l_tv = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))

        # L_smooth : pénalité Laplacien discret (anti-artefact haute fréquence)
        laplacian = (
            pred[:, :, 1:-1, 1:-1] * (-4)
            + pred[:, :, :-2, 1:-1]
            + pred[:, :, 2:, 1:-1]
            + pred[:, :, 1:-1, :-2]
            + pred[:, :, 1:-1, 2:]
        )
        l_smooth = torch.mean(laplacian ** 2)

        loss = self.lambda_obs * l_obs + self.lambda_tv * l_tv + self.lambda_smooth * l_smooth
        return loss, {
            "loss_total": loss.item(),
            "loss_obs": l_obs.item(),
            "loss_tv": l_tv.item(),
            "loss_smooth": l_smooth.item(),
        }


def sparse_collate_fn(samples: list[dict]) -> dict:
    """Collate pour batch de taille 1 avec obs sparse (longueurs variables)."""
    return {
        "era5_t0": torch.stack([s["era5_t0"] for s in samples]),
        "era5_t1": torch.stack([s["era5_t1"] for s in samples]),
        "dem_hr": torch.stack([s["dem_hr"] for s in samples]),
        # obs_* : longueurs variables d'une nuit à l'autre → liste
        "obs_tmin": [s["obs_tmin"] for s in samples],
        "obs_row": [s["obs_row"] for s in samples],
        "obs_col": [s["obs_col"] for s in samples],
        # Décalage d'altitude station−maille (m), si fourni (calibration MNT-aware).
        "obs_dz": [s.get("obs_dz") for s in samples],
        "date": [s["date"] for s in samples],
    }


def _warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int):
    """LambdaLR : warmup linéaire puis décroissance cosine (même forme que le U-Net)."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# LightningModule + DataModule
# ---------------------------------------------------------------------------

class PrithviFinetuneLitModule(pl.LightningModule):
    """Fine-tune adapter-only d'un modèle ``(era5_t0, era5_t1, dem_hr) → ŷ`` (Kelvin)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        loss_weights: dict | None = None,
        kelvin_to_celsius: bool = True,
        lapse_rate: float = -6.5e-3,      # correction d'altitude station/maille (K/m)
        elevation_aware: bool = True,     # valorise le MNT via obs_dz (si fourni)
    ):
        super().__init__()
        self.model = model
        lw = loss_weights or {}
        self.criterion = SparseSupervisedLoss(
            lambda_obs=lw.get("obs", 1.0),
            lambda_tv=lw.get("tv", 0.01),
            lambda_smooth=lw.get("smooth", 0.001),
        )
        self.kelvin_to_celsius = kelvin_to_celsius
        self.lapse_rate = lapse_rate
        self.elevation_aware = elevation_aware
        self.save_hyperparameters(ignore=["model"])

    def forward(self, era5_t0, era5_t1, dem_hr):
        return self.model(era5_t0, era5_t1, dem_hr)

    def on_train_epoch_start(self):
        # Backbone gelé : reste en eval même quand Lightning bascule le module en train.
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None:
            backbone.eval()

    def _shared_step(self, batch):
        pred = self(batch["era5_t0"], batch["era5_t1"], batch["dem_hr"])
        if self.kelvin_to_celsius:
            pred = pred - KELVIN  # comparer aux capteurs en °C
        # batch_size==1 (obs sparse) → on déballe la première (et unique) nuit.
        obs_dz = batch.get("obs_dz", [None])[0] if self.elevation_aware else None
        return self.criterion(
            pred, batch["obs_tmin"][0], batch["obs_row"][0], batch["obs_col"][0],
            obs_dz=obs_dz, lapse_rate=self.lapse_rate,
        )

    def training_step(self, batch, batch_idx):
        loss, parts = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=1)
        self.log("train/rmse", parts["loss_obs"], prog_bar=True, batch_size=1)
        self.log("train/tv", parts["loss_tv"], batch_size=1)
        self.log("train/smooth", parts["loss_smooth"], batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, parts = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, batch_size=1)
        self.log("val/rmse", parts["loss_obs"], prog_bar=True, batch_size=1)

    def configure_optimizers(self):
        # Optimise l'adapter seul (backbone gelé) ; instancié ici → voit bien les params.
        adapter = getattr(self.model, "adapter", None)
        params = adapter.parameters() if adapter is not None else self.parameters()
        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = _warmup_cosine(optimizer, self.hparams.warmup_epochs, self.hparams.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_save_checkpoint(self, checkpoint):
        # Ne garder que l'adapter — le backbone (2,3 B) est rechargé depuis HuggingFace.
        state = checkpoint.get("state_dict")
        if state:
            checkpoint["state_dict"] = {k: v for k, v in state.items() if ".adapter." in k}


class PrithviFinetuneDataModule(pl.LightningDataModule):
    """Split train/val par nuit (déterministe) + ``DataLoader`` batch=1 sparse."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_workers: int = 0,
        val_fraction: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset = dataset
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
            batch_size=1,  # 1 nuit à la fois (obs sparse de longueur variable)
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=sparse_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=sparse_collate_fn,
        )
