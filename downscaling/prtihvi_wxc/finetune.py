"""
finetune.py — Fine-tuning du CNN adapter Prithvi WxC avec données Netatmo.

Stratégie : supervision sparse aux stations (pas de supervision dense).
  - Le backbone Prithvi WxC reste gelé.
  - Seul le DEMConditionedAdapter (~2M params) est entraîné.
  - Loss calculée uniquement aux positions des stations Netatmo QC'd.
  - Régularisation TV (Total Variation) pour préserver la cohérence spatiale
    entre les stations (évite le surapprentissage aux positions observées).

Architecture de loss :
    L = L_obs + λ_tv × L_TV + λ_smooth × L_smooth

    L_obs    = RMSE(T_pred[stations] - T_netatmo_qc) — supervision principale
    L_TV     = variation totale du champ (lissage spatial)
    L_smooth = pénalité de gradient excessif (anti-artefact)

Données d'entraînement recommandées :
  - Période 2015–2022 (couverture Netatmo dense en Drôme)
  - Heures nocturnes uniquement (20h–08h UTC) — QC fiable
  - Saison de gel (oct–mai) — événements d'intérêt assuranciel
  - Split : 80% train / 20% validation par nuit (pas par heure)
    (évite la fuite d'information temporelle)

Référence :
  Yu et al. (2025) NASA NTRS 20250006603 — fine-tuning avec données éparses
  Nipen et al. (2020) — assimilation Netatmo en NWP opérationnel
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from downscaling.deep_learning.prithvi_wxc.loader import PrithviWxCDownscaler
from downscaling.deep_learning.prithvi_wxc.dataset import FrostNightDataset, ERA5_VARS
from downscaling.shared.netatmo_qc import NetatmoNocturnalQC, load_netatmo_parquet, tmin_nocturnal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset de fine-tuning (ERA5 + DEM + Netatmo sparse)
# ---------------------------------------------------------------------------

class NetatmoFineTuneDataset(Dataset):
    """
    Dataset pour fine-tuning : chaque sample = une nuit.

    Retourne :
      - Inputs : paires ERA5 (t0, t1) + DEM HR
      - Labels : Tmin Netatmo QC'd aux positions des stations (sparse)
      - Masques : positions de grille correspondant aux stations

    Le training loss est calculé uniquement sur les pixels
    correspondant aux stations Netatmo valides.
    """

    def __init__(
        self,
        era5_dataset: FrostNightDataset,
        netatmo_dir: str | Path,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        min_stations_per_night: int = 5,
        lapse_rate: float = -6.5e-3,
    ):
        self.era5 = era5_dataset
        self.netatmo_dir = Path(netatmo_dir)
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.min_stations = min_stations_per_night
        self.qc = NetatmoNocturnalQC(lapse_rate=lapse_rate)

        # Construire l'index des nuits avec assez de stations Netatmo
        self.valid_nights = self._build_night_index()
        log.info(
            f"Fine-tune dataset : {len(self.valid_nights)} nuits "
            f"avec ≥{min_stations_per_night} stations Netatmo QC'd"
        )

    def _build_night_index(self) -> list[dict]:
        """Indexe les nuits où les données Netatmo sont disponibles et suffisantes."""
        nights = []
        for i, t0 in enumerate(self.era5.time_pairs):
            date_str = t0.date().isoformat()
            netatmo_path = self.netatmo_dir / f"netatmo_{date_str}.parquet"
            if not netatmo_path.exists():
                continue
            try:
                obs_raw = load_netatmo_parquet(str(netatmo_path), date_str)
                obs_qc = self.qc.run(obs_raw)
                tmin = tmin_nocturnal(obs_qc)
                n_valid = (~np.isnan(tmin.values)).sum()
                if n_valid >= self.min_stations:
                    nights.append({
                        "era5_idx": i,
                        "date": date_str,
                        "netatmo_path": str(netatmo_path),
                        "n_stations": n_valid,
                    })
            except Exception as e:
                log.debug(f"Nuit {date_str} ignorée : {e}")
        return nights

    def __len__(self) -> int:
        return len(self.valid_nights)

    def __getitem__(self, idx: int) -> dict:
        night = self.valid_nights[idx]

        # ERA5 inputs
        era5_sample = self.era5[night["era5_idx"]]

        # Netatmo labels
        obs_raw = load_netatmo_parquet(night["netatmo_path"], night["date"])
        obs_qc = self.qc.run(obs_raw)
        tmin_obs = tmin_nocturnal(obs_qc)
        valid = ~np.isnan(tmin_obs.values)

        # Indices de grille pour chaque station valide
        lat_obs = obs_qc.lat[valid]
        lon_obs = obs_qc.lon[valid]
        tmin_vals = tmin_obs.values[valid]  # (n_obs,) en °C

        row_idx = np.argmin(np.abs(self.lat_grid[:, None] - lat_obs[None, :]), axis=0)
        col_idx = np.argmin(np.abs(self.lon_grid[:, None] - lon_obs[None, :]), axis=0)

        return {
            "era5_t0": era5_sample.era5_t0,               # (C, H_lr, W_lr)
            "era5_t1": era5_sample.era5_t1,               # (C, H_lr, W_lr)
            "dem_hr": era5_sample.dem_hr,                  # (3, H_hr, W_hr)
            "obs_tmin": torch.tensor(tmin_vals, dtype=torch.float32),  # (n_obs,)
            "obs_row": torch.tensor(row_idx, dtype=torch.long),
            "obs_col": torch.tensor(col_idx, dtype=torch.long),
            "date": night["date"],
        }


# ---------------------------------------------------------------------------
# Loss fonction
# ---------------------------------------------------------------------------

class SparseSupervisedLoss(nn.Module):
    """
    Loss combinée pour supervision sparse (stations Netatmo).

    L = λ_obs × L_obs + λ_tv × L_TV + λ_smooth × L_smooth

    L_obs   : RMSE aux pixels des stations
    L_TV    : Total Variation — régularisation spatiale
    L_smooth: Pénalité Laplacien — anti-bruit haute fréquence
    """

    def __init__(
        self,
        lambda_obs: float = 1.0,
        lambda_tv: float = 0.01,
        lambda_smooth: float = 0.001,
    ):
        super().__init__()
        self.lambda_obs = lambda_obs
        self.lambda_tv = lambda_tv
        self.lambda_smooth = lambda_smooth

    def forward(
        self,
        pred: torch.Tensor,        # (B, 1, H_hr, W_hr)
        obs_tmin: torch.Tensor,    # (n_obs,) — valeurs aux stations
        obs_row: torch.Tensor,     # (n_obs,) — indices ligne grille
        obs_col: torch.Tensor,     # (n_obs,) — indices colonne grille
        batch_idx: int = 0,        # Indice batch pour extraction
    ) -> tuple[torch.Tensor, dict]:

        # L_obs : supervision aux stations
        pred_at_obs = pred[batch_idx, 0, obs_row, obs_col]  # (n_obs,)
        l_obs = torch.sqrt(torch.mean((pred_at_obs - obs_tmin) ** 2))

        # L_TV : Total Variation (cohérence spatiale)
        diff_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        diff_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        l_tv = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))

        # L_smooth : pénalité Laplacien discret
        laplacian = (
            pred[:, :, 1:-1, 1:-1] * (-4)
            + pred[:, :, :-2, 1:-1]
            + pred[:, :, 2:, 1:-1]
            + pred[:, :, 1:-1, :-2]
            + pred[:, :, 1:-1, 2:]
        )
        l_smooth = torch.mean(laplacian ** 2)

        loss = (
            self.lambda_obs * l_obs
            + self.lambda_tv * l_tv
            + self.lambda_smooth * l_smooth
        )

        return loss, {
            "loss_total": loss.item(),
            "loss_obs": l_obs.item(),
            "loss_tv": l_tv.item(),
            "loss_smooth": l_smooth.item(),
        }


# ---------------------------------------------------------------------------
# Boucle de fine-tuning
# ---------------------------------------------------------------------------

class PrithviWxCFinetuner:
    """
    Fine-tune le CNN adapter Prithvi WxC sur données Netatmo.

    Seuls les poids de DEMConditionedAdapter sont mis à jour.
    Le backbone Prithvi WxC (2.3B params) reste gelé.

    Usage :
        finetuner = PrithviWxCFinetuner(model, config)
        finetuner.run(train_dataset, val_dataset, output_dir="checkpoints/")
    """

    def __init__(self, model: PrithviWxCDownscaler, config: dict):
        self.model = model
        self.config = config
        self.device = config.get("device", "cuda")

        # Vérifier que seul l'adapter est entraînable
        n_trainable = sum(
            p.numel() for p in model.adapter.parameters() if p.requires_grad
        )
        n_frozen = sum(
            p.numel() for p in model.backbone.parameters()
        )
        log.info(
            f"Paramètres entraînables (adapter) : {n_trainable:,} | "
            f"Gelés (backbone) : {n_frozen:,}"
        )

    def run(
        self,
        finetune_dataset: NetatmoFineTuneDataset,
        output_dir: str | Path,
        val_fraction: float = 0.2,
        epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ) -> dict:
        """
        Lance le fine-tuning et sauvegarde les checkpoints.

        Returns: dict avec historique de loss train/val
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split train/val par nuit (pas par heure — évite la fuite temporelle)
        n_val = max(1, int(len(finetune_dataset) * val_fraction))
        n_train = len(finetune_dataset) - n_val
        train_ds, val_ds = random_split(
            finetune_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        log.info(f"Train: {n_train} nuits | Val: {n_val} nuits")

        train_loader = DataLoader(
            train_ds,
            batch_size=1,           # 1 nuit à la fois (obs sparse variables)
            shuffle=True,
            num_workers=self.config.get("num_workers", 2),
            collate_fn=_sparse_collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=_sparse_collate_fn
        )

        # Optimiseur sur l'adapter uniquement
        optimizer = AdamW(
            self.model.adapter.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        criterion = SparseSupervisedLoss(
            lambda_obs=self.config.get("lambda_obs", 1.0),
            lambda_tv=self.config.get("lambda_tv", 0.01),
            lambda_smooth=self.config.get("lambda_smooth", 0.001),
        )

        history = {"train": [], "val": [], "best_val": float("inf")}

        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            # Backbone gelé même en mode train
            self.model.backbone.eval()

            train_losses = []
            for batch in train_loader:
                optimizer.zero_grad()

                era5_t0 = batch["era5_t0"].to(self.device)
                era5_t1 = batch["era5_t1"].to(self.device)
                dem_hr = batch["dem_hr"].to(self.device)
                obs_tmin = batch["obs_tmin"][0].to(self.device)  # (n_obs,)
                obs_row = batch["obs_row"][0].to(self.device)
                obs_col = batch["obs_col"][0].to(self.device)

                # Forward (backbone.encode en no_grad via loader.py)
                pred = self.model(era5_t0, era5_t1, dem_hr)  # (1,1,H,W) en K
                pred_c = pred - 273.15  # Kelvin → Celsius pour comparer Netatmo

                loss, metrics = criterion(pred_c, obs_tmin, obs_row, obs_col)
                loss.backward()

                # Gradient clipping (adapter peut avoir des gradients instables)
                nn.utils.clip_grad_norm_(self.model.adapter.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(metrics["loss_obs"])

            # --- Validation ---
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    era5_t0 = batch["era5_t0"].to(self.device)
                    era5_t1 = batch["era5_t1"].to(self.device)
                    dem_hr = batch["dem_hr"].to(self.device)
                    obs_tmin = batch["obs_tmin"][0].to(self.device)
                    obs_row = batch["obs_row"][0].to(self.device)
                    obs_col = batch["obs_col"][0].to(self.device)

                    pred = self.model(era5_t0, era5_t1, dem_hr)
                    pred_c = pred - 273.15
                    _, metrics = criterion(pred_c, obs_tmin, obs_row, obs_col)
                    val_losses.append(metrics["loss_obs"])

            train_rmse = np.mean(train_losses)
            val_rmse = np.mean(val_losses)
            history["train"].append(train_rmse)
            history["val"].append(val_rmse)

            scheduler.step()

            log.info(
                f"Epoch {epoch+1:3d}/{epochs} — "
                f"Train RMSE: {train_rmse:.3f}°C | Val RMSE: {val_rmse:.3f}°C"
            )

            # Checkpoint si meilleure val
            if val_rmse < history["best_val"]:
                history["best_val"] = val_rmse
                self._save_checkpoint(output_dir / "best_adapter.pt", epoch, val_rmse)

        # Checkpoint final
        self._save_checkpoint(output_dir / "last_adapter.pt", epochs - 1, val_rmse)

        log.info(
            f"Fine-tuning terminé. Best val RMSE: {history['best_val']:.3f}°C\n"
            f"Checkpoint : {output_dir / 'best_adapter.pt'}"
        )
        return history

    def _save_checkpoint(self, path: Path, epoch: int, val_rmse: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "val_rmse": val_rmse,
                # Sauvegarder uniquement l'adapter (backbone reste sur HuggingFace)
                "adapter": {
                    f"adapter.{k}": v
                    for k, v in self.model.adapter.state_dict().items()
                },
            },
            path,
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sparse_collate_fn(samples: list[dict]) -> dict:
    """Collate pour batch de taille 1 avec obs sparse (longueurs variables)."""
    batch = {
        "era5_t0": torch.stack([s["era5_t0"] for s in samples]),
        "era5_t1": torch.stack([s["era5_t1"] for s in samples]),
        "dem_hr": torch.stack([s["dem_hr"] for s in samples]),
        # obs_tmin / obs_row / obs_col : longueurs variables → liste
        "obs_tmin": [s["obs_tmin"] for s in samples],
        "obs_row": [s["obs_row"] for s in samples],
        "obs_col": [s["obs_col"] for s in samples],
        "date": [s["date"] for s in samples],
    }
    return batch
