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
import torch
from torch.utils.data import Dataset

from .dataset import FrostNightDataset

# Loss sparse + collate vivent désormais dans lightning_finetune (source unique).
# Réexport pour compat des imports historiques.
from .lightning_finetune import SparseSupervisedLoss, sparse_collate_fn  # noqa: F401
from .loader import PrithviWxCDownscaler
from .netatmo_qc import NetatmoNocturnalQC, load_netatmo_parquet, tmin_nocturnal
from .sencrop import load_sencrop
from .stations import assign_to_grid, elevation_offset

# Alias historique (l'ancien nom était préfixé _).
_sparse_collate_fn = sparse_collate_fn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset de fine-tuning (ERA5 + DEM + capteurs sparse, source-agnostique)
# ---------------------------------------------------------------------------


class StationFineTuneDataset(Dataset):
    """
    Dataset de fine-tuning sparse — 1 sample = 1 nuit. **Source-agnostique**
    (Netatmo, Sencrop) : le loader d'observations est injecté.

    Retourne par nuit : entrées (ERA5 t0/t1 + DEM HR), labels Tmin QC'd aux
    stations, indices grille (row/col), et **décalage d'altitude** ``obs_dz``
    (station − maille, m) si ``elevation_grid`` est fourni — exploité par la loss
    pour la correction lapse-rate (valorisation du MNT au point de calibration).
    """

    def __init__(
        self,
        era5_dataset: FrostNightDataset,
        obs_dir: str | Path,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        *,
        obs_loader,
        file_template: str,
        elevation_grid: np.ndarray | None = None,
        min_stations_per_night: int = 5,
        lapse_rate: float = -6.5e-3,
    ):
        self.era5 = era5_dataset
        self.obs_dir = Path(obs_dir)
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.elevation_grid = elevation_grid
        self.obs_loader = obs_loader
        self.file_template = file_template
        self.min_stations = min_stations_per_night
        self.qc = NetatmoNocturnalQC(lapse_rate=lapse_rate)

        self.valid_nights = self._build_night_index()
        log.info(
            "Fine-tune dataset : %d nuits avec ≥%d stations QC'd",
            len(self.valid_nights),
            min_stations_per_night,
        )

    def _path(self, date_str: str) -> Path:
        return self.obs_dir / self.file_template.format(date=date_str)

    def _build_night_index(self) -> list[dict]:
        nights = []
        for i, t0 in enumerate(self.era5.time_pairs):
            date_str = t0.date().isoformat()
            path = self._path(date_str)
            if not path.exists():
                continue
            try:
                obs_qc = self.qc.run(self.obs_loader(str(path), date_str))
                tmin = tmin_nocturnal(obs_qc)
                n_valid = (~np.isnan(tmin.values)).sum()
                if n_valid >= self.min_stations:
                    nights.append({"era5_idx": i, "date": date_str, "n_stations": n_valid})
            except Exception as e:
                log.debug("Nuit %s ignorée : %s", date_str, e)
        return nights

    def __len__(self) -> int:
        return len(self.valid_nights)

    def __getitem__(self, idx: int) -> dict:
        night = self.valid_nights[idx]
        era5_sample = self.era5[night["era5_idx"]]

        obs_qc = self.qc.run(self.obs_loader(str(self._path(night["date"])), night["date"]))
        tmin_obs = tmin_nocturnal(obs_qc)
        valid = ~np.isnan(tmin_obs.values)

        lat_obs, lon_obs = obs_qc.lat[valid], obs_qc.lon[valid]
        elev_obs = obs_qc.elevation_m[valid]
        tmin_vals = tmin_obs.values[valid]  # (n_obs,) en °C

        row_idx, col_idx = assign_to_grid(lat_obs, lon_obs, self.lat_grid, self.lon_grid)
        obs_dz = elevation_offset(elev_obs, row_idx, col_idx, self.elevation_grid)

        return {
            "era5_t0": era5_sample.era5_t0,  # (C, H_lr, W_lr)
            "era5_t1": era5_sample.era5_t1,  # (C, H_lr, W_lr)
            "dem_hr": era5_sample.dem_hr,  # (3, H_hr, W_hr)
            "obs_tmin": torch.tensor(tmin_vals, dtype=torch.float32),  # (n_obs,)
            "obs_row": torch.tensor(row_idx, dtype=torch.long),
            "obs_col": torch.tensor(col_idx, dtype=torch.long),
            "obs_dz": torch.tensor(obs_dz, dtype=torch.float32),  # station − maille (m)
            "date": night["date"],
        }


class NetatmoFineTuneDataset(StationFineTuneDataset):
    """Fine-tuning sur stations **Netatmo** (``netatmo_<date>.parquet``)."""

    def __init__(
        self,
        era5_dataset,
        netatmo_dir,
        lat_grid,
        lon_grid,
        *,
        elevation_grid=None,
        min_stations_per_night=5,
        lapse_rate=-6.5e-3,
    ):
        super().__init__(
            era5_dataset,
            netatmo_dir,
            lat_grid,
            lon_grid,
            obs_loader=load_netatmo_parquet,
            file_template="netatmo_{date}.parquet",
            elevation_grid=elevation_grid,
            min_stations_per_night=min_stations_per_night,
            lapse_rate=lapse_rate,
        )


class SencropFineTuneDataset(StationFineTuneDataset):
    """Fine-tuning sur stations **Sencrop** (``sencrop_<date>.csv`` / ``.parquet``).

    Identique à Netatmo côté chaîne ; seul le loader d'observations change. La
    calibration MNT-aware (``obs_dz``) s'applique de la même façon — c'est l'étage
    C du chemin CERRA → 1 km → capteurs (cf. ``docs/architecture.md``).
    """

    def __init__(
        self,
        era5_dataset,
        sencrop_dir,
        lat_grid,
        lon_grid,
        *,
        elevation_grid=None,
        file_template="sencrop_{date}.csv",
        min_stations_per_night=5,
        lapse_rate=-6.5e-3,
    ):
        super().__init__(
            era5_dataset,
            sencrop_dir,
            lat_grid,
            lon_grid,
            obs_loader=load_sencrop,
            file_template=file_template,
            elevation_grid=elevation_grid,
            min_stations_per_night=min_stations_per_night,
            lapse_rate=lapse_rate,
        )


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
        n_trainable = sum(p.numel() for p in model.adapter.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in model.backbone.parameters())
        log.info(
            f"Paramètres entraînables (adapter) : {n_trainable:,} | Gelés (backbone) : {n_frozen:,}"
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
        """Fine-tune l'adapter via Lightning et sauvegarde le meilleur checkpoint.

        Le scheduler warmup→cosine, le clipping de gradient, le meilleur
        checkpoint (adapter seul) et l'early stopping sont des briques Lightning
        (cf. :mod:`downscaling.prtihvi_wxc.lightning_finetune`).

        Returns: ``{"best_val_rmse": float | None}``.
        """
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger

        from .lightning_finetune import PrithviFinetuneDataModule, PrithviFinetuneLitModule

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        lit = PrithviFinetuneLitModule(
            self.model,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=self.config.get("warmup_epochs", 5),
            max_epochs=epochs,
            loss_weights={
                "obs": self.config.get("lambda_obs", 1.0),
                "tv": self.config.get("lambda_tv", 0.01),
                "smooth": self.config.get("lambda_smooth", 0.001),
            },
        )
        datamodule = PrithviFinetuneDataModule(
            finetune_dataset,
            val_fraction=val_fraction,
            num_workers=self.config.get("num_workers", 2),
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=self.config.get("accelerator", "auto"),
            devices=self.config.get("devices", 1),
            precision=self.config.get("precision", "32-true"),
            gradient_clip_val=1.0,  # seul l'adapter a des gradients (backbone gelé)
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(output_dir),
                    filename="best_adapter",
                    monitor="val/rmse",
                    mode="min",
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor="val/rmse",
                    mode="min",
                    patience=self.config.get("patience", 10),
                ),
            ],
            logger=CSVLogger(save_dir=str(output_dir), name="logs"),
        )
        trainer.fit(lit, datamodule=datamodule)

        best = trainer.checkpoint_callback.best_model_score
        best = float(best) if best is not None else None
        log.info(
            "Fine-tuning terminé. Best val/rmse : %s",
            f"{best:.3f}°C" if best is not None else "n/a",
        )
        return {"best_val_rmse": best}
