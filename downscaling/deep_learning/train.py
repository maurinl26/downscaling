"""
Entraînement du modèle de descente d'échelle par deep learning (Lightning).

Fonctionnalités
---------------
- Loss composite : MSE + loss spectrale (sur T2m) + loss de gradient spatial
- Scheduler cosine annealing avec warmup linéaire (``configure_optimizers``)
- Meilleur checkpoint (``ModelCheckpoint(monitor="val/rmse")``) + early stopping
- Logger MLflow si ``MLFLOW_TRACKING_URI`` est défini, sinon CSV
- Accelerator / precision pilotés par le groupe Hydra ``cluster=`` (local / cloud)

La boucle ``torch`` manuelle a été remplacée par un ``LightningModule`` +
``LightningDataModule`` (cf. :mod:`downscaling.deep_learning.lightning_module`).

Usage
-----
    python -m downscaling.deep_learning.train \
        --data-dir data/training/ \
        --epochs 100 --batch-size 8 \
        --checkpoint-dir checkpoints/ \
        --override cluster=cloud dl.base_ch=64
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import lightning.pytorch as pl
except ImportError as e:
    raise ImportError("PyTorch + Lightning requis : pip install 'downscaling[dl]'") from e

from downscaling.config import load_config
from .dataset import DownscalingDataset
from .model import build_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fonctions de perte
# ---------------------------------------------------------------------------

class SpectralLoss(nn.Module):
    """
    Pénalise les erreurs dans le domaine fréquentiel (FFT 2D).
    Favorise la préservation des structures à haute fréquence spatiale.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Prend le premier canal (T2m ou la variable principale)
        p = pred[:, 0, :, :]
        t = target[:, 0, :, :]
        p_fft = torch.fft.rfft2(p)
        t_fft = torch.fft.rfft2(t)
        return nn.functional.mse_loss(p_fft.abs(), t_fft.abs())


class GradientLoss(nn.Module):
    """
    Pénalise les erreurs sur les gradients spatiaux (préserve les contours).
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
        dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
        return nn.functional.l1_loss(dy_pred, dy_tgt) + nn.functional.l1_loss(dx_pred, dx_tgt)


class DownscalingLoss(nn.Module):
    """
    Loss composite pour la descente d'échelle :
        L = λ_mse · MSE + λ_spec · SpectralLoss + λ_grad · GradientLoss

    Parameters
    ----------
    lambda_mse, lambda_spectral, lambda_gradient:
        Pondérations des termes. Valeurs par défaut basées sur Höhlein et al. 2020.
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_spectral: float = 0.1,
        lambda_gradient: float = 0.05,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_spectral = lambda_spectral
        self.lambda_gradient = lambda_gradient
        self.mse = nn.MSELoss()
        self.spectral = SpectralLoss()
        self.gradient = GradientLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mse = self.mse(pred, target)
        spec = self.spectral(pred, target)
        grad = self.gradient(pred, target)
        total = self.lambda_mse * mse + self.lambda_spectral * spec + self.lambda_gradient * grad
        breakdown = {"mse": mse.item(), "spectral": spec.item(), "gradient": grad.item()}
        return total, breakdown


# ---------------------------------------------------------------------------
# Scheduler cosine avec warmup
# ---------------------------------------------------------------------------

def cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Métriques de validation
# ---------------------------------------------------------------------------

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """RMSE, MAE et biais par canal."""
    diff = pred - target
    rmse = (diff ** 2).mean(dim=(0, 2, 3)).sqrt()   # (C,)
    mae = diff.abs().mean(dim=(0, 2, 3))             # (C,)
    bias = diff.mean(dim=(0, 2, 3))                  # (C,)
    return {
        "rmse": rmse.mean().item(),
        "mae": mae.mean().item(),
        "bias": bias.mean().item(),
    }


# ---------------------------------------------------------------------------
# Trainer Lightning (callbacks + logger pilotés par la config cluster)
# ---------------------------------------------------------------------------

def _build_logger(checkpoint_dir: Path):
    """MLflow si ``MLFLOW_TRACKING_URI`` est défini, sinon CSV local (sans dép)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        try:
            from lightning.pytorch.loggers import MLFlowLogger

            return MLFlowLogger(experiment_name="downscaling", tracking_uri=uri)
        except Exception:  # pragma: no cover - dépend de l'install mlflow
            log.warning("MLflow indisponible — repli sur CSVLogger.")
    from lightning.pytorch.loggers import CSVLogger

    return CSVLogger(save_dir=str(checkpoint_dir), name="logs")


def build_trainer(cluster: dict, *, max_epochs: int, patience: int,
                  checkpoint_dir: str | Path) -> pl.Trainer:
    """Construit le ``pl.Trainer`` : accelerator/precision viennent de ``cluster=``."""
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir), filename="best_model",
            monitor="val/rmse", mode="min", save_top_k=1,
        ),
        EarlyStopping(monitor="val/rmse", mode="min", patience=patience),
    ]
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=cluster.get("accelerator", "auto"),
        devices=cluster.get("devices", 1),
        precision=cluster.get("precision", "32-true"),
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=_build_logger(checkpoint_dir),
        log_every_n_steps=1,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Entraînement du U-Net de descente d'échelle")
    p.add_argument("--override", nargs="*", default=[],
                   help="Overrides Hydra (ex: dl.base_ch=32 dl.n_levels=3)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no-film", action="store_true", help="Désactive FiLM (ablation)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    cfg = load_config(args.override)

    dl_cfg = cfg.get("deep_learning", {})
    cluster = cfg.get("cluster", {})
    data_dir = Path(args.data_dir)

    # Fichiers d'entraînement
    coarse_files = sorted((data_dir / "coarse").glob("*.nc"))
    fine_files = sorted((data_dir / "fine").glob("*.nc"))
    dem_file = data_dir / dl_cfg.get("dem_attributes_file", "dem_attributes.nc")

    if not coarse_files:
        raise FileNotFoundError(f"Aucun fichier coarse dans {data_dir}/coarse/")

    met_vars = dl_cfg.get("met_vars", ["t2m", "tp", "u10", "v10", "sp"])
    patch_size = dl_cfg.get("patch_size", 64)
    stats_file = Path(args.checkpoint_dir) / "normalization_stats.json"

    dataset = DownscalingDataset(
        coarse_files=coarse_files,
        fine_files=fine_files,
        dem_file=dem_file,
        met_vars=met_vars,
        patch_size=patch_size,
        stats_file=stats_file,
    )

    if not stats_file.exists():
        log.info("Calcul des statistiques de normalisation…")
        dataset.compute_stats()

    model = build_model(
        architecture=dl_cfg.get("architecture", "unet"),
        met_in_ch=len(met_vars),
        dem_in_ch=dl_cfg.get("dem_in_ch", 4),
        base_ch=dl_cfg.get("base_ch", 64),
        n_levels=dl_cfg.get("n_levels", 4),
        use_film=not args.no_film,
    )

    # Import tardif : Lightning n'est requis que pour l'entraînement effectif.
    from .lightning_module import DownscalingDataModule, DownscalingLitModule

    lit = DownscalingLitModule(
        model,
        lr=args.lr,
        weight_decay=dl_cfg.get("weight_decay", 1e-4),
        warmup_epochs=dl_cfg.get("warmup_epochs", 5),
        max_epochs=args.epochs,
        loss_weights=dl_cfg.get("loss_weights"),
    )
    datamodule = DownscalingDataModule(
        dataset,
        batch_size=args.batch_size,
        num_workers=cluster.get("num_workers", 0),
    )
    trainer = build_trainer(
        cluster,
        max_epochs=args.epochs,
        patience=dl_cfg.get("patience", 15),
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.fit(lit, datamodule=datamodule)

    best = trainer.checkpoint_callback.best_model_score
    log.info("Entraînement terminé. Meilleur val/rmse : %s",
             f"{best:.4f}" if best is not None else "n/a")


if __name__ == "__main__":
    main()
