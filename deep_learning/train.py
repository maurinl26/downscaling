"""
Boucle d'entraînement du modèle de descente d'échelle par deep learning.

Fonctionnalités
---------------
- Loss composite : MSE + loss spectrale (sur T2m) + loss de gradient spatial
- Scheduler cosine annealing avec warmup linéaire
- Sauvegarde du meilleur modèle (val RMSE)
- Logging tensorboard (optionnel)
- Early stopping

Usage
-----
    python -m downscaling.deep_learning.train \
        --config config/drome_ardeche.yml \
        --data-dir data/training/ \
        --epochs 100 --batch-size 8 \
        --checkpoint-dir checkpoints/
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
except ImportError as e:
    raise ImportError("PyTorch requis : pip install torch") from e

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
# Boucle d'entraînement principale
# ---------------------------------------------------------------------------

class Trainer:
    """
    Gère l'entraînement complet du modèle de descente d'échelle.

    Parameters
    ----------
    model:
        Modèle PyTorch (DownscalingUNet ou LightSRCNN).
    train_dataset, val_dataset:
        Datasets d'entraînement et de validation.
    batch_size:
        Taille des lots.
    lr:
        Taux d'apprentissage initial.
    epochs:
        Nombre d'époques maximum.
    checkpoint_dir:
        Répertoire pour sauvegarder les checkpoints.
    device:
        'cuda', 'mps' (Apple Silicon) ou 'cpu'.
    patience:
        Nombre d'époques sans amélioration avant early stopping.
    loss_weights:
        Dictionnaire des pondérations de la loss (mse, spectral, gradient).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: DownscalingDataset,
        val_dataset: DownscalingDataset,
        batch_size: int = 8,
        lr: float = 1e-4,
        epochs: int = 100,
        checkpoint_dir: str | Path = "checkpoints/",
        device: str | None = None,
        patience: int = 15,
        loss_weights: dict | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = patience

        # Device auto-detect
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        log.info(f"Device : {self.device}")

        self.model = self.model.to(self.device)

        # Loss
        lw = loss_weights or {}
        self.criterion = DownscalingLoss(
            lambda_mse=lw.get("mse", 1.0),
            lambda_spectral=lw.get("spectral", 0.1),
            lambda_gradient=lw.get("gradient", 0.05),
        )

        # Optimiseur + scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = cosine_with_warmup(self.optimizer, warmup_epochs=5, total_epochs=epochs)

        self.best_val_rmse = float("inf")
        self.epochs_no_improve = 0

    def train(self) -> dict:
        """Lance l'entraînement complet. Retourne l'historique des métriques."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        history = {"train_loss": [], "val_rmse": [], "val_mae": []}

        for epoch in range(1, self.epochs + 1):
            # ---- Entraînement -------------------------------------------
            self.model.train()
            train_loss = 0.0
            for x_coarse, dem, y_fine in train_loader:
                x_coarse = x_coarse.to(self.device)
                dem = dem.to(self.device)
                y_fine = y_fine.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(x_coarse, dem)
                loss, breakdown = self.criterion(pred, y_fine)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.scheduler.step()

            # ---- Validation ---------------------------------------------
            val_metrics = self._validate(val_loader)
            val_rmse = val_metrics["rmse"]

            history["train_loss"].append(train_loss)
            history["val_rmse"].append(val_rmse)
            history["val_mae"].append(val_metrics["mae"])

            log.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_rmse:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}"
            )

            # ---- Checkpoint + early stopping ----------------------------
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_rmse, best=True)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    log.info(f"Early stopping à l'époque {epoch} (patience={self.patience}).")
                    break

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_rmse, best=False)

        return history

    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_coarse, dem, y_fine in loader:
                x_coarse = x_coarse.to(self.device)
                dem = dem.to(self.device)
                pred = self.model(x_coarse, dem)
                all_preds.append(pred.cpu())
                all_targets.append(y_fine)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        return compute_metrics(preds, targets)

    def _save_checkpoint(self, epoch: int, val_rmse: float, best: bool):
        name = "best_model.pt" if best else f"checkpoint_epoch{epoch:04d}.pt"
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_rmse": val_rmse,
            },
            path,
        )
        if best:
            log.info(f"  → Meilleur modèle sauvegardé : {path} (val_rmse={val_rmse:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Entraînement du U-Net de descente d'échelle")
    p.add_argument("--config", required=True, help="Fichier de configuration YAML")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default=None)
    p.add_argument("--no-film", action="store_true", help="Désactive FiLM (ablation)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dl_cfg = cfg.get("deep_learning", {})
    domain = cfg.get("domain", {})
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

    # Split train / val (80/20)
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    model = build_model(
        architecture=dl_cfg.get("architecture", "unet"),
        met_in_ch=len(met_vars),
        dem_in_ch=dl_cfg.get("dem_in_ch", 4),
        base_ch=dl_cfg.get("base_ch", 64),
        n_levels=dl_cfg.get("n_levels", 4),
        use_film=not args.no_film,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        patience=dl_cfg.get("patience", 15),
    )

    history = trainer.train()
    log.info(f"Entraînement terminé. Meilleur val RMSE : {trainer.best_val_rmse:.4f}")


if __name__ == "__main__":
    main()
