"""Calibration sparse du U-Net FiLM sur capteurs in situ (chemin B, étage C).

Chemin **déployable** indépendant de Prithvi/MERRA-2 :

    CERRA 5.5 km → U-Net FiLM (conditionné MNT) → 1 km → calibration capteurs

Ce module ajuste la sortie 1 km du U-Net pour qu'elle reproduise les Tmin des
stations Sencrop / Netatmo, avec **correction d'altitude** (MNT) au point de
calibration — cf. ``docs/architecture.md`` §4. Réutilise la loss sparse
elevation-aware (:class:`SparseSupervisedLoss`) et les helpers de stations.

Le U-Net est appelé ``model(x_met, x_dem) → (B, C_out, H, W)`` ; on sélectionne
le canal cible (T2m) avant la supervision sparse.
"""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from downscaling.prtihvi_wxc.lightning_finetune import SparseSupervisedLoss
from downscaling.prtihvi_wxc.netatmo_qc import NetatmoNocturnalQC
from downscaling.prtihvi_wxc.sencrop import load_sencrop
from downscaling.prtihvi_wxc.stations import night_station_targets

from .train import cosine_with_warmup

KELVIN = 273.15


def unet_sparse_collate(samples: list[dict]) -> dict:
    """Collate batch=1 : empile les entrées denses, liste les obs sparse."""
    out = {
        "x_met": torch.stack([s["x_met"] for s in samples]),
        "x_dem": torch.stack([s["x_dem"] for s in samples]),
        "obs_tmin": [s["obs_tmin"] for s in samples],
        "obs_row": [s["obs_row"] for s in samples],
        "obs_col": [s["obs_col"] for s in samples],
        "obs_dz": [s.get("obs_dz") for s in samples],
        "date": [s["date"] for s in samples],
    }
    # FiLM conditioning scalaire (régime + ERA5 + saisonnalité), optionnel
    if samples and samples[0].get("cond_vec") is not None:
        out["cond_vec"] = torch.stack([s["cond_vec"] for s in samples])
    # Champs d'enveloppe physique (CERRA, SURFEX) pour la tête bornée (#81), optionnel.
    if samples and samples[0].get("x_env") is not None:
        out["x_env"] = torch.stack([s["x_env"] for s in samples])
    return out


class UNetSparseCalibrationModule(pl.LightningModule):
    """Calibre un U-Net ``(x_met, x_dem) → champ multi-canaux`` sur stations sparse."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        target_channel: int = 0,  # indice du canal supervisé (T2m)
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        loss_weights: dict | None = None,
        loss_quantile: float | None = None,
        kelvin_to_celsius: bool = True,
        denorm: tuple[float, float] | None = None,  # (µ, σ) t2m : dénormalise la sortie → °C
        lapse_rate: float = -6.5e-3,
        elevation_aware: bool = True,
        hourly: bool = False,  # descente horaire puis réduction (Tmin correct)
        reduce: str = "min",  # réduction temporelle des prédictions ('min' = Tmin)
        target_mode: str = "raw",  # 'raw' = Tmin directe ; 'residual' = first-guess + résidu
        first_guess: str = "bilinear",  # first-guess du mode résidu : 'bilinear' | 'lapse'
        first_guess_dz: torch.Tensor | None = None,  # (H, W) = z_DEM_1km − z_orog_coarse_1km (m)
        clamp: bool = False,  # #81 : borne la sortie dans l'enveloppe physique (x_env)
        clamp_margin: float = 0.0,  # marge °C ajoutée à la demi-largeur d'enveloppe
    ):
        super().__init__()
        self.model = model
        self.denorm = tuple(denorm) if denorm is not None else None
        if target_mode not in ("raw", "residual"):
            raise ValueError(f"target_mode must be 'raw' or 'residual', got {target_mode!r}")
        if first_guess not in ("bilinear", "lapse"):
            raise ValueError(f"first_guess must be 'bilinear' or 'lapse', got {first_guess!r}")
        if first_guess == "lapse" and first_guess_dz is None:
            raise ValueError(
                "first_guess='lapse' requiert first_guess_dz = z_DEM_1km − z_orog_coarse_1km (m)"
            )
        lw = loss_weights or {}
        self.criterion = SparseSupervisedLoss(
            lambda_obs=lw.get("obs", 1.0),
            lambda_tv=lw.get("tv", 0.01),
            lambda_smooth=lw.get("smooth", 0.001),
            loss_quantile=loss_quantile,
        )
        self.target_channel = target_channel
        self.kelvin_to_celsius = kelvin_to_celsius
        self.lapse_rate = lapse_rate
        self.elevation_aware = elevation_aware
        self.hourly = hourly
        self.reduce = reduce
        self.target_mode = target_mode
        self.first_guess = first_guess
        self.clamp = clamp
        self.clamp_margin = float(clamp_margin)
        # dz = (z_DEM_1km − z_orog_coarse_1km) en mètres, STATIQUE (orographie et DEM
        # time-invariants). Registré en buffer → suit le device (MPS/CUDA). None en
        # mode bilinear.
        if first_guess_dz is not None:
            self.register_buffer("first_guess_dz", first_guess_dz.float())
        else:
            self.first_guess_dz = None
        self.save_hyperparameters(ignore=["model", "first_guess_dz"])

    def forward(self, x_met: torch.Tensor, x_dem: torch.Tensor) -> torch.Tensor:
        return self.model(x_met, x_dem)

    def _reduce_time(self, series: torch.Tensor) -> torch.Tensor:
        """``(T, H, W) → (H, W)`` : min (Tmin) / max / mean sur les heures."""
        if self.reduce == "min":
            return series.min(dim=0).values
        if self.reduce == "max":
            return series.max(dim=0).values
        return series.mean(dim=0)

    def _predict_target(self, batch) -> torch.Tensor:
        """Champ cible 1 km ``(1, 1, H, W)`` — descente horaire + réduction si ``hourly``."""
        c = self.target_channel
        cond_vec = batch.get("cond_vec")  # (B, cond_dim) ou None
        if self.hourly:
            # x_met : (1, T, C, H, W) → descente d'échelle heure par heure.
            xm = batch["x_met"][0]  # (T, C, H, W)
            xd = batch["x_dem"].expand(xm.shape[0], -1, -1, -1)  # (T, C_dem, H, W)
            if cond_vec is not None:
                # broadcast (B=1, cond_dim) → (T, cond_dim) pour chaque step
                cv = cond_vec.expand(xm.shape[0], -1) if cond_vec.dim() == 2 else cond_vec
                series = self.model(xm, xd, cv)[:, c]
            else:
                series = self.model(xm, xd)[:, c]  # (T, H, W)
            pred = self._reduce_time(series)[None, None]  # (1, 1, H, W)
        else:
            if cond_vec is not None:
                pred = self.model(batch["x_met"], batch["x_dem"], cond_vec)[:, c : c + 1]
            else:
                pred = self(batch["x_met"], batch["x_dem"])[:, c : c + 1]  # (1, 1, H, W)
        # Tête BORNÉE (#81) : la sortie vit dans l'enveloppe physique [lo, hi] des
        # champs de `x_env` (CERRA, SURFEX). Résidu tanh-borné, différentiable :
        #   T = centre + demi_largeur · tanh(raw) ∈ [lo − m, hi + m]
        # `raw` = sortie CNN normalisée (O(1), idéale pour tanh) ; le biais grande
        # échelle est porté par le centre d'enveloppe. Risque de modèle plafonné
        # par l'écart |CERRA − SURFEX| → composant assurable, explicabilité bornée.
        if self.clamp:
            return self._bounded_head(pred, batch)
        # La sortie du U-Net est en espace NORMALISÉ (z-score des stats d'entraînement).
        # Dénormaliser → °C (les cibles d'entraînement sont en °C). À défaut de stats,
        # repli historique : soustraction Kelvin (suppose une sortie physique en K).
        if self.denorm is not None:
            mean, std = self.denorm
            pred = pred * std + mean
        elif self.kelvin_to_celsius:
            pred = pred - KELVIN
        # Résidu (v2) : le U-Net apprend la CORRECTION à un first-guess physique
        # (CERRA descendu à 1 km = canal météo cible), au lieu de re-prédire toute
        # la Tmin. `pred = first_guess + résidu`. Plancher : résidu → 0 ⇒ on retombe
        # sur le first-guess. Le biais grande-échelle est porté par le first-guess,
        # le modèle ne façonne que la structure fine (cuvette, radiatif).
        # NB altitude : le first-guess porte déjà la valeur CERRA ; la correction
        # station↔maille (lapse_rate·obs_dz) reste gérée UNE seule fois dans la loss.
        if self.target_mode == "residual":
            pred = self._first_guess(batch) + pred
        return pred

    def _bounded_head(self, raw: torch.Tensor, batch) -> torch.Tensor:
        """Sortie bornée tanh dans l'enveloppe de ``x_env`` (#81, variante c).

        ``raw`` = ``(1, 1, H, W)`` sortie CNN (normalisée). ``x_env`` = ``(1, K, H, W)``
        champs physiques (CERRA, SURFEX) en °C. Retourne
        ``centre + demi_largeur·tanh(raw)`` ∈ ``[min(env) − m, max(env) + m]``,
        prouvablement dans l'enveloppe (m = ``clamp_margin``).
        """
        env = batch.get("x_env")
        if env is None:
            raise RuntimeError("clamp=True requiert x_env (champs d'enveloppe) dans le batch")
        lo = env.min(dim=1, keepdim=True).values  # (1, 1, H, W)
        hi = env.max(dim=1, keepdim=True).values
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) + self.clamp_margin
        return center + half * torch.tanh(raw)

    def _first_guess(self, batch) -> torch.Tensor:
        """First-guess physique ``(1, 1, H, W)``.

        - ``bilinear`` : canal météo cible (CERRA bilinéaire à 1 km, °C). Plancher
          faible (≈ valeur coarse brute, biais chaud) — le résidu part quasi de zéro.
        - ``lapse``    : + correction lapse-rate GRILLE
          ``lapse_rate·(z_DEM_1km − z_orog_coarse_1km)`` → plancher ≈ Lot B (~1,6°C).
          ⚠️ Correction GRILLE (maille 1 km vs orographie coarse), DISTINCTE de la
          correction STATION↔maille (``lapse_rate·obs_dz``) faite dans la loss : deux
          ``dz`` différents, pas de double-comptage.
        """
        c = self.target_channel
        xm = batch["x_met"]
        if self.hourly:
            # (1, T, C, H, W) → réduction temporelle du canal cible (même réduction
            # que les prédictions : min = Tmin).
            base = self._reduce_time(xm[0, :, c])[None, None]
        else:
            base = xm[:, c : c + 1]  # (1, 1, H, W)
        if self.first_guess == "lapse":
            if self.first_guess_dz is None:
                raise RuntimeError("first_guess='lapse' mais first_guess_dz absent")
            # (H, W) broadcast → (1, 1, H, W)
            base = base + self.lapse_rate * self.first_guess_dz
        return base

    def _shared_step(self, batch):
        pred = self._predict_target(batch)
        obs_dz = batch.get("obs_dz", [None])[0] if self.elevation_aware else None
        return self.criterion(
            pred,
            batch["obs_tmin"][0],
            batch["obs_row"][0],
            batch["obs_col"][0],
            obs_dz=obs_dz,
            lapse_rate=self.lapse_rate,
        )

    @staticmethod
    def _is_empty(batch) -> bool:
        """Nuit sans station dans ce rôle (leave-station-out) → sample vide à sauter."""
        obs = batch["obs_tmin"][0]
        return int(getattr(obs, "numel", lambda: len(obs))()) == 0

    def training_step(self, batch, batch_idx):
        if self._is_empty(batch):
            return None  # Lightning saute le batch (pas d'obs dans ce rôle)
        loss, parts = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=1)
        # Always log RMSE in degrees Celsius for comparability across loss types
        self.log(
            "train/rmse", parts.get("rmse_obs", parts["loss_obs"]), prog_bar=True, batch_size=1
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self._is_empty(batch):
            return None  # nuit val sans station held-out → skip (n'entre pas dans val/rmse)
        loss, parts = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, batch_size=1)
        # Always log RMSE in °C for comparability across loss types (EarlyStopping monitors this)
        self.log("val/rmse", parts.get("rmse_obs", parts["loss_obs"]), prog_bar=True, batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
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


class UNetSparseDataModule(pl.LightningDataModule):
    """Split train/val déterministe + ``DataLoader`` batch=1 (collate sparse)."""

    def __init__(
        self, dataset: Dataset, *, num_workers: int = 0, val_fraction: float = 0.2, seed: int = 42
    ):
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None

    def setup(self, stage: str | None = None):
        # Leave-station-out (#33) : si le dataset porte un holdout_bbox et sait se
        # cloner par rôle, train/val couvrent les MÊMES nuits mais des stations
        # DISJOINTES (train = hors bbox, val = dans bbox) → aucune fuite station.
        if getattr(self.dataset, "holdout_bbox", None) is not None and hasattr(
            self.dataset, "with_role"
        ):
            self.train_ds = self.dataset.with_role("train")
            self.val_ds = self.dataset.with_role("val")
            return
        # Fallback : split aléatoire par nuit (rétro-compat, fuite station connue).
        n_val = max(1, int(len(self.dataset) * self.val_fraction))
        n_train = len(self.dataset) - n_val
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds = random_split(
            self.dataset, [n_train, n_val], generator=generator
        )

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=unet_sparse_collate,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)


class UNetStationDataset(Dataset):
    """Dataset de calibration sparse pour le U-Net : 1 sample = 1 nuit.

    ``coarse_provider(date) -> (x_met, x_dem)`` fournit les entrées U-Net de la
    nuit (CERRA dégradé / ERA5 + attributs MNT) — c'est le **point d'intégration**
    avec les données réelles. Les capteurs (Sencrop par défaut) fournissent les
    Tmin sparse ; ``elevation_grid`` (altitude maille HR, m) active la correction
    d'altitude (``obs_dz``).
    """

    def __init__(
        self,
        dates,
        coarse_provider,
        obs_dir: str | Path,
        lat_grid,
        lon_grid,
        *,
        obs_loader=load_sencrop,
        file_template: str = "sencrop_{date}.csv",
        elevation_grid=None,
        min_stations: int = 5,
        lapse_rate: float = -6.5e-3,
        holdout_bbox: tuple[float, float, float, float] | None = None,
        role: str = "all",
        surfex_provider=None,  # #81 : callable date -> (H, W) champ SURFEX °C (enveloppe)
    ):
        self.coarse_provider = coarse_provider
        self.surfex_provider = surfex_provider
        self.obs_dir = Path(obs_dir)
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.obs_loader = obs_loader
        self.file_template = file_template
        self.elevation_grid = elevation_grid
        self.min_stations = min_stations
        self.qc = NetatmoNocturnalQC(lapse_rate=lapse_rate)
        self.holdout_bbox = holdout_bbox
        self.role = role
        self.dates = [d for d in dates if self._path(d).exists()]

    def with_role(self, role: str) -> UNetStationDataset:
        """Clone superficiel (partage grilles/provider) avec un rôle leave-station-out."""
        import copy

        clone = copy.copy(self)
        clone.role = role
        return clone

    def _path(self, date: str) -> Path:
        return self.obs_dir / self.file_template.format(date=date)

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict:
        date = self.dates[idx]
        x_met, x_dem = self.coarse_provider(date)
        sample_env = None
        if self.surfex_provider is not None:
            # enveloppe = [CERRA 1km (canal météo 0), SURFEX 1km], même grille fine, °C
            surfex = self.surfex_provider(date)  # (H, W)
            cerra = x_met[0]  # (H, W)
            sample_env = torch.stack(
                [cerra, torch.as_tensor(surfex, dtype=torch.float32)], dim=0
            )  # (2, H, W)
        obs_qc = self.qc.run(self.obs_loader(str(self._path(date)), date))
        tmin, row, col, dz = night_station_targets(
            obs_qc,
            self.lat_grid,
            self.lon_grid,
            self.elevation_grid,
            holdout_bbox=self.holdout_bbox,
            role=self.role,
        )
        sample = {
            "x_met": x_met,
            "x_dem": x_dem,
            "obs_tmin": torch.as_tensor(tmin, dtype=torch.float32),
            "obs_row": torch.as_tensor(row, dtype=torch.long),
            "obs_col": torch.as_tensor(col, dtype=torch.long),
            "obs_dz": torch.as_tensor(dz, dtype=torch.float32),
            "date": date,
        }
        if sample_env is not None:
            sample["x_env"] = sample_env
        return sample
