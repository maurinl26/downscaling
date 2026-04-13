"""
inference.py — Inférence Prithvi WxC pour réanalyse des nuits de gel.

Implémente le rolling temporel à 3h décrit dans Yu et al. (2025) :
  - Entrée : deux timestamps ERA5 (t0, t0+3h)
  - Prédiction : t0+6h à haute résolution
  - Rolling : la prédiction devient le second timestamp pour l'étape suivante
  - Couvrir une nuit complète (20h → 08h) = 8 itérations

Sortie finale :
  - Tmin nocturne haute résolution (1 km) par nuit
  - Fichier Zarr partitionné par mois, compatible DuckDB

Usage :
    python scripts/run_prithvi_inference.py \\
        --config config/prithvi_wxc_drome.yml \\
        --checkpoint checkpoints/prithvi_adapter.pt \\  # optionnel
        --start 2000-01-01 \\
        --end  2024-12-31 \\
        --out  output/frost_reanalysis_prithvi.zarr
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader

from downscaling.deep_learning.prithvi_wxc.loader import PrithviWxCDownscaler
from downscaling.deep_learning.prithvi_wxc.dataset import FrostNightDataset
from downscaling.shared.indices import spring_frost, frost_hours

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration par défaut
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "batch_size": 4,
    "num_workers": 2,
    "scale_factor": 6,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "frost_threshold_celsius": 0.0,
    "gdd_threshold": 50,   # Débourrement approximatif amandier/abricotier
    "output_chunks": {"time": 30, "lat": 128, "lon": 128},
}


# ---------------------------------------------------------------------------
# Boucle d'inférence principale
# ---------------------------------------------------------------------------

class FrostReanalysisRunner:
    """
    Orchestrateur de la réanalyse gel avec Prithvi WxC.

    Produit pour chaque nuit :
      - T2m_min : minimum nocturne haute résolution (K)
      - frost_flag : booléen T2m_min < seuil (binaire 0/1)
      - spring_frost_flag : gel après débourrement (GDD > seuil)

    Compatible avec le schéma de sortie du pipeline statistique existant
    (shared/indices.py) pour comparaison directe basis risk.
    """

    def __init__(self, config: dict):
        self.config = {**DEFAULT_CONFIG, **config}
        self.device = self.config["device"]
        log.info(f"Runner initialisé sur {self.device}")

    def load_model(
        self,
        checkpoint_path: str | Path | None = None,
        use_granite: bool = True,
    ) -> PrithviWxCDownscaler:
        return PrithviWxCDownscaler.from_pretrained(
            checkpoint_path=checkpoint_path,
            use_granite_downscaling=use_granite,
            scale_factor=self.config["scale_factor"],
            device=self.device,
        )

    def run(
        self,
        model: PrithviWxCDownscaler,
        dataset: FrostNightDataset,
        output_zarr: str | Path,
    ) -> xr.Dataset:
        """
        Exécute la réanalyse complète et écrit les résultats en Zarr.

        Returns:
            xr.Dataset avec variables : T2m_min, frost_flag, spring_frost_flag.
        """
        output_zarr = Path(output_zarr)
        output_zarr.parent.mkdir(parents=True, exist_ok=True)

        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            pin_memory=(self.device == "cuda"),
            collate_fn=_frost_collate_fn,
        )

        # Accumulateur : Tmin par nuit (date → tenseur HR)
        nightly_tmin: dict[str, np.ndarray] = {}

        log.info(f"Démarrage inférence : {len(dataset)} paires temporelles")

        with torch.no_grad():
            for batch in loader:
                era5_t0 = batch["era5_t0"].to(self.device)    # (B, C, H, W)
                era5_t1 = batch["era5_t1"].to(self.device)
                dem_hr = batch["dem_hr"].to(self.device)       # (B, 3, H_hr, W_hr)
                times: list[pd.Timestamp] = batch["valid_time"]

                # Prédiction T2m haute résolution (B, 1, H_hr, W_hr)
                t2m_hr = model(era5_t0, era5_t1, dem_hr)

                # T2m en Kelvin → Celsius pour les indices
                t2m_celsius = (t2m_hr - 273.15).squeeze(1).cpu().numpy()

                # Accumuler par nuit (date sans heure)
                for i, ts in enumerate(times):
                    night_key = ts.date().isoformat()
                    if night_key not in nightly_tmin:
                        nightly_tmin[night_key] = []
                    nightly_tmin[night_key].append(t2m_celsius[i])

        log.info(f"{len(nightly_tmin)} nuits traitées — calcul Tmin et indices...")

        return self._build_output_dataset(
            nightly_tmin, dataset.dem_hr, dataset.ds, output_zarr
        )

    # ------------------------------------------------------------------
    # Rolling inference pour une nuit complète (méthode Yu et al.)
    # ------------------------------------------------------------------

    def rolling_night_inference(
        self,
        model: PrithviWxCDownscaler,
        era5_t0: torch.Tensor,
        era5_t1: torch.Tensor,
        dem_hr: torch.Tensor,
        n_steps: int = 8,
    ) -> torch.Tensor:
        """
        Rolling temporel sur n_steps × 3h = 24h.
        Couvre une nuit de 20h UTC à 20h UTC+1 (8 pas de 3h).

        Selon Yu et al. (2025) : le second timestamp devient le premier
        à l'étape suivante, et la prédiction devient le second timestamp.

        Args:
            era5_t0: (1, C, H_lr, W_lr) — premier timestamp
            era5_t1: (1, C, H_lr, W_lr) — second timestamp (t0+3h)
            dem_hr:  (1, 3, H_hr, W_hr) — DEM
            n_steps: nombre d'itérations (8 pour une journée complète)

        Returns:
            Tensor (n_steps, H_hr, W_hr) — T2m HR à chaque pas de temps
        """
        model.eval()
        predictions = []

        current_t0 = era5_t0
        current_t1 = era5_t1

        with torch.no_grad():
            for step in range(n_steps):
                # Prédiction haute résolution
                t2m_pred = model(current_t0, current_t1, dem_hr)  # (1,1,H,W)
                predictions.append(t2m_pred.squeeze(0).squeeze(0))  # (H,W)

                # Rolling : t1 devient t0, prédiction upsampled devient t1
                # On reprojecte la prédiction HR → LR pour le prochain pas
                pred_lr = _downsample_to_lr(
                    t2m_pred, target_shape=current_t1.shape[-2:]
                )
                current_t0 = current_t1
                # Remplacer le canal T2m dans t1 par la prédiction
                current_t1 = current_t1.clone()
                current_t1[:, 0:1, :, :] = pred_lr  # canal 0 = T2m

        return torch.stack(predictions, dim=0)  # (n_steps, H_hr, W_hr)

    # ------------------------------------------------------------------
    # Construction du dataset de sortie
    # ------------------------------------------------------------------

    def _build_output_dataset(
        self,
        nightly_tmin: dict[str, list[np.ndarray]],
        dem_hr: torch.Tensor,
        ds_lr: xr.Dataset,
        output_zarr: Path,
    ) -> xr.Dataset:
        """
        Assemble les résultats en xr.Dataset et écrit en Zarr.
        """
        dates = sorted(nightly_tmin.keys())
        H_hr, W_hr = dem_hr.shape[-2:]

        tmin_arrays = []
        frost_flags = []
        threshold_k = self.config["frost_threshold_celsius"] + 273.15

        for date in dates:
            night_preds = np.stack(nightly_tmin[date], axis=0)  # (steps, H, W)
            tmin = night_preds.min(axis=0)  # (H, W) — Tmin nocturne
            tmin_arrays.append(tmin)
            frost_flags.append((tmin < threshold_k - 273.15).astype(np.int8))

        tmin_arr = np.stack(tmin_arrays, axis=0)    # (days, H_hr, W_hr)
        frost_arr = np.stack(frost_flags, axis=0)   # (days, H_hr, W_hr)

        # Coordonnées haute résolution interpolées depuis ERA5
        lat_lr = ds_lr.get("latitude", ds_lr.get("lat")).values
        lon_lr = ds_lr.get("longitude", ds_lr.get("lon")).values
        lat_hr = np.linspace(lat_lr.max(), lat_lr.min(), H_hr)
        lon_hr = np.linspace(lon_lr.min(), lon_lr.max(), W_hr)

        ds_out = xr.Dataset(
            {
                "T2m_min_night": xr.DataArray(
                    tmin_arr,
                    dims=["time", "lat", "lon"],
                    coords={
                        "time": pd.to_datetime(dates),
                        "lat": lat_hr,
                        "lon": lon_hr,
                    },
                    attrs={
                        "units": "degC",
                        "long_name": "Minimum nocturne T2m haute résolution",
                        "model": "Prithvi WxC + CNN adapter",
                        "source": "NASA/IBM Prithvi WxC foundation model",
                    },
                ),
                "frost_flag": xr.DataArray(
                    frost_arr,
                    dims=["time", "lat", "lon"],
                    attrs={
                        "units": "1",
                        "long_name": f"Gel nocturne (T2m_min < {self.config['frost_threshold_celsius']}°C)",
                    },
                ),
            },
            attrs={
                "title": "Réanalyse nuits de gel — Drôme/Ardèche",
                "institution": "Karpos / Prokarpos",
                "method": "Prithvi WxC downscaling (NASA/IBM) + DEM orographique",
                "scale_factor": self.config["scale_factor"],
                "bbox": str(
                    {
                        "lat_min": float(lat_hr.min()),
                        "lat_max": float(lat_hr.max()),
                        "lon_min": float(lon_hr.min()),
                        "lon_max": float(lon_hr.max()),
                    }
                ),
            },
        )

        # Écriture Zarr avec chunks compatibles DuckDB
        chunks = self.config["output_chunks"]
        encoding = {
            "T2m_min_night": {"chunks": (chunks["time"], chunks["lat"], chunks["lon"])},
            "frost_flag": {"chunks": (chunks["time"], chunks["lat"], chunks["lon"])},
        }

        log.info(f"Écriture Zarr → {output_zarr}")
        ds_out.chunk(chunks).to_zarr(str(output_zarr), mode="w")

        log.info(
            f"Réanalyse terminée : {len(dates)} nuits, "
            f"{frost_arr.sum()} nuits de gel détectées "
            f"({100 * frost_arr.mean():.1f}% du total)"
        )

        return ds_out


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _frost_collate_fn(samples):
    """Collate function préservant les timestamps Python."""
    return {
        "era5_t0": torch.stack([s.era5_t0 for s in samples]),
        "era5_t1": torch.stack([s.era5_t1 for s in samples]),
        "dem_hr": torch.stack([s.dem_hr for s in samples]),
        "valid_time": [s.valid_time for s in samples],
    }


def _downsample_to_lr(
    tensor_hr: torch.Tensor,
    target_shape: tuple[int, int],
) -> torch.Tensor:
    """Reprojette un tenseur HR → LR (pour rolling step)."""
    return torch.nn.functional.interpolate(
        tensor_hr,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {**DEFAULT_CONFIG, **cfg}
