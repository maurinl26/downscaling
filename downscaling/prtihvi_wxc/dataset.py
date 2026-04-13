"""
dataset.py — Dataset PyTorch pour inférence Prithvi WxC en mode downscaling.

Gère :
  - Chargement ERA5 ou CERRA (NetCDF/Zarr) sur la fenêtre Drôme/Ardèche
  - Paires temporelles (t, t+3h) comme attendu par Prithvi WxC
  - DEM haute résolution (COP-DEM) préprocessé : élévation + pente + exposition
  - Filtrage sur les nuits de gel (mois d'octobre à mai, heures 20h–08h UTC)
  - Normalisation avec climatologie MERRA-2

Référence architecture : Yu et al. (2025), NASA NTRS 20250006603
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Fenêtre spatiale Drôme / Ardèche
# ---------------------------------------------------------------------------
BBOX_DROME_ARDECHE = {
    "lat_min": 44.0,
    "lat_max": 45.5,
    "lon_min": 4.0,
    "lon_max": 5.5,
}

# Mois de la saison de gel arboricole (octobre → mai inclus)
FROST_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]

# Heures UTC correspondant aux nuits (20h → 08h)
FROST_HOURS_UTC = list(range(20, 24)) + list(range(0, 9))

# Variables ERA5 surface utilisées (subset compatible Prithvi WxC MERRA-2 mapping)
ERA5_VARS = ["t2m", "u10", "v10", "sp", "q", "tcwv"]

# Mapping ERA5 → MERRA-2 pour normalisation avec climatologie Prithvi WxC
ERA5_TO_MERRA2 = {
    "t2m": "T2M",
    "u10": "U10M",
    "v10": "V10M",
    "sp": "PS",
    "q": "QV2M",
    "tcwv": "TQV",
}


@dataclass
class FrostNightSample:
    """Un échantillon = une paire de timestamps pour inférence rolling."""
    t0: pd.Timestamp
    t1: pd.Timestamp  # t0 + 3h
    era5_t0: torch.Tensor   # (C, H_lr, W_lr)
    era5_t1: torch.Tensor   # (C, H_lr, W_lr)
    dem_hr: torch.Tensor    # (3, H_hr, W_hr) : élévation, pente, exposition
    valid_time: pd.Timestamp


class FrostNightDataset(Dataset):
    """
    Dataset des paires ERA5/CERRA (t, t+3h) pour les nuits de gel.

    Args:
        era5_path: Chemin vers le fichier NetCDF ou Zarr ERA5/CERRA.
        dem_path: Chemin vers le GeoTIFF COP-DEM haute résolution.
        start_date: Début de la période de réanalyse.
        end_date: Fin de la période de réanalyse.
        source: "era5" ou "cerra".
        climatology_path: Chemin vers la climatologie MERRA-2 (HuggingFace cache
                          ou fichier local). Si None, normalisation min-max.
        frost_only: Si True, filtre uniquement les heures nocturnes d'octobre–mai.
    """

    def __init__(
        self,
        era5_path: str | Path,
        dem_path: str | Path,
        start_date: str = "1985-01-01",
        end_date: str = "2024-12-31",
        source: Literal["era5", "cerra"] = "era5",
        climatology_path: str | Path | None = None,
        frost_only: bool = True,
    ):
        self.era5_path = Path(era5_path)
        self.dem_path = Path(dem_path)
        self.source = source
        self.frost_only = frost_only

        # Charger les données ERA5/CERRA
        print(f"[FrostDataset] Chargement {source} depuis {era5_path} ...")
        self.ds = self._load_reanalysis(start_date, end_date)

        # Construire l'index temporel des paires (t0, t0+3h)
        self.time_pairs = self._build_time_pairs()
        print(
            f"[FrostDataset] {len(self.time_pairs)} paires temporelles "
            f"({'nuits de gel' if frost_only else 'toutes heures'})"
        )

        # Charger et préprocesser le DEM
        self.dem_hr = self._load_dem()
        print(f"[FrostDataset] DEM chargé : {self.dem_hr.shape}")

        # Charger la climatologie pour normalisation
        self.climatology = self._load_climatology(climatology_path)

    # ------------------------------------------------------------------
    # Chargement des données
    # ------------------------------------------------------------------

    def _load_reanalysis(self, start_date: str, end_date: str) -> xr.Dataset:
        """Charge ERA5 ou CERRA et découpe sur la fenêtre Drôme/Ardèche."""
        if self.era5_path.suffix in (".zarr", ""):
            ds = xr.open_zarr(str(self.era5_path))
        else:
            ds = xr.open_dataset(str(self.era5_path))

        # Sélection temporelle
        ds = ds.sel(time=slice(start_date, end_date))

        # Sélection spatiale — gérer les deux conventions lat/lon
        lat_name = "latitude" if "latitude" in ds.dims else "lat"
        lon_name = "longitude" if "longitude" in ds.dims else "lon"

        ds = ds.sel(
            **{
                lat_name: slice(
                    BBOX_DROME_ARDECHE["lat_max"],
                    BBOX_DROME_ARDECHE["lat_min"],
                ),
                lon_name: slice(
                    BBOX_DROME_ARDECHE["lon_min"],
                    BBOX_DROME_ARDECHE["lon_max"],
                ),
            }
        )

        # Garder uniquement les variables utiles (celles disponibles)
        available = [v for v in ERA5_VARS if v in ds.data_vars]
        if not available:
            raise ValueError(
                f"Aucune variable ERA5 trouvée dans {self.era5_path}. "
                f"Variables attendues : {ERA5_VARS}"
            )
        return ds[available]

    def _build_time_pairs(self) -> list[pd.Timestamp]:
        """
        Construit la liste des timestamps t0 pour lesquels t0+3h existe.
        Filtre optionnellement sur les nuits de gel (oct–mai, 20h–08h UTC).
        """
        times = pd.DatetimeIndex(self.ds.time.values)
        # Vérifier que t0+3h existe dans le dataset
        time_set = set(times)
        delta = pd.Timedelta("3h")

        pairs = []
        for t in times:
            if (t + delta) not in time_set:
                continue
            if self.frost_only:
                if t.month not in FROST_MONTHS:
                    continue
                if t.hour not in FROST_HOURS_UTC:
                    continue
            pairs.append(t)

        return pairs

    def _load_dem(self) -> torch.Tensor:
        """
        Charge le COP-DEM et dérive élévation, pente, exposition.
        Retourne un tenseur (3, H_hr, W_hr).
        """
        try:
            import rasterio  # type: ignore
            from rasterio.warp import reproject, Resampling  # type: ignore
        except ImportError:
            raise ImportError("pip install rasterio")

        with rasterio.open(str(self.dem_path)) as src:
            elevation = src.read(1).astype(np.float32)

        # Pente et exposition par gradient discret
        dy, dx = np.gradient(elevation)
        slope = np.sqrt(dx**2 + dy**2).astype(np.float32)
        aspect = np.arctan2(dy, dx).astype(np.float32)

        # Normalisation robuste (percentiles pour éviter outliers falaises)
        def robust_norm(arr: np.ndarray) -> np.ndarray:
            p2, p98 = np.percentile(arr, [2, 98])
            return np.clip((arr - p2) / (p98 - p2 + 1e-6), 0, 1)

        elevation_norm = robust_norm(elevation)
        slope_norm = robust_norm(slope)
        aspect_norm = (aspect + np.pi) / (2 * np.pi)  # [0, 1]

        dem = np.stack([elevation_norm, slope_norm, aspect_norm], axis=0)
        return torch.from_numpy(dem)

    def _load_climatology(
        self, climatology_path: str | Path | None
    ) -> dict[str, dict] | None:
        """
        Charge la climatologie MERRA-2 pour normalisation.
        Si non disponible, retourne None (normalisation min-max sera utilisée).
        """
        if climatology_path is None:
            return None
        try:
            import pickle
            with open(climatology_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.time_pairs)

    def __getitem__(self, idx: int) -> FrostNightSample:
        t0 = self.time_pairs[idx]
        t1 = t0 + pd.Timedelta("3h")

        era5_t0 = self._extract_tensor(t0)
        era5_t1 = self._extract_tensor(t1)

        return FrostNightSample(
            t0=t0,
            t1=t1,
            era5_t0=era5_t0,
            era5_t1=era5_t1,
            dem_hr=self.dem_hr,
            valid_time=t0,
        )

    def _extract_tensor(self, t: pd.Timestamp) -> torch.Tensor:
        """
        Extrait et normalise un snapshot ERA5 à l'instant t.
        Retourne (C, H_lr, W_lr).
        """
        snap = self.ds.sel(time=t)
        arrays = []
        for var in ERA5_VARS:
            if var not in snap:
                # Variable absente : zéro-padding (modèle tolère données partielles)
                arrays.append(
                    np.zeros(
                        (snap.dims.get("latitude", snap.dims.get("lat", 1)),
                         snap.dims.get("longitude", snap.dims.get("lon", 1))),
                        dtype=np.float32,
                    )
                )
                continue

            data = snap[var].values.astype(np.float32)

            # Normalisation
            if self.climatology and ERA5_TO_MERRA2.get(var) in self.climatology:
                merra2_var = ERA5_TO_MERRA2[var]
                mean = self.climatology[merra2_var]["mean"]
                std = self.climatology[merra2_var]["std"]
                data = (data - mean) / (std + 1e-6)
            else:
                # Normalisation min-max par défaut
                data = (data - data.mean()) / (data.std() + 1e-6)

            arrays.append(data)

        return torch.from_numpy(np.stack(arrays, axis=0))

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------

    @property
    def lr_shape(self) -> tuple[int, int, int]:
        """Forme (C, H_lr, W_lr) d'un tenseur basse résolution."""
        return (len(ERA5_VARS),) + self[0].era5_t0.shape[-2:]

    @property
    def hr_shape(self) -> tuple[int, int, int]:
        """Forme (3, H_hr, W_hr) du DEM haute résolution."""
        return tuple(self.dem_hr.shape)

    def get_night_window(self, date: str) -> list[int]:
        """
        Retourne les indices dataset correspondant à une nuit complète.
        Utile pour calculer Tmin sur une nuit donnée.

        Args:
            date: Date au format "YYYY-MM-DD" (nuit du date au date+1).
        """
        target = pd.Timestamp(date)
        next_day = target + pd.Timedelta("1D")
        return [
            i for i, t in enumerate(self.time_pairs)
            if target <= t < next_day and t.hour in FROST_HOURS_UTC
        ]
