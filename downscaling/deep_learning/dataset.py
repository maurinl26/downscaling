"""
Dataset PyTorch pour l'entraînement du modèle de descente d'échelle par DL.

Architecture des données
------------------------
Chaque exemple d'entraînement est un tuple :
    (x_coarse, dem_features, y_fine)

- x_coarse  : champs météo basse résolution rééchantillonnés sur la grille fine
              shape (C_met, H, W)  — e.g. (5, 64, 64)
- dem_features : attributs du MNT haute résolution
              shape (C_dem, H, W)  — élévation + pente + exposition + courbure
- y_fine    : champs météo cibles haute résolution
              shape (C_met, H, W)

Normalisation
-------------
Chaque variable est standardisée (µ, σ) calculés sur le jeu d'entraînement.
Les statistiques sont sauvegardées dans le Dataset pour l'inférence.

Tuilage spatial
---------------
Pour les grands domaines, on découpe en tuiles (patch_size × patch_size) avec
un chevauchement optionnel, puis on recombine à l'inférence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError("PyTorch requis : pip install torch") from e


# Variables météo prises en entrée / sortie par défaut
DEFAULT_MET_VARS = ["t2m", "tp", "u10", "v10", "sp"]
DEFAULT_DEM_VARS = ["elevation", "slope", "aspect", "curvature"]


class DownscalingDataset(Dataset):
    """
    Dataset de paires (ERA5/CERRA basse résolution, haute résolution) pour
    l'entraînement du U-Net conditionné par le MNT.

    Parameters
    ----------
    coarse_files:
        Liste de fichiers NetCDF basse résolution (ERA5 ou CERRA dégradé).
        Chacun couvre une période temporelle (journée, mois…).
    fine_files:
        Fichiers NetCDF haute résolution correspondants (même ordre).
    dem_file:
        Fichier NetCDF des attributs MNT (élévation, pente, exposition…)
        tel que produit par DEMLoader.terrain_attributes().
    met_vars:
        Variables météo à utiliser.
    patch_size:
        Taille des tuiles spatiales (pixels). None = domaine entier.
    stride:
        Pas entre tuiles (pixels). Défaut = patch_size (pas de chevauchement).
    stats_file:
        Chemin JSON pour sauvegarder/charger les statistiques de normalisation.
        Si le fichier existe, il est chargé ; sinon il est créé lors du fit.
    """

    def __init__(
        self,
        coarse_files: list[str | Path],
        fine_files: list[str | Path],
        dem_file: str | Path,
        met_vars: list[str] = DEFAULT_MET_VARS,
        patch_size: int | None = 64,
        stride: int | None = None,
        stats_file: str | Path | None = None,
    ):
        assert len(coarse_files) == len(fine_files), \
            "coarse_files et fine_files doivent avoir la même longueur."

        self.coarse_files = [Path(f) for f in coarse_files]
        self.fine_files = [Path(f) for f in fine_files]
        self.dem_file = Path(dem_file)
        self.met_vars = met_vars
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.stats_file = Path(stats_file) if stats_file else None

        # Chargement du MNT (constant dans le temps)
        self._dem_tensor: torch.Tensor | None = None

        # Statistiques de normalisation {varname: (mean, std)}
        self.stats: dict[str, tuple[float, float]] = {}
        if self.stats_file and self.stats_file.exists():
            self._load_stats()

        # Construction de l'index des tuiles
        self._patches: list[tuple[int, int, int]] = []  # (file_idx, i0, j0)
        self._build_patch_index()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_idx, i0, j0 = self._patches[idx]

        # Chargement des fichiers (on garde en mémoire seulement 1 à la fois)
        coarse = xr.open_dataset(self.coarse_files[file_idx], engine="netcdf4")
        fine = xr.open_dataset(self.fine_files[file_idx], engine="netcdf4")

        # Sélection d'un pas de temps aléatoire si dimension time présente
        if "time" in coarse.dims:
            t = np.random.randint(0, len(coarse.time))
            coarse = coarse.isel(time=t)
            fine = fine.isel(time=t)

        ps = self.patch_size
        i1 = i0 + ps if ps else None
        j1 = j0 + ps if ps else None

        x_coarse = self._to_tensor(coarse, self.met_vars, i0, i1, j0, j1, normalize=True)
        y_fine = self._to_tensor(fine, self.met_vars, i0, i1, j0, j1, normalize=True)
        dem = self._get_dem_patch(i0, i1, j0, j1)

        return x_coarse, dem, y_fine

    # ------------------------------------------------------------------
    def compute_stats(self) -> dict[str, tuple[float, float]]:
        """
        Calcule µ et σ pour chaque variable sur l'ensemble du jeu de données.
        Appeler avant l'entraînement si stats_file n'existe pas.
        """
        accum = {v: [] for v in self.met_vars + DEFAULT_DEM_VARS}

        for fp in self.fine_files:
            ds = xr.open_dataset(fp, engine="netcdf4")
            for v in self.met_vars:
                if v in ds:
                    accum[v].append(ds[v].values.ravel())

        # Attributs MNT
        dem_ds = xr.open_dataset(self.dem_file, engine="netcdf4")
        for v in DEFAULT_DEM_VARS:
            if v in dem_ds:
                accum[v].append(dem_ds[v].values.ravel())

        for v, chunks in accum.items():
            if not chunks:
                continue
            data = np.concatenate(chunks)
            data = data[np.isfinite(data)]
            self.stats[v] = (float(data.mean()), float(data.std()) + 1e-8)

        if self.stats_file:
            self._save_stats()
        return self.stats

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _build_patch_index(self):
        """Construit l'index (file_idx, i0, j0) de toutes les tuiles."""
        if not self.fine_files:
            return
        # Détermine la taille du domaine à partir du premier fichier
        ds0 = xr.open_dataset(self.fine_files[0], engine="netcdf4")
        spatial_dims = [d for d in ds0.dims if d not in ("time",)]
        if len(spatial_dims) < 2:
            raise ValueError("Impossible de déterminer les dimensions spatiales.")
        ny = ds0.dims[spatial_dims[-2]]
        nx = ds0.dims[spatial_dims[-1]]

        if self.patch_size is None:
            for fi in range(len(self.fine_files)):
                self._patches.append((fi, 0, 0))
            return

        ps = self.patch_size
        st = self.stride
        for fi in range(len(self.fine_files)):
            for i0 in range(0, ny - ps + 1, st):
                for j0 in range(0, nx - ps + 1, st):
                    self._patches.append((fi, i0, j0))

    def _to_tensor(
        self,
        ds: xr.Dataset,
        vars: list[str],
        i0: int, i1: int | None,
        j0: int, j1: int | None,
        normalize: bool = True,
    ) -> torch.Tensor:
        channels = []
        for v in vars:
            if v not in ds:
                continue
            data = ds[v].values
            # Sélection spatiale (2 dernières dims)
            if data.ndim == 2:
                patch = data[i0:i1, j0:j1]
            elif data.ndim >= 3:
                patch = data[..., i0:i1, j0:j1]
                if patch.ndim > 2:
                    patch = patch.reshape(-1, patch.shape[-2], patch.shape[-1])[0]
            else:
                continue
            patch = np.where(np.isfinite(patch), patch, 0.0).astype(np.float32)
            if normalize and v in self.stats:
                mu, sigma = self.stats[v]
                patch = (patch - mu) / sigma
            channels.append(patch)
        return torch.from_numpy(np.stack(channels, axis=0))

    def _get_dem_patch(self, i0: int, i1: int | None, j0: int, j1: int | None) -> torch.Tensor:
        if self._dem_tensor is None:
            dem_ds = xr.open_dataset(self.dem_file, engine="netcdf4")
            channels = []
            for v in DEFAULT_DEM_VARS:
                if v not in dem_ds:
                    continue
                data = dem_ds[v].values.astype(np.float32)
                data = np.where(np.isfinite(data), data, 0.0)
                if v in self.stats:
                    mu, sigma = self.stats[v]
                    data = (data - mu) / sigma
                channels.append(data)
            self._dem_tensor = torch.from_numpy(np.stack(channels, axis=0))
        return self._dem_tensor[:, i0:i1, j0:j1]

    def _save_stats(self):
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_file, "w") as f:
            json.dump({k: list(v) for k, v in self.stats.items()}, f, indent=2)

    def _load_stats(self):
        with open(self.stats_file) as f:
            raw = json.load(f)
        self.stats = {k: tuple(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Fonction utilitaire : prépare un batch en inférence (sans target)
# ---------------------------------------------------------------------------

def prepare_inference_batch(
    coarse_ds: xr.Dataset,
    dem_ds: xr.Dataset,
    met_vars: list[str],
    stats: dict[str, tuple[float, float]],
    time_idx: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prépare les tenseurs d'entrée pour l'inférence (pas de target).

    Returns
    -------
    (x_coarse, dem_features) — chacun shape (1, C, H, W)
    """
    coarse_step = coarse_ds.isel(time=time_idx) if "time" in coarse_ds.dims else coarse_ds

    def _field_to_np(ds, var, mu, sigma):
        data = ds[var].values.astype(np.float32)
        data = np.where(np.isfinite(data), data, 0.0)
        return (data - mu) / sigma

    met_channels = []
    for v in met_vars:
        if v in coarse_step:
            mu, sigma = stats.get(v, (0.0, 1.0))
            met_channels.append(_field_to_np(coarse_step, v, mu, sigma))

    dem_channels = []
    for v in DEFAULT_DEM_VARS:
        if v in dem_ds:
            mu, sigma = stats.get(v, (0.0, 1.0))
            dem_channels.append(_field_to_np(dem_ds, v, mu, sigma))

    x = torch.from_numpy(np.stack(met_channels, axis=0)).unsqueeze(0).to(device)
    dem = torch.from_numpy(np.stack(dem_channels, axis=0)).unsqueeze(0).to(device)
    return x, dem
