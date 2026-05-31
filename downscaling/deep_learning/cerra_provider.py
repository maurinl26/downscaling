"""``coarse_provider`` concret pour CERRA — entrées U-Net par nuit (chemin B).

Fournit ``(x_met, x_dem)`` normalisés pour une date, à brancher sur
:class:`downscaling.deep_learning.sparse_calibration.UNetStationDataset`.

Hypothèses (cohérence avec l'entraînement, cf. ``DownscalingDataset``) :
  - Les fichiers CERRA sont **déjà rééchantillonnés sur la grille fine** (même
    grille que le MNT) — c'est l'étape de préparation des données amont.
  - Normalisation = mêmes statistiques (µ, σ) que l'entraînement
    (``normalization_stats.json``), passées via ``stats``/``stats_file``.

Pour la calibration **Tmin nocturne**, le champ CERRA de la nuit (20h → 08h) est
réduit par maille (``reduce="min"`` par défaut) avant descente d'échelle, en
cohérence avec la cible Tmin des stations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import xarray as xr

from .dataset import DEFAULT_MET_VARS, prepare_inference_batch


class CERRACoarseProvider:
    """Construit ``(x_met, x_dem)`` pour une nuit à partir de CERRA + MNT."""

    def __init__(
        self,
        cerra_dir: str | Path,
        dem_file: str | Path,
        met_vars: list[str] = DEFAULT_MET_VARS,
        *,
        stats: dict | None = None,
        stats_file: str | Path | None = None,
        file_template: str = "cerra_{date}.nc",
        night_start: str = "20h",
        night_end: str = "8h",
        reduce: str = "min",   # 'min' (Tmin) | 'mean' | 'max'
        hourly: bool = True,   # True = pile horaire (descente puis min) ; False = réduction coarse avant U-Net
    ):
        self.cerra_dir = Path(cerra_dir)
        self.dem_file = Path(dem_file)
        self.met_vars = met_vars
        self.file_template = file_template
        self.hourly = hourly
        self.night_start = night_start
        self.night_end = night_end
        self.reduce = reduce

        if stats is None and stats_file is not None:
            with open(stats_file) as f:
                stats = {k: tuple(v) for k, v in json.load(f).items()}
        self.stats = stats or {}
        self._dem_ds: xr.Dataset | None = None

    # ------------------------------------------------------------------
    def path(self, date: str) -> Path:
        return self.cerra_dir / self.file_template.format(date=date)

    def dates(self) -> list[str]:
        """Dates disponibles, déduites des fichiers présents (``YYYY-MM-DD``)."""
        prefix, suffix = self.file_template.split("{date}")
        out = []
        for p in sorted(self.cerra_dir.glob(f"{prefix}*{suffix}")):
            out.append(p.name[len(prefix): len(p.name) - len(suffix)])
        return out

    # ------------------------------------------------------------------
    def _dem(self) -> xr.Dataset:
        if self._dem_ds is None:
            self._dem_ds = xr.open_dataset(self.dem_file, engine="netcdf4")
        return self._dem_ds

    def _open_night(self, date: str) -> xr.Dataset:
        """Ouvre le fichier CERRA et restreint à la fenêtre nocturne (20h → 08h)."""
        ds = xr.open_dataset(self.path(date), engine="netcdf4")
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        if "time" in ds.dims:
            start = pd.Timestamp(date) + pd.Timedelta(self.night_start)
            end = pd.Timestamp(date) + pd.Timedelta("1D") + pd.Timedelta(self.night_end)
            ds = ds.sel(time=slice(start, end))
        return ds

    def __call__(self, date: str):
        """Retourne ``(x_met, x_dem)`` pour la nuit.

        - ``hourly=True`` : ``x_met`` de forme ``(T, C, H, W)`` (un champ par heure) ;
          la réduction Tmin se fait **après** descente d'échelle (côté module).
        - ``hourly=False`` : ``x_met`` de forme ``(C, H, W)`` (réduction coarse par
          maille via ``reduce`` avant le U-Net).
        """
        ds = self._open_night(date)

        if self.hourly and "time" in ds.dims:
            mets, x_dem = [], None
            for t in range(ds.sizes["time"]):
                x, dem = prepare_inference_batch(
                    ds, self._dem(), self.met_vars, self.stats, time_idx=t, device="cpu"
                )
                mets.append(x.squeeze(0))   # (C, H, W)
                x_dem = dem.squeeze(0)       # (C_dem, H, W) — constant dans le temps
            return torch.stack(mets, dim=0), x_dem   # (T, C, H, W), (C_dem, H, W)

        # Réduction coarse par maille (ou pas de dim temporelle).
        if "time" in ds.dims:
            ds = getattr(ds, self.reduce)("time")
        x_met, x_dem = prepare_inference_batch(
            ds, self._dem(), self.met_vars, self.stats, device="cpu"
        )
        return x_met.squeeze(0), x_dem.squeeze(0)
