"""``coarse_provider`` concret pour CERRA — entrées U-Net par nuit (chemin B).

Fournit ``(x_met, x_dem)`` normalisés pour une date, à brancher sur
:class:`downscaling.deep_learning.sparse_calibration.UNetStationDataset`.

Robuste aux **vraies données CERRA** :
  - harmonisation des coordonnées (``latitude``/``longitude``/``valid_time``) ;
  - mapping de noms de variables ``var_map`` (CERRA → canonique) si besoin ;
  - validation **bruyante** des variables attendues (liste les disponibles) ;
  - rééchantillonnage optionnel sur la grille du MNT (``regrid=True``) si les
    fichiers ne sont pas déjà sur la grille fine ;
  - :meth:`inspect` pour diagnostiquer un fichier avant de lancer la calibration.

Normalisation = mêmes statistiques (µ, σ) que l'entraînement
(``normalization_stats.json``). Pour la calibration **Tmin nocturne**, la nuit
(20h → 08h) est descendue heure par heure puis réduite (côté module, voir
``UNetSparseCalibrationModule``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import xarray as xr

from .dataset import DEFAULT_MET_VARS, prepare_inference_batch

# Exemple de mapping noms CERRA → canonique (à ajuster au vrai export ; vide par
# défaut → on s'appuie sur la validation pour révéler les noms réels).
CERRA_VAR_MAP_EXAMPLE = {
    "t2m": "2t",
    "u10": "10u",
    "v10": "10v",
    "sp": "sp",
    "tp": "tp",
}


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
        reduce: str = "min",
        hourly: bool = True,
        var_map: dict | None = None,  # {canonique: nom_cerra} si les noms diffèrent
        regrid: bool = False,  # True si CERRA pas déjà sur la grille fine (MNT)
        regrid_method: str = "linear",
    ):
        self.cerra_dir = Path(cerra_dir)
        self.dem_file = Path(dem_file)
        self.met_vars = list(met_vars)
        self.file_template = file_template
        self.hourly = hourly
        self.night_start = night_start
        self.night_end = night_end
        self.reduce = reduce
        self.var_map = dict(var_map or {})
        self.regrid = regrid
        self.regrid_method = regrid_method

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
        return [
            p.name[len(prefix) : len(p.name) - len(suffix)]
            for p in sorted(self.cerra_dir.glob(f"{prefix}*{suffix}"))
        ]

    def _dem(self) -> xr.Dataset:
        if self._dem_ds is None:
            self._dem_ds = xr.open_dataset(self.dem_file, engine="netcdf4")
        return self._dem_ds

    # ------------------------------------------------------------------
    # Harmonisation / validation / regrid
    # ------------------------------------------------------------------
    def _harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """Renomme coordonnées (lat/lon/time) et variables (``var_map``)."""
        renames = {}
        for src, dst in (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon")):
            if src in ds.dims or src in ds.coords:
                renames[src] = dst
        if renames:
            ds = ds.rename(renames)
        vmap = {cerra: canon for canon, cerra in self.var_map.items() if cerra in ds}
        return ds.rename(vmap) if vmap else ds

    def _validate(self, ds: xr.Dataset, date: str) -> None:
        missing = [v for v in self.met_vars if v not in ds]
        if missing:
            raise KeyError(
                f"CERRA {date} : variables attendues manquantes {missing}. "
                f"Disponibles : {list(ds.data_vars)}. "
                f"Renseigner `var_map={{canonique: nom_cerra}}` (cf. CERRA_VAR_MAP_EXAMPLE)."
            )

    def _maybe_regrid(self, ds: xr.Dataset) -> xr.Dataset:
        if not self.regrid:
            return ds
        from downscaling.shared.loaders import regrid_to_dem

        dem_elev = self._dem()["elevation"]
        return xr.Dataset(
            {v: regrid_to_dem(ds[v], dem_elev, method=self.regrid_method) for v in self.met_vars}
        )

    def _open_night(self, date: str) -> xr.Dataset:
        """Ouvre le fichier CERRA, harmonise, restreint à la nuit (20h → 08h)."""
        ds = self._harmonize(xr.open_dataset(self.path(date), engine="netcdf4"))
        if "time" in ds.dims:
            start = pd.Timestamp(date) + pd.Timedelta(self.night_start)
            end = pd.Timestamp(date) + pd.Timedelta("1D") + pd.Timedelta(self.night_end)
            ds = ds.sel(time=slice(start, end))
        return ds

    # ------------------------------------------------------------------
    def inspect(self, date: str) -> dict:
        """Diagnostic d'un fichier CERRA (pour valider le format des vraies données)."""
        raw = xr.open_dataset(self.path(date), engine="netcdf4")
        ds = self._harmonize(raw)
        present = [v for v in self.met_vars if v in ds]
        dem_grid = tuple(self._dem()["elevation"].shape[-2:])
        field_grid = tuple(ds[present[0]].shape[-2:]) if present else None
        return {
            "path": str(self.path(date)),
            "data_vars": list(raw.data_vars),
            "met_present": present,
            "met_missing": [v for v in self.met_vars if v not in ds],
            "coords": list(raw.coords),
            "dims": dict(raw.sizes),
            "has_time": "time" in ds.dims,
            "n_time": int(ds.sizes.get("time", 0)),
            "field_grid": field_grid,  # (H, W) d'un champ météo
            "dem_grid": dem_grid,  # (H, W) du MNT
            # Grilles alignées → regrid inutile ; sinon passer regrid=True.
            "grid_matches_dem": field_grid == dem_grid if field_grid else None,
        }

    # ------------------------------------------------------------------
    def __call__(self, date: str):
        """Retourne ``(x_met, x_dem)`` pour la nuit.

        - ``hourly=True`` : ``x_met`` de forme ``(T, C, H, W)`` (un champ par heure).
        - ``hourly=False`` : ``x_met`` de forme ``(C, H, W)`` (réduction coarse).
        """
        ds = self._open_night(date)
        self._validate(ds, date)
        ds = self._maybe_regrid(ds)

        if self.hourly and "time" in ds.dims:
            mets, x_dem = [], None
            for t in range(ds.sizes["time"]):
                x, dem = prepare_inference_batch(
                    ds, self._dem(), self.met_vars, self.stats, time_idx=t, device="cpu"
                )
                mets.append(x.squeeze(0))  # (C, H, W)
                x_dem = dem.squeeze(0)
            return torch.stack(mets, dim=0), x_dem  # (T, C, H, W), (C_dem, H, W)

        if "time" in ds.dims:
            ds = getattr(ds, self.reduce)("time")
        x_met, x_dem = prepare_inference_batch(
            ds, self._dem(), self.met_vars, self.stats, device="cpu"
        )
        return x_met.squeeze(0), x_dem.squeeze(0)
