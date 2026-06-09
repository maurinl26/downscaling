"""Entry point Hydra — calibration sparse du U-Net sur capteurs (chemin B, étage C).

Calibre un U-Net FiLM déjà entraîné pour qu'il reproduise les Tmin des stations
in situ (Sencrop / Netatmo), avec correction d'altitude (MNT). Chaîne déployable
``CERRA → U-Net 1 km → capteurs`` (cf. ``docs/architecture.md``).

    run-calibration                                   # config par défaut
    run-calibration cluster=cloud calibration.epochs=50
    run-calibration calibration.reduce=mean calibration.elevation_aware=false

Données attendues : CERRA rééchantillonné sur la grille fine
(``cerra_<date>.nc``), MNT (``data.dem_attrs``), exports Sencrop
(``sencrop_<date>.csv``), checkpoint U-Net entraîné, stats de normalisation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import xarray as xr
from omegaconf import DictConfig, OmegaConf

try:
    import torch
except ImportError as e:  # pragma: no cover - dépend de l'extra dl
    raise ImportError("PyTorch + Lightning requis : pip install 'downscaling[dl]'") from e

from downscaling.deep_learning.cerra_provider import CERRACoarseProvider
from downscaling.deep_learning.model import build_model
from downscaling.deep_learning.sparse_calibration import (
    UNetSparseCalibrationModule,
    UNetSparseDataModule,
    UNetStationDataset,
)
from downscaling.deep_learning.train import build_trainer
from downscaling.paths import CONFIG_DIR

log = logging.getLogger(__name__)


def grids_from_dem(dem: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrait ``(lat_grid 1D, lon_grid 1D, elevation_grid 2D)`` du MNT.

    Tolérant aux noms de coordonnées (``lat``/``latitude``) et aux grilles 1D ou
    2D (curvilignes → on prend une ligne/colonne représentative).
    """
    def _axis(names, axis):
        for n in names:
            if n in dem.coords or n in dem.variables:
                arr = np.asarray(dem[n].values)
                return arr if arr.ndim == 1 else (arr[:, 0] if axis == 0 else arr[0, :])
        return None

    elevation = np.asarray(dem["elevation"].values)
    ny, nx = elevation.shape[-2:]
    lat_grid = _axis(("lat", "latitude", "y"), axis=0)
    lon_grid = _axis(("lon", "longitude", "x"), axis=1)
    if lat_grid is None:
        lat_grid = np.arange(ny)
    if lon_grid is None:
        lon_grid = np.arange(nx)
    return lat_grid, lon_grid, elevation


def _load_unet_weights(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Charge les poids U-Net depuis un checkpoint Lightning (.ckpt) ou torch (.pt)."""
    state = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in state:  # checkpoint Lightning : préfixe `model.`
        sd = {k[len("model."):]: v for k, v in state["state_dict"].items()
              if k.startswith("model.")}
    elif "model_state_dict" in state:  # ancien format manuel
        sd = state["model_state_dict"]
    else:
        sd = state
    model.load_state_dict(sd)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.DEBUG if cfg.run.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Groupe Hydra `dl` (alias historique `deep_learning`).
    dl = cfg.dl if "dl" in cfg else cfg.deep_learning
    cal = cfg.calibration

    # U-Net + poids entraînés à calibrer.
    model = build_model(
        architecture=dl.get("architecture", "unet"),
        met_in_ch=len(dl.met_vars),
        dem_in_ch=dl.get("dem_in_ch", 4),
        base_ch=dl.get("base_ch", 64),
        n_levels=dl.get("n_levels", 4),
        use_film=dl.get("use_film", True),
    )
    log.info("Chargement du U-Net entraîné : %s", cal.checkpoint)
    _load_unet_weights(model, cal.checkpoint)

    # Entrées CERRA par nuit + grilles depuis le MNT.
    provider = CERRACoarseProvider(
        cal.cerra_fine_dir, cfg.data.dem_attrs,
        met_vars=list(dl.met_vars), stats_file=cal.stats_file,
        reduce=cal.reduce, hourly=cal.hourly,
        file_template=cal.get("file_template_cerra", "cerra_{date}.nc"),
        var_map=OmegaConf.to_container(cal.get("var_map", {})) or {},
        regrid=cal.get("regrid", False),
        regrid_method=cal.get("regrid_method", "linear"),
    )

    # Mode diagnostic : valide le format des données réelles sans entraîner.
    if cal.get("inspect", False):
        dates = provider.dates()
        if not dates:
            log.error("Aucun fichier CERRA dans %s", cal.cerra_fine_dir)
            return
        info = provider.inspect(dates[0])
        log.info("Inspection CERRA (%s) :", dates[0])
        for k, v in info.items():
            log.info("  %-18s %s", k, v)
        return
    lat_grid, lon_grid, elevation_grid = grids_from_dem(
        xr.open_dataset(cfg.data.dem_attrs, engine="netcdf4")
    )

    dataset = UNetStationDataset(
        provider.dates(), provider, cal.sencrop_dir, lat_grid, lon_grid,
        file_template=cal.file_template, elevation_grid=elevation_grid,
        min_stations=cal.min_stations, lapse_rate=cal.lapse_rate,
    )
    log.info("Nuits de calibration disponibles : %d", len(dataset))

    # Dénormalisation de la sortie U-Net (espace normalisé → °C) avec les stats t2m
    # d'entraînement, sinon la loss sparse compare des z-scores à des °C (non physique).
    t2m_stats = provider.stats.get("t2m")
    denorm = tuple(t2m_stats) if t2m_stats is not None else None
    lit = UNetSparseCalibrationModule(
        model, target_channel=cal.target_channel, lr=cal.lr, max_epochs=cal.epochs,
        denorm=denorm, kelvin_to_celsius=(denorm is None),
        lapse_rate=cal.lapse_rate, elevation_aware=cal.elevation_aware,
        hourly=cal.hourly, reduce=cal.reduce,
    )
    datamodule = UNetSparseDataModule(dataset, num_workers=cfg.cluster.get("num_workers", 0))
    trainer = build_trainer(
        cfg.cluster, max_epochs=cal.epochs, patience=cal.patience,
        checkpoint_dir=Path(cal.out).parent,
    )
    trainer.fit(lit, datamodule=datamodule)

    out = Path(cal.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, out)
    log.info("U-Net calibré → %s", out)


if __name__ == "__main__":
    main()
