"""Entry point Hydra — descente d'échelle statistique ERA5/CERRA → 1 km.

Remplace l'``argparse`` + ``yaml.safe_load`` du pipeline statistique par une
composition Hydra. La région et le compute se sélectionnent par groupe :

    downscaling-run                                   # drome_ardeche / local
    downscaling-run cluster=cloud
    downscaling-run experiment=drome_ardeche run.compute_indices=true
    downscaling-run statistical.quantile_mapping.enabled=false

Le ``config_path`` est **absolu** (``downscaling.paths.CONFIG_DIR``) : sinon
Hydra, lancé depuis l'entry point console, chercherait un module Python
``configs`` inexistant.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import xarray as xr
from omegaconf import DictConfig, ListConfig, OmegaConf

from downscaling.paths import CONFIG_DIR
from downscaling.shared.indices import compute_all_indices
from downscaling.statistical.pipeline import StatisticalDownscalingPipeline

log = logging.getLogger(__name__)

# Dimensions spatiales connues — tout le reste est candidat à l'axe temporel.
_SPATIAL_DIMS = {"x", "y", "lat", "lon", "latitude", "longitude"}


def _source_list(era5_sl) -> list[str]:
    """Normalise ``data.era5_sl`` en liste de chemins.

    Accepte un fichier unique (``data.era5_sl=run.nc``) ou une liste de
    mensuels à concaténer (``data.era5_sl=[avril.nc,mai.nc]``), ce dernier cas
    servant les saisons multi-fichiers de ``run_campaign``.
    """
    if isinstance(era5_sl, (list, ListConfig)):
        return [str(p) for p in era5_sl]
    return [str(era5_sl)]


def _time_dim(ds: xr.Dataset) -> str:
    """Repère la dimension temporelle pour concaténer des sorties mensuelles."""
    for name in ("time", "valid_time", "t"):
        if name in ds.dims:
            return name
    non_spatial = [d for d in ds.dims if d not in _SPATIAL_DIMS]
    if not non_spatial:
        raise ValueError(
            f"Aucune dimension temporelle pour la concaténation : dims={tuple(ds.dims)}"
        )
    return non_spatial[0]


def run_statistical(cfg: DictConfig) -> Path:
    """Exécute la descente d'échelle statistique à partir d'une config composée.

    Retourne le chemin du NetCDF haute résolution écrit.
    """
    stat = cfg.statistical
    gamma = np.asarray(OmegaConf.to_container(stat.lapse_rate.monthly_gamma), dtype=float)

    pipeline = StatisticalDownscalingPipeline(
        dem_path=cfg.data.dem_raw,
        obs_ref_path=cfg.data.get("obs_ref"),
        lapse_rate=gamma,
        use_qdm=stat.quantile_mapping.enabled,
        n_quantiles=stat.quantile_mapping.n_quantiles,
    )

    # Calibration QDM si modèle + observations de référence disponibles.
    obs_ref = cfg.data.get("obs_ref")
    mod_ref = cfg.data.get("mod_ref")
    if obs_ref and mod_ref:
        log.info("Calibration QDM sur la période de référence…")
        pipeline.calibrate(
            xr.open_dataset(mod_ref, engine="netcdf4"),
            xr.open_dataset(obs_ref, engine="netcdf4"),
        )

    variables = list(stat.variables)
    sources = _source_list(cfg.data.era5_sl)
    if len(sources) == 1:
        log.info("Descente d'échelle de %s…", sources[0])
        ds_out = pipeline.run(source=sources[0], variables=variables)
    else:
        log.info("Descente d'échelle de %d fichiers (concaténation temporelle)…", len(sources))
        parts = [pipeline.run(source=s, variables=variables) for s in sources]
        ds_out = xr.concat(parts, dim=_time_dim(parts[0]))

    out_template = stat.output.file
    out_path = Path(cfg.run.out or out_template.format(date=cfg.run.date))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp = OmegaConf.to_container(stat.output.compression)
    ds_out.to_netcdf(out_path, encoding={v: comp for v in ds_out.data_vars})
    log.info("Champs haute résolution → %s", out_path)

    if cfg.run.compute_indices:
        log.info("Calcul des indices d'assurance paramétrique…")
        ds_idx = compute_all_indices(
            ds_out, unit_tp=cfg.indices.unit_tp, freq=cfg.indices.aggregation_freq
        )
        idx_path = out_path.with_name(out_path.stem + "_indices.nc")
        ds_idx.to_netcdf(idx_path)
        log.info("Indices → %s", idx_path)

    return out_path


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.DEBUG if cfg.run.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_statistical(cfg)


if __name__ == "__main__":
    main()
