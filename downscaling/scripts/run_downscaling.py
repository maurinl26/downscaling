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
from omegaconf import DictConfig, OmegaConf

from downscaling.paths import CONFIG_DIR
from downscaling.shared.indices import compute_all_indices
from downscaling.statistical.pipeline import StatisticalDownscalingPipeline

log = logging.getLogger(__name__)


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
    log.info("Descente d'échelle de %s…", cfg.data.era5_sl)
    ds_out = pipeline.run(source=cfg.data.era5_sl, variables=variables)

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
