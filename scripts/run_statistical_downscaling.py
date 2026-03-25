#!/usr/bin/env python
"""
Script de descente d'échelle statistique ERA5/CERRA → 1 km.

Exemple
-------
    python scripts/run_statistical_downscaling.py \
        --config config/drome_ardeche.yml \
        --era5-sl data/era5/era5_sl_20210427.nc \
        --dem     data/dem/copdem_drome_100m.tif \
        --date    20210427 \
        --compute-indices
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import xarray as xr

from shared.loaders import ERA5Loader, DEMLoader
from statistical.pipeline import StatisticalDownscalingPipeline
from shared.indices import compute_all_indices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/drome_ardeche.yml")
    p.add_argument("--era5-sl", required=True)
    p.add_argument("--dem", required=True)
    p.add_argument("--obs-ref", default=None, help="Référence pour calibration QDM")
    p.add_argument("--mod-ref", default=None, help="Modèle historique pour calibration QDM")
    p.add_argument("--date", default="", help="Date YYYYMMDD (pour le nom de fichier de sortie)")
    p.add_argument("--out", default=None)
    p.add_argument("--compute-indices", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stat_cfg = cfg.get("statistical", {})
    lapse_cfg = stat_cfg.get("lapse_rate", {})
    qm_cfg = stat_cfg.get("quantile_mapping", {})

    # Gradient thermique mensuel depuis la config
    import numpy as np
    gamma = np.array(lapse_cfg.get("monthly_gamma", [-6.5e-3] * 12))

    pipeline = StatisticalDownscalingPipeline(
        dem_path=args.dem,
        obs_ref_path=args.obs_ref,
        lapse_rate=gamma,
        use_qdm=qm_cfg.get("enabled", True),
        n_quantiles=qm_cfg.get("n_quantiles", 100),
    )

    # Calibration QDM si données de référence fournies
    if args.obs_ref and args.mod_ref:
        log.info("Calibration QDM sur la période de référence…")
        obs_ref = xr.open_dataset(args.obs_ref, engine="netcdf4")
        mod_ref = xr.open_dataset(args.mod_ref, engine="netcdf4")
        pipeline.calibrate(mod_ref, obs_ref)

    # Descente d'échelle
    log.info(f"Descente d'échelle de {args.era5_sl}…")
    ds_out = pipeline.run(
        source=args.era5_sl,
        variables=stat_cfg.get("variables", ["t2m", "tp", "u10", "v10"]),
    )

    # Fichier de sortie
    out_template = stat_cfg.get("output", {}).get("file", "output/stat_downscaled_{date}.nc")
    out_path = Path(args.out or out_template.format(date=args.date))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoding = {
        v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars
    }
    ds_out.to_netcdf(out_path, encoding=encoding)
    log.info(f"Champs haute résolution → {out_path}")

    # Calcul des indices paramétriques
    if args.compute_indices:
        log.info("Calcul des indices d'assurance paramétrique…")
        idx_cfg = cfg.get("indices", {})
        ds_idx = compute_all_indices(
            ds_out,
            unit_tp=idx_cfg.get("unit_tp", "m"),
            freq=idx_cfg.get("aggregation_freq", "YS"),
        )
        idx_path = out_path.with_name(out_path.stem + "_indices.nc")
        ds_idx.to_netcdf(idx_path)
        log.info(f"Indices → {idx_path}")
        for v in ds_idx.data_vars:
            log.info(f"  {v}: mean={float(ds_idx[v].mean()):.3f}")


if __name__ == "__main__":
    main()
