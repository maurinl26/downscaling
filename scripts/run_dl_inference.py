#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "xarray>=2023.1",
#   "netCDF4>=1.6",
#   "pyyaml>=6.0",
#   "torch>=2.0",
#   "downscaling",
# ]
# ///
"""
Inférence DL : applique le modèle entraîné sur ERA5/CERRA → champs 1 km.

Exemple
-------
    python scripts/run_dl_inference.py \
        --config     config/drome_ardeche.yml \
        --checkpoint checkpoints/drome_ardeche/best_model.pt \
        --era5-sl    data/era5/era5_sl_20210427.nc \
        --dem-attrs  data/dem/dem_attributes.nc \
        --stats      checkpoints/drome_ardeche/normalization_stats.json \
        --out        output/dl_downscaled_20210427.nc \
        --compute-indices
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import xarray as xr

from deep_learning.inference import DLInferencePipeline
from shared.indices import compute_all_indices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/drome_ardeche.yml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--era5-sl", required=True)
    p.add_argument("--dem-attrs", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
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

    pipeline = DLInferencePipeline(
        checkpoint_path=args.checkpoint,
        config=cfg,
        stats_path=args.stats,
        device=args.device,
    )

    coarse_ds = xr.open_dataset(args.era5_sl, engine="netcdf4")
    dem_ds = xr.open_dataset(args.dem_attrs, engine="netcdf4")

    ds_out = pipeline.run(coarse_ds, dem_ds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=encoding)
    log.info(f"Champs haute résolution → {out_path}")

    if args.compute_indices:
        idx_cfg = cfg.get("indices", {})
        ds_idx = compute_all_indices(
            ds_out,
            unit_tp=idx_cfg.get("unit_tp", "m"),
            freq=idx_cfg.get("aggregation_freq", "YS"),
        )
        idx_path = out_path.with_name(out_path.stem + "_indices.nc")
        ds_idx.to_netcdf(idx_path)
        log.info(f"Indices paramétriques → {idx_path}")


if __name__ == "__main__":
    main()
