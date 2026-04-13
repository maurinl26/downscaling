#!/usr/bin/env python
"""
Script to apply statistical downscaling over ERA5-Land data for parametric insurance.
Because the high-res DEM file is missing from the repository, this script allows 
passing a `--dem` argument or generates a dummy DEM if none is provided to avoid crashing.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import xarray as xr

# Remove path insertion to rely on standard module execution
import yaml
from downscaling.statistical.pipeline import StatisticalDownscalingPipeline
from downscaling.shared.indices import compute_all_indices

def create_dummy_dem(domain_cfg, out_path="dummy_dem.nc"):
    """
    Creates a dummy DEM (elevation=0) based on the domain configuration to allow 
    the pipeline to run even if the actual DEM is missing.
    Warning: This will not perform proper lapse-rate correction!
    """
    lat_min = domain_cfg.get("lat_min", 44.0)
    lat_max = domain_cfg.get("lat_max", 45.5)
    lon_min = domain_cfg.get("lon_min", 4.0)
    lon_max = domain_cfg.get("lon_max", 5.5)
    ny = domain_cfg.get("ny", 167)
    nx = domain_cfg.get("nx", 118)
    
    lats = np.linspace(lat_max, lat_min, ny)
    lons = np.linspace(lon_min, lon_max, nx)
    
    # Dummy elevation of 0 meters for all points
    data = np.zeros((ny, nx), dtype=np.float32)
    
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "lat": (["y", "x"], np.repeat(lats[:, None], nx, axis=1)),
            "lon": (["y", "x"], np.repeat(lons[None, :], ny, axis=0))
        },
        name="elevation",
        attrs={"units": "m", "long_name": "elevation", "source": "dummy"}
    )
    da.to_netcdf(out_path)
    return out_path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/drome_ardeche.yml")
    p.add_argument("--era5land-dir", default="../data/raw/era5land/2m_temperature/")
    p.add_argument("--dem", default=None, help="Path to DEM file. If None, uses dummy_dem.nc")
    p.add_argument("--out-dir", default="../output/era5land_downscaled/")
    p.add_argument("--compute-indices", action="store_true", default=True)
    p.add_argument("-v", "--verbose", action="store_true", default=True)
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
    
    gamma = np.array(lapse_cfg.get("monthly_gamma", [-6.5e-3] * 12))
    
    dem_path = args.dem
    if not dem_path or not Path(dem_path).exists():
        log.warning(f"DEM file not found or not provided. Generating a dummy DEM at dummy_dem.nc. LAPSE RATE CORRECTION WILL BE ZEROED.")
        dem_path = create_dummy_dem(cfg.get("domain", {}), "dummy_dem.nc")

    pipeline = StatisticalDownscalingPipeline(
        dem_path=dem_path,
        obs_ref_path=None, 
        lapse_rate=gamma,
        use_qdm=False, # QDM requires reference data, typically we don't have it for ERA5-Land
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process all era5land files
    era5land_dir = Path(args.era5land_dir)
    files = sorted(era5land_dir.glob("*.nc"))
    
    if not files:
        log.error(f"No ERA5-Land files found in {era5land_dir}")
        return

    for file_path in files:
        log.info(f"Processing {file_path.name}...")
        # Use explicit dask chunks on the time dimension to avoid OOM on massive DEM regridding
        ds_source = xr.open_dataset(file_path, engine="netcdf4", chunks={"valid_time": 24})
        
        # Rename dimensions to match pipeline expectations
        rename_dict = {}
        if "latitude" in ds_source.dims:
            rename_dict["latitude"] = "lat"
        if "longitude" in ds_source.dims:
            rename_dict["longitude"] = "lon"
        if "valid_time" in ds_source.dims:
            rename_dict["valid_time"] = "time"
            
        if rename_dict:
            ds_source = ds_source.rename(rename_dict)
        
        ds_out = pipeline.run(
            source=ds_source,
            variables=["t2m"], # Only temperature for frost
        )
        
        out_path = out_dir / f"stat_downscaled_era5land_{file_path.stem}.nc"
        
        encoding = {"t2m": {"zlib": True, "complevel": 4}}
        ds_out.to_netcdf(out_path, encoding=encoding)
        log.info(f"High-res output saved to {out_path}")

        if args.compute_indices:
            idx_cfg = cfg.get("indices", {})
            ds_idx = compute_all_indices(
                ds_out,
                unit_tp=idx_cfg.get("unit_tp", "m"),
                freq="MS", # Monthly computation since file is monthly
            )
            idx_path = out_dir / f"frost_indices_era5land_{file_path.stem}.nc"
            # We are interested in frost
            ds_idx.to_netcdf(idx_path)
            log.info(f"Indices -> {idx_path}")
            
        # Break after processing the first file for rapid testing
        break

if __name__ == "__main__":
    main()
