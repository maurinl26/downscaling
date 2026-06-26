# Quickstart

This page walks you through an end-to-end downscaling run with the
statistical pipeline on a single year and small geographical bounding box.
Once you have a working install (see [Installation](installation.md)) and
CDS credentials, the full sequence below takes ~30 minutes on a modern
laptop.

We use the **statistical Lot B pipeline** (lapse-rate + quantile delta
mapping + sparse Sencrop residual correction) — it has no GPU requirement
and is the fastest path to a first usable output.

## Prerequisites

- A working install with the `statistical` extra:

  ```bash
  uv sync --extra statistical
  ```

- CDS credentials in `~/.cdsapirc` (see [Installation](installation.md))
- Access to a Sencrop bulk export, or any equivalent in-situ network with
  the same schema (station id, timestamp, temperature, lat, lon, altitude)
- A working directory with ~5 GB free (CERRA NetCDF + Zarr output)

## Scenario

We will downscale the **CERRA 5.5 km** 2-meter temperature for the
**frost season 2024 (February–April)** over a small bounding box covering
the **Baronnies Provençales** (Drôme, France), using a SRTM DEM
auto-fetched at 30 m resolution and the Sencrop network of in-orchard
sensors for residual calibration.

Target output: a Zarr dataset at ~1 km resolution containing the calibrated
nightly minimum temperature (`Tmin`) for each frost-window night.

## Step 1 — Bootstrap the DEM

If you do not already have a DEM NetCDF for your area, the project includes
a bootstrap step that downloads SRTM 30 m via the `elevation` package:

```bash
mkdir -p /tmp/karpos-quickstart/dem

uv run eio clip \
  --bounds 4.95 44.20 5.65 44.55 \
  -o /tmp/karpos-quickstart/dem/srtm_baronnies.tif

# Convert GeoTIFF → NetCDF with the conventions expected by the pipeline
uv run python -c "
import rasterio, xarray as xr, numpy as np
with rasterio.open('/tmp/karpos-quickstart/dem/srtm_baronnies.tif') as src:
    a = src.read(1).astype(np.float32)
    h, w = a.shape
    lats = np.linspace(src.bounds.top, src.bounds.bottom, h, dtype=np.float64)
    lons = np.linspace(src.bounds.left, src.bounds.right, w, dtype=np.float64)
ds = xr.Dataset({'elevation': (('lat', 'lon'), a)},
                coords={'lat': lats, 'lon': lons})
ds.attrs['source'] = 'SRTM3 30m via elevation package'
ds.to_netcdf('/tmp/karpos-quickstart/dem/srtm_baronnies.nc')
print(f'wrote {ds.dims}')
"
```

For production work, replace SRTM by IGN BD ALTI 25 m if you have access.

## Step 2 — Download CERRA for the year and bounding box

CERRA download uses the CDS API. For Lot B we need:

- `cerra_atm_2024.nc` — 2-meter temperature (single levels)
- `cerra_land_2024.nc` — skin temperature (CERRA-Land), loaded for symmetry
- `cerra_orography.nc` — time-invariant orography (essential for lapse-rate)

Use the project download script:

```bash
export CDSAPI_URL=https://cds.climate.copernicus.eu/api
export CDSAPI_KEY=<your-cds-key>

uv run python -m downscaling.scripts.download_cerra_for_recalibration \
  --start 2024-02-01 \
  --end 2024-04-30 \
  --bbox-lat-min 44.2 --bbox-lat-max 44.55 \
  --bbox-lon-min 4.95 --bbox-lon-max 5.65 \
  --out-atm /tmp/karpos-quickstart/cerra-atm \
  --out-land /tmp/karpos-quickstart/cerra-land
```

Concat monthly files into yearly NetCDFs (the recalibration pipeline expects
yearly inputs):

```bash
uv run python -c "
import xarray as xr, glob
for prefix, outdir in [('cerra_atm','/tmp/karpos-quickstart/cerra-atm'),
                       ('cerra_land','/tmp/karpos-quickstart/cerra-land')]:
    files = sorted(glob.glob(f'{outdir}/{prefix}_2024_*.nc'))
    if not files: continue
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='valid_time',
                            parallel=False, decode_times=False,
                            mask_and_scale=False)
    ds.to_netcdf(f'{outdir}/{prefix}_2024.nc')
    print(f'wrote {outdir}/{prefix}_2024.nc from {len(files)} files')
"
```

## Step 3 — Run the statistical recalibration

This is the core step. It:

1. Loads CERRA atm + CERRA-Land yearly NetCDF
2. Computes nightly minimum temperature (resample 1D after shift -9h to
   align the night on the morning date)
3. Applies a **lapse-rate correction** to the fine DEM grid (~1 km)
4. Applies optional **QDM correction** if a pre-fitted joblib is provided
5. Applies a **sparse Sencrop RBF residual** correction
   (`obs_Sencrop - grid_at_station` propagated with σ = 7 km)
6. Writes a Zarr dataset at ~1 km

```bash
uv run python -m downscaling.scripts.recalibrate_statistical \
  --year 2024 \
  --cerra-atm  /tmp/karpos-quickstart/cerra-atm/cerra_atm_2024.nc \
  --cerra-land /tmp/karpos-quickstart/cerra-land/cerra_land_2024.nc \
  --cerra-orog /tmp/karpos-quickstart/cerra-atm/cerra_orography.nc \
  --dem        /tmp/karpos-quickstart/dem/srtm_baronnies.nc \
  --sencrop    /path/to/your/sencrop/bulk \
  --out        /tmp/karpos-quickstart/recalibrated \
  --wandb-disabled
```

Output: `/tmp/karpos-quickstart/recalibrated/2024.zarr` (~500 MB compressed).

## Step 4 — Inspect the output

```python
import xarray as xr

ds = xr.open_zarr("/tmp/karpos-quickstart/recalibrated/2024.zarr")
print(ds)
# <xarray.Dataset>
# Dimensions:      (time: 90, lat: 1620, lon: 2520)
# Coordinates:
#   * time         (time) datetime64[ns] 2024-02-01 ... 2024-04-30
#   * lat          (lat) float64 44.2 ... 44.55
#   * lon          (lon) float64 4.95 ... 5.65
# Data variables:
#     tmin_calibrated (time, lat, lon) float32 ...

# Quick look at one night
ds.tmin_calibrated.sel(time="2024-04-08").plot(cmap="RdBu_r")
```

## Step 5 — Compute POD / FAR / CSI metrics

To evaluate against in-situ observations:

```bash
uv run python -m downscaling.scripts.analyze_recalibrated_statistical \
  --root /tmp/karpos-quickstart/recalibrated \
  --sencrop /path/to/your/sencrop/bulk \
  --years 2024 \
  --threshold-c -2.2 \
  --wandb-disabled
```

This produces a `summary.json` describing the contingency table at the
flowering apricot threshold (–2.2 °C). The script writes the summary to
standard output and to W&B (unless `--wandb-disabled`). To evaluate
multiple thresholds, re-run with a different `--threshold-c` value
(0.0 for the reference, –5.0 for severe events).

### Reading Zarr from S3 directly

`--root` accepts `s3://` URIs so you can analyse remote outputs without
pulling them locally (useful when running from a RunPod pod or any
compute node sharing the bucket). Set the S3 endpoint via the standard
`AWS_ENDPOINT_URL` env var (Scaleway in the example below) and provide
credentials the usual way (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
or `~/.aws/credentials`):

```bash
AWS_ENDPOINT_URL=https://s3.fr-par.scw.cloud \
uv run python -m downscaling.scripts.analyze_recalibrated_statistical \
  --root s3://karpos-backtest-data/recalibrated/statistical \
  --sencrop s3://karpos-backtest-data/sencrop \
  --years 2022 2023 2024 \
  --threshold-c -2.2 \
  --wandb-disabled
```

The `<year>.posthoc.json` sidecars are written to the current working
directory by default when `--root` is remote (they cannot be written
back next to the source without write credentials); use `--out-dir` to
choose another local destination.

## Step 6 (optional) — Stratify by synoptic regime

If you want to interpret your scores per atmospheric regime (radiative vs
advective vs cloudy etc.), run the regime classifier first, then re-run
the analyzer with the `--regimes-csv` option:

```bash
# 1. Download ERA5 synoptic-scale fields for 2024
uv run python -m downscaling.scripts.download_era5_synoptic \
  --years 2024 \
  --bbox-lat 42 47 --bbox-lon 0 8 \
  --out /tmp/karpos-quickstart/era5

# 2. Classify nights into R1 / R2 / R3 / R4 (and R4a / R4b after split)
uv run python -m downscaling.scripts.flag_regimes \
  --era5-dir /tmp/karpos-quickstart/era5 \
  --years 2024 \
  --bbox-lat 44.2 44.55 \
  --bbox-lon 4.95 5.65 \
  --out /tmp/karpos-quickstart/regimes

# 3. Re-run the analyzer with regime stratification
uv run python -m downscaling.scripts.analyze_recalibrated_statistical \
  --root /tmp/karpos-quickstart/recalibrated \
  --sencrop /path/to/your/sencrop/bulk \
  --years 2024 \
  --regimes-csv /tmp/karpos-quickstart/regimes \
  --threshold-c -2.2 \
  --wandb-disabled
```

You will get a `contingency_by_regime` field in `summary.json` with
POD/FAR/CSI/RMSE per regime, as described in
[user-guide / regime stratification](../user-guide/regime-stratification.md).

## Where to go from here

- **Multi-year runs**: replace `--year 2024` by a loop over 2022–2026, or
  use the project-level orchestrator
  [`scripts/recalibration_pipeline.sh`](https://github.com/maurinl26/parametric_insurance/blob/main/scripts/recalibration_pipeline.sh)
  for a full pipeline with optional DL FiLM stage.
- **DL FiLM pipeline**: see
  [user-guide / dl-film-pipeline](../user-guide/dl-film-pipeline.md).
- **Methodology and validation**: see
  [methodology / lot-b-calibration-report](../methodology/lot-b-calibration-report.md).
- **Indices for parametric insurance**: read out the Zarr nightly minimum
  and combine with the BBCH-stage thresholds (Proebsting & Mills 1978).
