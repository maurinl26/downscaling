# Quickstart

> 📝 **Stub** — to be expanded with a minimal end-to-end downscaling example.

A complete first-run walkthrough will be provided here, covering:

1. Preparing the CERRA reanalysis input for a small bounding box and time
   window (using `download_cerra_for_recalibration`)
2. Preparing a digital elevation model (DEM) at the target resolution
3. Running the statistical pipeline (`recalibrate_statistical`) to produce a
   Zarr output at 1 km
4. Inspecting the output with `xarray` and basic plotting

For the time being, please refer to the
[`scripts/recalibration_pipeline.sh`](https://github.com/maurinl26/parametric_insurance/blob/main/scripts/recalibration_pipeline.sh)
orchestrator and the `tests/` folder for working examples.
