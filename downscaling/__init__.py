"""karpos-downscaling — atmospheric reanalysis downscaling with sparse in-situ calibration.

This package provides three families of methods for downscaling coarse
atmospheric reanalyses (ERA5, CERRA) to kilometer-scale resolution,
calibrated against sparse in-situ sensor networks (Sencrop) and stratified
by synoptic regime for interpretable performance reporting:

- ``downscaling.shared`` — common loaders (ERA5, CERRA, DEM), indices, utilities
- ``downscaling.deep_learning`` — U-Net with FiLM conditioning on DEM and
  synoptic regime, sparse Sencrop calibration loss, PyTorch Lightning training
- ``downscaling.scripts`` — command-line entry points for the full pipeline
  (download, recalibrate, calibrate QDM, classify regimes, analyze metrics)

The package targets applications in parametric agricultural insurance
(frost on fruit orchards and hillside vineyards), but the methodology is
generic and transferable to any region where a sparse in-situ network
exists.

The license is Apache 2.0; the scientific methodology is documented under
``docs/methodology/`` and in the forthcoming GMD paper.
"""
