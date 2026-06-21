"""Command-line entry points for the karpos-downscaling pipeline.

Each script is invocable as ``python -m downscaling.scripts.<name>`` and
exposes ``--help`` for argument documentation. The main entry points are:

- ``recalibrate_statistical`` — Lot B: lapse-rate + QDM + sparse Sencrop residual
- ``recalibrate_dl_film`` — Lot C: U-Net FiLM trained on sparse Sencrop calibration loss
- ``calibrate_qdm`` — fit quantile delta mapping joblib for use by ``recalibrate_statistical``
- ``flag_regimes`` — classify frost-flo nights into synoptic regimes (R1/R2/R3/R4a/R4b)
- ``analyze_recalibrated_statistical`` — compute POD/FAR/CSI/RMSE/bias metrics,
  optionally stratified by regime via ``--regimes-csv``
- ``download_era5_synoptic`` — fetch ERA5 synoptic-scale variables for regime classification
- ``download_cerra_for_recalibration`` — fetch CERRA single levels and CERRA-Land

The orchestration of these scripts as a full multi-year pipeline lives in the
parametric_insurance repository under ``scripts/recalibration_pipeline.sh``.
"""
