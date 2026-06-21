# Statistical pipeline (Lot B)

> 📝 **Stub** — full user-guide section to be written.

The statistical pipeline (`recalibrate_statistical`) is the CPU-only path,
combining:

1. **Lapse-rate correction** from coarse reanalysis to fine DEM
2. **Quantile mapping (QDM)** with optional conditional regimes
3. **Sparse Sencrop residual correction** via gaussian RBF interpolation

See [`scripts/recalibrate_statistical.py`](https://github.com/maurinl26/downscaling/blob/main/scripts/recalibrate_statistical.py)
and the [Lot B calibration report](../methodology/lot-b-calibration-report.md)
for details until this page is expanded.
