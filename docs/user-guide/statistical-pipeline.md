# Statistical pipeline (KarposSLR)

> 📝 **Stub** — full user-guide section to be written.

The statistical pipeline (`recalibrate_karpos_slr`) is the CPU-only path,
combining:

1. **Lapse-rate correction** from coarse reanalysis to fine DEM
2. **Quantile mapping (QDM)** with optional conditional regimes
3. **Sparse Sencrop residual correction** via gaussian RBF interpolation

See [`scripts/recalibrate_karpos_slr.py`](https://github.com/maurinl26/karpos-downscaling/blob/main/scripts/recalibrate_karpos_slr.py)
and the [KarposSLR calibration report](../methodology/karpos-slr-calibration-report.md)
for details until this page is expanded.
