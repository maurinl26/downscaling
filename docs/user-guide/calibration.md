# Sparse in-situ calibration

> 📝 **Stub** — full user-guide section to be written.

`karpos-downscaling` calibrates its downscaled fields against sparse in-situ
sensor networks, with current support for the **Sencrop** network in France
(approximately 4 000 agricultural weather stations).

Key implementation points (to be detailed):

- **Bulk Sencrop dataset loader** (Spark-partitioned CSV → xarray)
- **QC filtering** (`temperature_source == 'station'` etc.)
- **Out-of-sample bias adjustment** per station (median bias over training
  years, applied as offset at inference)
- **Fallback** to global bias for stations not seen in training
- **RBF residual propagation** (Lot B) to spread sparse corrections across
  the full grid
