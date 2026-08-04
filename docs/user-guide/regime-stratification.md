# Synoptic regime stratification (EPIC C5)

> 📝 **Stub** — full user-guide section to be written.

To support interpretable performance reporting, `karpos-downscaling` provides
a rule-based classification of frost nights into five synoptic regimes,
computed from ERA5 synoptic-scale fields (wind, total cloud cover, MSLP,
dewpoint depression).

| Regime | Wind (m/s) | TCC | MSLP | Interpretation |
|---|---|---|---|---|
| **R1 Radiative** | ≤ 3 | ≤ 0.50 | — | Radiative frost (Baronnies typical) |
| **R2 Advective** | > 3 | ≤ 0.50 | — | Forced mixing, advective frost possible |
| **R3 Cloudy windy** | > 3 | > 0.50 | — | Disturbed, frost rare |
| **R4a Cold pool** | ≤ 3 | > 0.50 | > 1020 hPa | Anticyclonic inversion, valley cold pool |
| **R4b Cloudy calm mild** | ≤ 3 | > 0.50 | ≤ 1020 hPa | Mild cloudy nights, frost limited |

The classification is implemented in
[`scripts/flag_regimes.py`](https://github.com/maurinl26/karpos-downscaling/blob/main/scripts/flag_regimes.py)
and consumed by `analyze_karpos_slr.py` via the `--regimes-csv`
option, which produces stratified POD/FAR/CSI/RMSE/bias metrics per regime.

This stratification is described in detail in the methodology section and in
the forthcoming GMD paper draft.
