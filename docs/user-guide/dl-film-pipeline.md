# Deep learning FiLM pipeline (Lot C)

> 📝 **Stub** — full user-guide section to be written.

The deep-learning pipeline (`recalibrate_dl_film`) trains a U-Net conditioned
on the digital elevation model and the synoptic regime via FiLM layers
(Feature-wise Linear Modulation, Perez et al. 2017).

Key features:

- **Dual-FiLM conditioning** on DEM (terrain) and synoptic regime
  (R1/R2/R3/R4a/R4b)
- **Sparse Sencrop loss** for in-situ calibration
- PyTorch Lightning training with W&B logging
- Configurable U-Net capacity, early stopping, learning rate schedules

References to come: Park et al. 2022 (Weather4Cast), Chapman et al. 2024
(RUFCO). See [`scripts/recalibrate_dl_film.py`](https://github.com/maurinl26/karpos-downscaling/blob/main/scripts/recalibrate_dl_film.py).
