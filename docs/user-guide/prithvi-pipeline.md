# Prithvi WxC pipeline

> 📝 **Stub** — full user-guide section to be written.

The Prithvi pipeline uses the NASA/IBM **Prithvi WxC** foundation model as a
frozen backbone, with adapter layers conditioned on the digital elevation
model (DEM).

Status: experimental. The backbone is loaded from `prithviwxc` (and
optionally fine-tuned via `terratorch[wxc]`); see `pyproject.toml` for the
`prithvi` extra and the [`scripts/`](https://github.com/maurinl26/downscaling/tree/main/scripts)
folder for entry points.
