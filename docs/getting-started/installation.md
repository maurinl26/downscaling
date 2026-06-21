# Installation

> 📝 **Stub** — to be expanded from the main `README.md` installation section.

`karpos-downscaling` is published as a Python package and managed via
[`uv`](https://docs.astral.sh/uv/).

## Quick install

```bash
git clone git@github.com:maurinl26/downscaling.git
cd downscaling
uv sync --extra statistical
```

## Optional extras

| Extra | Use case |
|---|---|
| `statistical` | Lapse-rate + QDM + RBF Sencrop pipeline (CPU) |
| `dl` | U-Net FiLM deep-learning pipeline (PyTorch) |
| `prithvi` | NASA/IBM foundation model fine-tuning |
| `viz` | Plotting and notebooks |
| `regrid` | Conservative regridding (xESMF) |
| `docs` | Sphinx documentation build |
| `all` | Full development setup |

See the project [`README.md`](https://github.com/maurinl26/downscaling/blob/main/README.md)
for the authoritative and latest installation instructions.
