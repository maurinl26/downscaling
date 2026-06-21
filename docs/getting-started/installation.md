# Installation

`karpos-downscaling` is a Python package targeting **Python 3.11+** and
managed with [`uv`](https://docs.astral.sh/uv/), Astral's fast package and
project manager. We strongly recommend `uv` over `pip` / `conda` for
reproducibility, speed, and lockfile-driven environments.

## Prerequisites

- **Python 3.11 or later**
- **`uv`** — install with:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **System dependencies for scientific libraries** (mostly transparent via
  Python wheels, but worth knowing):
  - On macOS Apple Silicon: nothing special, all dependencies have arm64
    wheels.
  - On Linux: GDAL and PROJ may need OS packages
    (`libgdal-dev`, `libproj-dev`) for some optional extras
    (`cartopy`, `xesmf`).
  - For GPU training: a working CUDA installation if you intend to use the
    `dl` extra on NVIDIA hardware. Apple Silicon MPS works out of the box.

## Cloning the repository

```bash
git clone git@github.com:maurinl26/downscaling.git
cd downscaling
```

## Optional dependency groups

The project exposes several **optional extras** corresponding to the
scientific pipelines and use cases:

| Extra | Contents | Use case |
|---|---|---|
| `statistical` | `scikit-learn`, `elevation`, `dask` | Lapse-rate + QDM + RBF Sencrop pipeline. **Required for the statistical downscaling**. |
| `dl` | `torch`, `pytorch-lightning`, `wandb` | Deep-learning pipeline (U-Net + FiLM conditioning). |
| `prithvi` | `prithviwxc` | NASA/IBM Prithvi WxC foundation model fine-tuning. |
| `pmap` | `pmap` (git source) | PMAP-LES dynamical model integration (advanced). |
| `viz` | `matplotlib`, `cartopy`, `jupyter` | Plotting, notebooks. |
| `regrid` | `xesmf` | Conservative regridding to non-rectilinear grids. |
| `docs` | `sphinx`, `myst-parser`, `sphinx-rtd-theme`, `sphinxcontrib-bibtex`, `sphinx-autodoc-typehints` | Build this documentation locally. |
| `all` | everything above | Full development setup. |

## Recommended installation paths

### Minimum (statistical pipeline only)

```bash
uv sync --extra statistical
```

This installs the lapse-rate, quantile mapping (QDM), and RBF sparse
calibration toolkit on CPU only. Adequate for running `recalibrate_statistical`,
`calibrate_qdm`, `flag_regimes`, and `analyze_recalibrated_statistical`.

### Statistical + deep learning

```bash
uv sync --extra statistical --extra dl
```

Adds PyTorch Lightning and Weights & Biases for the U-Net FiLM pipeline.

### Full developer setup

```bash
uv sync --extra all
```

Pulls every optional dependency, including documentation tooling and the
Prithvi backbone.

## Verifying the installation

```bash
# Make sure the package is importable
uv run python -c "import downscaling; print(downscaling.__file__)"

# Run the test suite
uv run pytest --collect-only -q | head -20

# Run a quick sanity test
uv run pytest tests/test_indices.py -v
```

Tests that depend on `dl` or `prithvi` extras use `pytest.importorskip(...)`
to skip gracefully — so a minimal install can still run the relevant subset.

## Building the documentation locally

```bash
uv sync --extra docs --extra statistical
uv run sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

## External data dependencies

`karpos-downscaling` does not bundle any meteorological data. To run the
pipelines, you will need:

| Dataset | Source | Access |
|---|---|---|
| **CERRA single levels** (`2m_temperature`) | Copernicus CDS | [CDS API](https://cds.climate.copernicus.eu/api-how-to) with your token |
| **CERRA-Land** (`skin_temperature`) | Copernicus CDS | idem |
| **ERA5 synoptic** (`mslp`, `u10`, `v10`, `tcc`, `t2m`, `d2m`) | Copernicus CDS | idem (for regime classification) |
| **DEM** (SRTM 30 m or IGN BD ALTI 25 m) | `elevation` package or IGN | SRTM auto-downloadable via the `elevation` Python package |
| **Sencrop in-situ observations** | Sencrop bulk export | Commercial / partnership agreement with [Sencrop](https://sencrop.com) |

For Sencrop data, please contact [contact@sencrop.com](mailto:contact@sencrop.com)
to discuss research or commercial access. Karpos uses a bulk export of
~4 000 stations across France since 2021.

For CDS API setup, see the
[Copernicus CDS API documentation](https://cds.climate.copernicus.eu/api-how-to).

## Troubleshooting

### `uv sync` fails resolving `cartopy` on Linux

Install system PROJ and GDAL:

```bash
sudo apt install libproj-dev proj-data proj-bin libgdal-dev
```

### `pytorch` does not detect MPS on macOS

Ensure you are on macOS 12.3+ with PyTorch ≥ 2.0. MPS is enabled
automatically; check with:

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

### `cdsapi` reports "Missing/incomplete configuration file"

Create `~/.cdsapirc` with your CDS UID and key:

```yaml
url: https://cds.climate.copernicus.eu/api
key: <UID>:<API-KEY>
```

See [Copernicus CDS API](https://cds.climate.copernicus.eu/api-how-to) for
how to obtain credentials.

### `xesmf` install fails

`xesmf` requires `esmpy` which can be tricky on some platforms. The
statistical pipeline does **not** need `xesmf`; only some advanced regridding
use cases do. Skip the `regrid` extra unless you specifically need it.

## Next steps

- See [Quickstart](quickstart.md) for an end-to-end downscaling example.
- See [User guide](../user-guide/statistical-pipeline.md) for the detailed
  pipeline documentation.
- See [Methodology](../methodology/README.md) for the scientific background.
