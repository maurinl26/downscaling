# API reference

This section is auto-generated from Python docstrings in the
`downscaling` package. Pages are grouped by sub-package:

- **[Shared utilities](shared.md)** — common loaders (ERA5, CERRA, DEM),
  regridding helpers, parametric indices (frost, BBCH thresholds).
- **[Deep learning](deep-learning.md)** — U-Net with FiLM conditioning,
  PyTorch Lightning training modules, sparse Sencrop calibration loss,
  CERRA provider, inference.
- **[Configuration and paths](config-and-paths.md)** — Hydra-based
  configuration loading and project-level path constants.

The command-line entry points under `downscaling.scripts.*` are documented
in the [user guide](../user-guide/statistical-pipeline.md) and via the
`--help` flag on each script, rather than autodoc-generated pages.

```{toctree}
:maxdepth: 2
:hidden:

shared
deep-learning
config-and-paths
```
