# Shared utilities

Common readers and helpers used across the statistical, deep-learning, and
foundation-model pipelines.

## `downscaling.shared.loaders`

NetCDF readers for the main scientific inputs: ERA5, CERRA single levels
and CERRA-Land, and DEM (digital elevation model). Also includes a generic
`regrid_to_dem` utility.

```{eval-rst}
.. automodule:: downscaling.shared.loaders
   :members:
   :show-inheritance:
   :member-order: bysource
```

## `downscaling.shared.indices`

Parametric agricultural insurance indices computed on downscaled
meteorological fields (frost-window detection, BBCH-stage thresholds,
event severity classification, etc.).

```{eval-rst}
.. automodule:: downscaling.shared.indices
   :members:
   :show-inheritance:
   :member-order: bysource
```
