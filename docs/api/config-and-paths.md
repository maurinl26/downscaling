# Configuration and paths

## `downscaling.config`

Hydra-based configuration loading. Replaces the previous flat
`yaml.safe_load(...)` workflow with composition-based configs under
`configs/`.

```{eval-rst}
.. automodule:: downscaling.config
   :members:
   :show-inheritance:
   :member-order: bysource
```

## `downscaling.paths`

Absolute path constants used across the project (config directory, data
roots, output directories). Centralised here so that console entry points
work regardless of the working directory.

```{eval-rst}
.. automodule:: downscaling.paths
   :members:
   :show-inheritance:
   :member-order: bysource
```
