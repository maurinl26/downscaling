# Deep learning

U-Net with FiLM conditioning on the digital elevation model and the
synoptic regime, with sparse Sencrop calibration loss. Trained via
PyTorch Lightning.

## `downscaling.deep_learning.model`

Neural architectures: FiLM layers, DEM encoder, full U-Net, lightweight
SRCNN baseline, and the `build_model` factory.

```{eval-rst}
.. automodule:: downscaling.deep_learning.model
   :members:
   :show-inheritance:
   :member-order: bysource
```

## `downscaling.deep_learning.lightning_module`

PyTorch Lightning training module and data module wrapping the U-Net.

```{eval-rst}
.. automodule:: downscaling.deep_learning.lightning_module
   :members:
   :show-inheritance:
   :member-order: bysource
```

## `downscaling.deep_learning.dataset`

PyTorch `Dataset` for downscaling training: pairs of coarse atmospheric
input + DEM context + fine-resolution target.

```{eval-rst}
.. automodule:: downscaling.deep_learning.dataset
   :members:
   :show-inheritance:
   :member-order: bysource
```

## `downscaling.deep_learning.cerra_provider`

CERRA NetCDF provider used by the deep-learning dataset and inference
pipelines.

```{eval-rst}
.. automodule:: downscaling.deep_learning.cerra_provider
   :members:
   :show-inheritance:
   :member-order: bysource
```

## `downscaling.deep_learning.sparse_calibration`

Sparse Sencrop calibration loss and the associated Lightning module for
fine-tuning a pre-trained downscaling backbone on in-situ observations.

```{eval-rst}
.. automodule:: downscaling.deep_learning.sparse_calibration
   :members:
   :show-inheritance:
   :member-order: bysource
```

## `downscaling.deep_learning.train`

Training utilities: cosine learning rate with warm-up, training entry
point.

```{eval-rst}
.. automodule:: downscaling.deep_learning.train
   :members:
   :show-inheritance:
   :member-order: bysource
```

## `downscaling.deep_learning.inference`

Inference utilities for applying a trained model on a CERRA year.

```{eval-rst}
.. automodule:: downscaling.deep_learning.inference
   :members:
   :show-inheritance:
   :member-order: bysource
```
