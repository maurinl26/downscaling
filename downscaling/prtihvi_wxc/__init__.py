"""
deep_learning.prithvi_wxc
=========================
Inférence Prithvi WxC (NASA/IBM) pour downscaling avec conditioning orographique.

Composants :
  - loader.py   : PrithviWxCDownscaler (backbone + CNN adapter + DEM)
  - dataset.py  : FrostNightDataset (paires ERA5 + DEM, filtrées nuits de gel)
  - inference.py: FrostReanalysisRunner (rolling temporel → Tmin HR → Zarr)

Exemple rapide :
    from deep_learning.prithvi_wxc import FrostReanalysisRunner, FrostNightDataset
    from deep_learning.prithvi_wxc.loader import PrithviWxCDownscaler

    dataset = FrostNightDataset("data/era5/", "data/dem/copdem_drome.tif")
    runner  = FrostReanalysisRunner(config={})
    model   = runner.load_model()  # télécharge depuis HuggingFace
    runner.run(model, dataset, "output/frost_prithvi.zarr")
"""

from .dataset import FrostNightDataset
from .inference import FrostReanalysisRunner, load_config
from .loader import PrithviWxCDownscaler, DEMConditionedAdapter

__all__ = [
    "FrostNightDataset",
    "FrostReanalysisRunner",
    "PrithviWxCDownscaler",
    "DEMConditionedAdapter",
    "load_config",
]
