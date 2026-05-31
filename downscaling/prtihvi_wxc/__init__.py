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

# dataset/loader/inference tirent l'extra `prithvi`/`dl` (torch, huggingface_hub,
# poids HF). On les rend optionnels — en blocs séparés pour qu'une indispo de l'un
# ne masque pas l'autre — afin que les sous-modules légers (netatmo_qc, sencrop,
# stations) restent importables sans torch ni l'extra complet (ex. en CI).
try:
    from .dataset import FrostNightDataset
except ImportError:  # pragma: no cover - torch absent
    FrostNightDataset = None

try:
    from .loader import DEMConditionedAdapter, PrithviWxCDownscaler
except ImportError:  # pragma: no cover - dépend de l'install de l'extra prithvi
    PrithviWxCDownscaler = DEMConditionedAdapter = None

try:
    from .inference import FrostReanalysisRunner, load_config
except ImportError:  # pragma: no cover - dépend de l'install de l'extra prithvi
    FrostReanalysisRunner = load_config = None

__all__ = [
    "FrostNightDataset",
    "FrostReanalysisRunner",
    "PrithviWxCDownscaler",
    "DEMConditionedAdapter",
    "load_config",
]
