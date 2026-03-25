#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "scipy>=1.10",
#   "xarray>=2023.1",
#   "netCDF4>=1.6",
#   "rasterio>=1.3",
#   "pyyaml>=6.0",
#   "torch>=2.0",
#   "downscaling",
# ]
# ///
"""
Entraîne le U-Net de descente d'échelle conditionné par le MNT.

Préparer les données d'entraînement au préalable :
    data/training/
    ├── coarse/   ← ERA5 ou CERRA dégradé regrillé sur la grille fine (NetCDF)
    ├── fine/     ← CERRA haute résolution ou SAFRAN (NetCDF, même grille)
    └── dem_attributes.nc   ← sortie de DEMLoader.terrain_attributes()

Exemple
-------
    python scripts/run_dl_train.py \
        --config config/drome_ardeche.yml \
        --data-dir data/training/ \
        --epochs 100 \
        --batch-size 8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_learning.train import main

if __name__ == "__main__":
    main()
