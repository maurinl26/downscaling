"""Chemins absolus du projet.

``CONFIG_DIR`` doit être **absolu** : Hydra, appelé depuis un entry point console
(``downscaling-run``), interprète un ``config_path`` relatif comme un module
Python et échoue à le trouver. On l'ancre donc sur l'emplacement de ce fichier.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Racine du dépôt (dossier parent du package ``downscaling/``).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Dossier des configs Hydra composables (``configs/``).
CONFIG_DIR = REPO_ROOT / "configs"

#: Cache local des poids pré-entraînés (Prithvi WxC, Granite). Surchargeable par
#: ``DOWNSCALING_MODELSTORE`` (ex. volume réseau RunPod monté). Ignoré par git.
MODELSTORE = Path(os.environ.get("DOWNSCALING_MODELSTORE", REPO_ROOT / "modelstore"))
