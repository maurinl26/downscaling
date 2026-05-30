"""Chargement de la configuration via composition Hydra.

Remplace l'ancien ``yaml.safe_load("config/drome_ardeche.yml")``. ``load_config``
compose les groupes de ``configs/`` et retourne un **dict plat** dont les clés de
premier niveau reproduisent l'ancien monolithe (``domain``, ``data``,
``statistical``, ``deep_learning``, ``indices``, ``cluster``, ``run``) — les
pipelines et scripts existants restent ainsi inchangés.

Les overrides Hydra se passent en liste, p. ex. ::

    load_config(["experiment=drome_ardeche", "cluster=cloud", "dl.epochs=50"])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from downscaling.paths import CONFIG_DIR


def load_config(overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Compose la config Hydra et la renvoie comme dict (forme monolithe).

    ``config_dir`` est absolu (cf. :mod:`downscaling.paths`) pour fiabiliser
    l'appel depuis un entry point console.
    """
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # Le groupe Hydra `dl` correspond au bloc monolithe `deep_learning`.
    cfg_dict.setdefault("deep_learning", cfg_dict.get("dl", {}))
    return cfg_dict
