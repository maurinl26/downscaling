"""Pipeline d'entrée MERRA-2 pour le vrai backbone Prithvi WxC (issue #1).

Le backbone réel consomme un ``batch`` structuré (``x, y, static, climate,
input_time, lead_time``) sur la grille MERRA-2. On **délègue à l'API officielle**
du package ``PrithviWxC`` plutôt que de réimplémenter le format :

  - ``PrithviWxC.dataloaders.merra2.Merra2Dataset`` lit les NetCDF MERRA-2
    (surface + niveaux) + la climatologie et renvoie l'échantillon brut.
  - ``PrithviWxC.dataloaders.merra2.preproc`` est la *collate* qui assemble le
    ``batch`` attendu par ``PrithviWxC.forward`` (concatène surface + vertical en
    160 canaux, applique le padding d'alignement de grille).

Ce module fournit des fabriques minces (dataset + DataLoader) paramétrées par la
config Prithvi (variables / niveaux), pour que les entrées correspondent au
modèle chargé via :func:`downscaling.prtihvi_wxc.loader.load_prithvi_backbone`.

Caveat : le ``padding`` (alignement grille MERRA-2 → grille modèle) dépend des
données ; fournir celui de la notebook officielle pour les vrais fichiers.
"""

from __future__ import annotations

from functools import partial

ZERO_PADDING = {"level": (0, 0), "lat": (0, 0), "lon": (0, 0)}


def merra2_collate(padding: dict | None = None):
    """Retourne la *collate* officielle ``preproc`` figée avec son ``padding``."""
    from PrithviWxC.dataloaders.merra2 import preproc

    return partial(preproc, padding=padding or ZERO_PADDING)


def build_merra2_dataset(
    config,
    *,
    data_path_surface,
    data_path_vertical,
    time_range,
    lead_times=(0,),
    input_times=(-6,),
    climatology_path_surface=None,
    climatology_path_vertical=None,
    positional_encoding: str = "fourier",
):
    """Construit le ``Merra2Dataset`` officiel, variables/niveaux issus de la config.

    Args:
        config: ``PrithviWxCConfig`` (ou objet exposant ``surface_vars``,
            ``static_surface_vars``, ``vertical_vars``, ``levels``).
        data_path_surface / data_path_vertical: répertoires NetCDF MERRA-2.
        time_range: ``(début, fin)`` (str ou Timestamp).
        lead_times / input_times: horizons (h) — cf. API officielle.
        climatology_path_*: requis si le modèle utilise ``residual="climate"``.
        positional_encoding: doit matcher la config du modèle.
    """
    from PrithviWxC.dataloaders.merra2 import Merra2Dataset

    return Merra2Dataset(
        time_range=time_range,
        lead_times=list(lead_times),
        input_times=list(input_times),
        data_path_surface=data_path_surface,
        data_path_vertical=data_path_vertical,
        climatology_path_surface=climatology_path_surface,
        climatology_path_vertical=climatology_path_vertical,
        surface_vars=list(config.surface_vars),
        static_surface_vars=list(config.static_surface_vars),
        vertical_vars=list(config.vertical_vars),
        levels=list(config.levels),
        positional_encoding=positional_encoding,
    )


def build_prithvi_dataloader(dataset, *, padding=None, batch_size: int = 1, num_workers: int = 0):
    """``DataLoader`` produisant des ``batch`` prêts pour ``PrithviWxC.forward``."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=merra2_collate(padding),
        shuffle=False,
    )
