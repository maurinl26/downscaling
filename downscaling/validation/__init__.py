"""Cadre de validation actuarielle du risque de base résiduel Karpos.

Ce sous-paquet regroupe les outils de bornage probabiliste utilisés en aval
des pipelines de descente d'échelle (statistique Lot B et DL FiLM Lot C)
pour produire des intervalles de prédiction et des bornes de queue
opposables (audit Atekka).

Cette branche (``feat/cqr-validation``, sous-issue karpos-downscaling#65)
ne ré-exporte que le module ``conformal``. Le module ``gpd_tail``
(sous-issue #67, branche ``feat/gpd-tail-validation``) sera ajouté ici
au merge sur main pour ne pas créer de dépendance croisée entre les
deux PR.

Modules
-------
conformal
    Conformalized Quantile Regression (Romano, Patterson, Candès 2019,
    NeurIPS) avec correction finite-sample (Vovk & Shafer 2005) et
    variante Mondrian conditionnelle (Boström 2020, COPA). Frontline
    method de l'EPIC karpos-downscaling#64.

Voir aussi
----------
EPIC karpos-downscaling#64 (validation risque de base résiduel borné).
"""

from __future__ import annotations

from downscaling.validation.conformal import (
    coverage_gap,
    mondrian_cqr_calibrate,
    mondrian_cqr_predict,
    multi_quantile_loss,
    pinball_loss,
    split_cqr_calibrate,
    split_cqr_predict,
)

__all__ = [
    "coverage_gap",
    "mondrian_cqr_calibrate",
    "mondrian_cqr_predict",
    "multi_quantile_loss",
    "pinball_loss",
    "split_cqr_calibrate",
    "split_cqr_predict",
]
