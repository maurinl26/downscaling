"""Cadre de validation actuarielle du risque de base résiduel Karpos.

Ce sous-paquet regroupe les outils de bornage probabiliste utilisés en aval
des pipelines de descente d'échelle (statistique Lot B et DL FiLM KarposSR)
pour produire des intervalles de prédiction et des bornes de queue
opposables (audit Atekka).

Modules
-------
conformal
    Conformalized Quantile Regression (Romano, Patterson, Candès 2019,
    NeurIPS) avec correction finite-sample (Vovk & Shafer 2005) et
    variante Mondrian conditionnelle (Boström 2020, COPA). Frontline
    method de l'EPIC karpos-downscaling#64. Cf. sous-issue #65.

gpd_tail
    POT (Peaks-Over-Threshold) + GPD (Generalized Pareto Distribution) sur
    les résidus QDM en queue gauche froide (Coles 2001 chap 4-5). Bornage
    des extrêmes pour le pricing actuarial. Cf. sous-issue #67.

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
from downscaling.validation.gpd_tail import (
    GPDFit,
    fit_gpd_pot,
    ks_test_gpd,
    mean_excess_plot,
    quantile_gpd,
)

__all__ = [
    # conformal (issue #65)
    "coverage_gap",
    "mondrian_cqr_calibrate",
    "mondrian_cqr_predict",
    "multi_quantile_loss",
    "pinball_loss",
    "split_cqr_calibrate",
    "split_cqr_predict",
    # gpd_tail (issue #67)
    "GPDFit",
    "fit_gpd_pot",
    "ks_test_gpd",
    "mean_excess_plot",
    "quantile_gpd",
]
