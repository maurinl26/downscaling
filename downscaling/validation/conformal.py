"""Conformalized Quantile Regression (CQR) pour le U-Net FiLM Karpos.

Ce module implémente le bornage probabiliste opposable du U-Net FiLM
multi-quantile utilisé pour la descente d'échelle Tmin nocturne sur les
Baronnies. Il s'agit de la frontline method de l'EPIC validation du
risque de base résiduel (karpos-downscaling#64), sous-issue #65.

Méthode
-------
Soit ``q_low(x)`` et ``q_high(x)`` deux prédictions quantiles produites
par le réseau (typiquement tau=0.05 et tau=0.95) entraînées avec la
pinball loss de Koenker & Bassett (1978). Sur un jeu de calibration
exchangeable disjoint du jeu de test on calcule les scores de
non-conformité CQR de Romano, Patterson, Candès (2019) :

    s_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))

Puis le quantile finite-sample corrigé (Vovk & Shafer 2005) :

    q_hat = Quantile_{(1-alpha)(1 + 1/n_cal)}(s_1, ..., s_{n_cal})

Le prédicteur conformalisé renvoie l'intervalle élargi symétriquement :

    [q_low(x_test) - q_hat, q_high(x_test) + q_hat]

Garantie marginale : ``P(y_test in PI) >= 1 - alpha`` sous l'hypothèse
d'échangeabilité des paires ``(x_i, y_i)``. La garantie est
distribution-free.

Variante Mondrian conditionnelle (Boström 2020, COPA) : on partitionne
le jeu de calibration en strates exogènes (par exemple régime FiLM x bin
d'altitude x saison) et on calcule un ``q_hat`` par strate. La couverture
est alors garantie strate par strate, propriété nécessaire ici car la
moyenne de domaine peut masquer un sous-couverture catastrophique à
Nyons (effet cuvette de fond de vallée, cf. project-nyons-cold-pool).

Références
----------
- Romano, Y., Patterson, E., Candès, E. (2019). "Conformalized Quantile
  Regression". NeurIPS 32. https://arxiv.org/abs/1905.03222
- Vovk, V., Shafer, G. (2005). "Algorithmic Learning in a Random World".
  Springer. Chapitre 2 (correction finite-sample du quantile empirique).
- Boström, H. (2020). "Mondrian Conformal Predictors with a Single
  Calibration Set". COPA 2020. PMLR 128.
- Koenker, R., Bassett, G. (1978). "Regression Quantiles". Econometrica
  46(1), 33-50. (Pinball loss originelle.)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover — imports types pour mypy uniquement
    import torch


# ---------------------------------------------------------------------------
# Pinball loss (Koenker & Bassett 1978)
# ---------------------------------------------------------------------------


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Loss pinball (quantile regression) asymétrique pour un quantile fixé.

    Pour ``r = targets - predictions`` :

    - ``r >= 0`` (cible au-dessus de la prédiction) : pénalité ``quantile * r``
    - ``r <  0`` (cible en-dessous de la prédiction) : pénalité ``(quantile - 1) * r``

    Le minimiseur de l'espérance de cette loss est le quantile conditionnel
    de niveau ``quantile`` (Koenker & Bassett 1978).

    Parameters
    ----------
    predictions, targets:
        Tenseurs de même forme arbitraire (B, C, H, W) ou (N,). La
        réduction est une moyenne sur tous les éléments.
    quantile:
        Niveau de quantile dans (0, 1). Par exemple 0.05 pour la borne
        basse, 0.95 pour la borne haute.

    Returns
    -------
    torch.Tensor
        Scalaire (moyenne sur tous les éléments).

    Raises
    ------
    ValueError
        Si ``quantile`` n'est pas dans (0, 1) ou si les formes diffèrent.
    """
    import torch  # lazy : torch est extra optionnel (cf. pyproject `dl`)

    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must lie in (0, 1), got {quantile}")
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions {tuple(predictions.shape)} and targets "
            f"{tuple(targets.shape)} must share the same shape"
        )
    residual = targets - predictions
    loss = torch.where(
        residual >= 0,
        quantile * residual,
        (quantile - 1.0) * residual,
    )
    return loss.mean()


def multi_quantile_loss(
    preds: Mapping[float, torch.Tensor],
    target: torch.Tensor,
    quantiles: Iterable[float],
) -> torch.Tensor:
    """Somme des pinball losses pour plusieurs quantiles (multi-tête).

    Aucun terme de crossing penalty n'est ajouté ici : la cohérence
    ``q_low <= q_50 <= q_high`` se garantit en pratique soit par
    rééchantillonnage (Chernozhukov, Fernandez-Val, Galichon 2010) soit
    par un post-traitement par tri. Pour le scaffolding, on garde la
    loss minimale ; on ajoutera le crossing penalty si nécessaire au
    moment du training réel.

    Parameters
    ----------
    preds:
        Mapping ``{quantile: tenseur}`` (mêmes formes), typiquement
        ``{0.05: yq05, 0.50: yq50, 0.95: yq95}``.
    target:
        Cible (même forme que chaque prédiction).
    quantiles:
        Itérable des niveaux à sommer. Doit être un sous-ensemble des
        clés de ``preds``.

    Returns
    -------
    torch.Tensor
        Scalaire, somme non pondérée des pinball.
    """
    import torch  # lazy : torch est extra optionnel (cf. pyproject `dl`)

    quantiles_list = list(quantiles)
    if not quantiles_list:
        raise ValueError("quantiles is empty")
    missing = [q for q in quantiles_list if q not in preds]
    if missing:
        raise KeyError(f"preds is missing quantile heads: {missing}")
    losses = [pinball_loss(preds[q], target, q) for q in quantiles_list]
    return torch.stack(losses).sum()


# ---------------------------------------------------------------------------
# Split Conformalized Quantile Regression (Romano, Patterson, Candès 2019)
# ---------------------------------------------------------------------------


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Quantile empirique corrigé finite-sample (Vovk & Shafer 2005).

    Renvoie le quantile de niveau ``ceil((n + 1) * (1 - alpha)) / n`` de
    ``scores``, qui est exactement la valeur garantissant la couverture
    marginale ``1 - alpha`` pour l'échantillon de test échangeable de
    taille 1 (CQR Theorem 1 dans Romano et al. 2019).
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    n = scores.size
    if n == 0:
        raise ValueError("scores is empty; cannot calibrate")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    # Quantile level corrigé : (1 - alpha) * (1 + 1/n), clamp à 1.0 pour
    # les petits n où la couverture exacte n'est pas atteignable.
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    return float(np.quantile(scores, level, method="higher"))


def split_cqr_calibrate(
    q_low: np.ndarray,
    q_high: np.ndarray,
    y_calib: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Calibre la correction conformelle ``q_hat`` à partir d'un jeu hold-out.

    Calcule les scores de non-conformité CQR :

        s_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))

    et renvoie le quantile empirique finite-sample corrigé.

    Parameters
    ----------
    q_low, q_high:
        Prédictions des quantiles bas et haut sur le jeu de calibration,
        de même forme arbitraire (aplatie en interne). On suppose
        l'échangeabilité ``(x_i, y_i)`` mais pas l'absence de quantile
        crossing : ``max`` couvre les deux côtés.
    y_calib:
        Observations sur le jeu de calibration, même forme.
    alpha:
        Niveau de risque cible (couverture cible = ``1 - alpha``). Par
        défaut 0.05 (PI à 95 %).

    Returns
    -------
    float
        ``q_hat`` : correction additive symétrique à appliquer à
        ``q_low`` et ``q_high`` pour obtenir l'intervalle conformalisé.
    """
    q_low = np.asarray(q_low).reshape(-1)
    q_high = np.asarray(q_high).reshape(-1)
    y_calib = np.asarray(y_calib).reshape(-1)
    if not (q_low.shape == q_high.shape == y_calib.shape):
        raise ValueError(
            f"shape mismatch: q_low={q_low.shape}, q_high={q_high.shape}, y_calib={y_calib.shape}"
        )
    scores = np.maximum(q_low - y_calib, y_calib - q_high)
    return _finite_sample_quantile(scores, alpha)


def split_cqr_predict(
    q_low: np.ndarray,
    q_high: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Applique la correction ``q_hat`` à un jeu de test.

    Renvoie l'intervalle de prédiction conformalisé :

        [q_low - q_hat, q_high + q_hat]

    qui satisfait ``P(y in PI) >= 1 - alpha`` (couverture marginale,
    finite-sample, distribution-free, sous échangeabilité).

    Parameters
    ----------
    q_low, q_high:
        Prédictions des quantiles bas et haut sur le jeu de test.
    q_hat:
        Correction calibrée renvoyée par ``split_cqr_calibrate``.

    Returns
    -------
    (lower, upper):
        Tuple de deux ``np.ndarray`` de même forme que les entrées.
    """
    q_low = np.asarray(q_low)
    q_high = np.asarray(q_high)
    return q_low - q_hat, q_high + q_hat


# ---------------------------------------------------------------------------
# Mondrian CQR (Boström 2020) — couverture par strate
# ---------------------------------------------------------------------------


def mondrian_cqr_calibrate(
    q_low: np.ndarray,
    q_high: np.ndarray,
    y_calib: np.ndarray,
    strata: np.ndarray,
    alpha: float = 0.05,
    min_per_stratum: int = 20,
) -> dict[int, float]:
    """Calibration conformelle Mondrian par strate (Boström 2020, COPA).

    Partitionne ``(q_low, q_high, y_calib)`` selon ``strata`` (étiquettes
    entières) et applique ``split_cqr_calibrate`` à chaque strate.

    L'utilisation type chez Karpos est strata = encodage (regime FiLM x
    bin d'altitude x saison) — la stratification doit être *exogène*
    (indépendante de y) pour préserver l'échangeabilité au sein de
    chaque strate.

    Parameters
    ----------
    q_low, q_high, y_calib:
        Mêmes signatures que ``split_cqr_calibrate``, formes arbitraires
        aplaties en interne.
    strata:
        Tableau d'étiquettes entières de même forme aplatie. Une strate
        avec moins de ``min_per_stratum`` éléments lève ``ValueError``
        (garantie de couverture non significative en dessous).
    alpha:
        Niveau de risque cible par strate.
    min_per_stratum:
        Taille minimale d'une strate pour autoriser la calibration.
        Défaut 20, à augmenter si on vise un coverage gap serré.

    Returns
    -------
    dict[int, float]
        Mapping ``{label_strate: q_hat}``.

    Raises
    ------
    ValueError
        Si une strate est trop petite ou si les formes diffèrent.
    """
    q_low = np.asarray(q_low).reshape(-1)
    q_high = np.asarray(q_high).reshape(-1)
    y_calib = np.asarray(y_calib).reshape(-1)
    strata = np.asarray(strata).reshape(-1)
    if not (q_low.shape == q_high.shape == y_calib.shape == strata.shape):
        raise ValueError(
            f"shape mismatch: q_low={q_low.shape}, q_high={q_high.shape}, "
            f"y_calib={y_calib.shape}, strata={strata.shape}"
        )
    q_hats: dict[int, float] = {}
    for label in np.unique(strata):
        mask = strata == label
        n = int(mask.sum())
        if n < min_per_stratum:
            raise ValueError(
                f"stratum {int(label)} has only {n} samples, below "
                f"min_per_stratum={min_per_stratum}; either pool strata "
                f"or collect more calibration data"
            )
        q_hats[int(label)] = split_cqr_calibrate(
            q_low[mask], q_high[mask], y_calib[mask], alpha=alpha
        )
    return q_hats


def mondrian_cqr_predict(
    q_low: np.ndarray,
    q_high: np.ndarray,
    strata: np.ndarray,
    q_hats: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Applique les corrections par strate à un jeu de test.

    Parameters
    ----------
    q_low, q_high:
        Prédictions quantiles bas et haut sur le jeu de test.
    strata:
        Étiquettes de strate (mêmes que ``mondrian_cqr_calibrate``).
        Toute valeur absente de ``q_hats`` lève ``KeyError``.
    q_hats:
        Mapping ``{label: q_hat}`` produit par
        ``mondrian_cqr_calibrate``.

    Returns
    -------
    (lower, upper):
        Tuple de deux ``np.ndarray`` aplatis.
    """
    q_low = np.asarray(q_low).reshape(-1)
    q_high = np.asarray(q_high).reshape(-1)
    strata = np.asarray(strata).reshape(-1)
    if not (q_low.shape == q_high.shape == strata.shape):
        raise ValueError(
            f"shape mismatch: q_low={q_low.shape}, q_high={q_high.shape}, strata={strata.shape}"
        )
    lower = np.empty_like(q_low, dtype=float)
    upper = np.empty_like(q_high, dtype=float)
    for i, label in enumerate(strata):
        key = int(label)
        if key not in q_hats:
            raise KeyError(
                f"stratum {key} not present in calibration q_hats (keys={sorted(q_hats)})"
            )
        q_hat = q_hats[key]
        lower[i] = q_low[i] - q_hat
        upper[i] = q_high[i] + q_hat
    return lower, upper


# ---------------------------------------------------------------------------
# Diagnostic : coverage gap empirique
# ---------------------------------------------------------------------------


def coverage_gap(
    predictions_pi: tuple[np.ndarray, np.ndarray],
    y_true: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Écart entre la couverture nominale ``1 - alpha`` et l'empirique.

    Renvoie ``coverage_empirique - (1 - alpha)`` (positif = sur-couverture,
    négatif = sous-couverture). Pour l'audit Atekka on regarde la valeur
    absolue ; le gate de l'issue #65 est ``|coverage_gap| <= 0.03`` à
    alpha=0.05 sur la pire strate.

    Parameters
    ----------
    predictions_pi:
        Tuple ``(lower, upper)`` de l'intervalle de prédiction.
    y_true:
        Observations.
    alpha:
        Niveau cible.

    Returns
    -------
    float
        ``empirical_coverage - (1 - alpha)``.
    """
    lower, upper = predictions_pi
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    if not (lower.shape == upper.shape == y_true.shape):
        raise ValueError(
            f"shape mismatch: lower={lower.shape}, upper={upper.shape}, y_true={y_true.shape}"
        )
    covered = (y_true >= lower) & (y_true <= upper)
    empirical = float(covered.mean())
    return empirical - (1.0 - alpha)
