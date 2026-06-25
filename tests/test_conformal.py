"""Tests pour ``downscaling.validation.conformal`` (CQR + Mondrian).

Frontline method de l'EPIC validation risque de base résiduel
(karpos-downscaling#64, sous-issue #65). On vérifie ici les propriétés
théoriques sur des données synthétiques :

- pinball loss asymétrique par construction (Koenker & Bassett 1978) ;
- couverture marginale CQR à ``1 - alpha`` sur Gaussien (Romano,
  Patterson, Candès 2019) ;
- couverture conditionnelle Mondrian par strate (Boström 2020).

Les tests sont volontairement légers (n=500 par défaut) pour rester
dans le budget CI ; le test marginal utilise une seed fixée pour ne
pas être flaky.
"""

from __future__ import annotations

import numpy as np
import pytest

# torch est un extra optionnel (pyproject `dl`) ; les tests de pinball loss
# et multi_quantile_loss sont skip si torch n'est pas installé via
# ``requires_torch``. Les tests numpy-only (split CQR, Mondrian,
# coverage_gap) tournent toujours, donc la garantie marginale et
# conditionnelle CQR est couverte en CI même sans extra DL.
try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover, branche CI sans extra dl
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not HAS_TORCH,
    reason="torch is an optional extra (pyproject `dl`)",
)

from downscaling.validation.conformal import (  # noqa: E402
    coverage_gap,
    mondrian_cqr_calibrate,
    mondrian_cqr_predict,
    multi_quantile_loss,
    pinball_loss,
    split_cqr_calibrate,
    split_cqr_predict,
)

# ---------------------------------------------------------------------------
# Pinball loss (Koenker & Bassett 1978)
# ---------------------------------------------------------------------------


@requires_torch
def test_pinball_loss_asymmetry():
    """À tau=0.1, sur-prédire coûte 9x plus que sous-prédire (et inversement)."""
    targets = torch.zeros(100)
    over = torch.full((100,), 1.0)  # pred = target + 1 → r = -1
    under = torch.full((100,), -1.0)  # pred = target - 1 → r = +1

    loss_over_q10 = pinball_loss(over, targets, quantile=0.1).item()
    loss_under_q10 = pinball_loss(under, targets, quantile=0.1).item()
    # r = -1 : loss = (0.1 - 1) * (-1) = 0.9
    # r = +1 : loss =  0.1 * 1        = 0.1
    assert loss_over_q10 == pytest.approx(0.9, abs=1e-6)
    assert loss_under_q10 == pytest.approx(0.1, abs=1e-6)
    assert loss_over_q10 == pytest.approx(9.0 * loss_under_q10, rel=1e-6)


@requires_torch
def test_pinball_loss_symmetric_at_median():
    """À tau=0.5, sur- et sous-prédire de la même amplitude coûte autant."""
    targets = torch.zeros(50)
    over = torch.full((50,), 0.7)
    under = torch.full((50,), -0.7)
    loss_over = pinball_loss(over, targets, quantile=0.5).item()
    loss_under = pinball_loss(under, targets, quantile=0.5).item()
    assert loss_over == pytest.approx(loss_under, abs=1e-6)
    assert loss_over == pytest.approx(0.35, abs=1e-6)  # 0.5 * 0.7


@requires_torch
def test_pinball_loss_invalid_quantile():
    targets = torch.zeros(10)
    preds = torch.zeros(10)
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss(preds, targets, quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss(preds, targets, quantile=1.0)


@requires_torch
def test_multi_quantile_loss_sums_components():
    """multi_quantile_loss = somme des pinball par quantile, pas de pondération."""
    target = torch.zeros(20)
    preds = {
        0.05: torch.full((20,), -1.0),
        0.50: torch.full((20,), 0.0),
        0.95: torch.full((20,), +1.0),
    }
    expected = (
        pinball_loss(preds[0.05], target, 0.05).item()
        + pinball_loss(preds[0.50], target, 0.50).item()
        + pinball_loss(preds[0.95], target, 0.95).item()
    )
    got = multi_quantile_loss(preds, target, [0.05, 0.50, 0.95]).item()
    assert got == pytest.approx(expected, abs=1e-6)


@requires_torch
def test_multi_quantile_loss_missing_head_raises():
    target = torch.zeros(10)
    preds = {0.05: torch.zeros(10), 0.95: torch.zeros(10)}
    with pytest.raises(KeyError, match="missing"):
        multi_quantile_loss(preds, target, [0.05, 0.50, 0.95])


@requires_torch
def test_multi_quantile_unet_forward_backward():
    """End-to-end : MultiQuantileUNet -> multi_quantile_loss -> backward.

    Vérifie que la nouvelle tête multi-quantile branche correctement sur
    la pinball loss et que les gradients remontent à travers le backbone
    FiLM partagé.
    """
    from downscaling.deep_learning.model import MultiQuantileUNet

    m = MultiQuantileUNet(
        quantiles=(0.05, 0.50, 0.95),
        met_in_ch=2,
        met_out_ch=1,
        dem_in_ch=2,
        base_ch=8,  # gardé petit pour la CI
        n_levels=2,
        use_film=True,
        cond_dim=0,
    )
    x_met = torch.randn(2, 2, 8, 8)
    x_dem = torch.randn(2, 2, 8, 8)
    out = m(x_met, x_dem)
    assert set(out.keys()) == {0.05, 0.50, 0.95}
    for q, y in out.items():
        assert y.shape == (2, 1, 8, 8), f"head {q} shape={y.shape}"

    target = torch.randn(2, 1, 8, 8)
    loss = multi_quantile_loss(out, target, [0.05, 0.50, 0.95])
    assert torch.isfinite(loss)
    loss.backward()
    # Au moins un paramètre du backbone a reçu un gradient non nul.
    grads_nonzero = [
        p.grad.abs().sum().item() > 0 for p in m.backbone.parameters() if p.grad is not None
    ]
    assert any(grads_nonzero), "backbone n'a reçu aucun gradient"


# ---------------------------------------------------------------------------
# Split CQR — couverture marginale (Romano, Patterson, Candès 2019)
# ---------------------------------------------------------------------------


def _synthetic_quantile_predictions(rng, n, sigma=1.0, miscal=0.5):
    """Toy data : y ~ N(0, sigma^2), prédicteur quantile mal calibré.

    On simule un prédicteur qui produit un PI trop étroit (``miscal``
    fois l'écart-type vrai à 95 %) : CQR doit alors ajouter une
    correction positive ``q_hat`` pour rétablir la couverture nominale.
    Quand ``miscal=1.0``, le prédicteur est déjà calibré et q_hat ≈ 0.
    """
    y = rng.normal(0.0, sigma, size=n)
    # Borne nominale 95 % d'une Gaussienne : ±1.96 sigma.
    width = miscal * 1.96 * sigma
    q_low = np.full(n, -width)
    q_high = np.full(n, +width)
    return q_low, q_high, y


def test_split_cqr_marginal_coverage_gaussian():
    """Couverture empirique CQR à 95 % +/- 2 pts sur Gaussien (n_cal=500)."""
    rng = np.random.default_rng(42)
    alpha = 0.05

    # Calibration : 500 tirages, prédicteur sous-couvrant (miscal=0.6).
    q_low_c, q_high_c, y_c = _synthetic_quantile_predictions(rng, n=500, miscal=0.6)
    q_hat = split_cqr_calibrate(q_low_c, q_high_c, y_c, alpha=alpha)
    assert q_hat > 0.0, "q_hat doit être positif si le prédicteur sous-couvre"

    # Test : 5000 tirages indépendants, même prédicteur.
    q_low_t, q_high_t, y_t = _synthetic_quantile_predictions(rng, n=5000, miscal=0.6)
    lower, upper = split_cqr_predict(q_low_t, q_high_t, q_hat)
    covered = (y_t >= lower) & (y_t <= upper)
    empirical = covered.mean()
    # Garantie marginale CQR : E[coverage] >= 1 - alpha = 0.95.
    # Empiriquement sur 5000 tirages on tolère ±2 pts.
    assert empirical >= 0.93, f"sous-couverture : {empirical:.3f}"
    assert empirical <= 0.99, f"sur-couverture excessive : {empirical:.3f}"


def test_coverage_gap_signs():
    """coverage_gap > 0 = sur-couverture, < 0 = sous-couverture."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=1000)
    # PI trop large → sur-couverture.
    wide = (np.full_like(y, -5.0), np.full_like(y, 5.0))
    gap_wide = coverage_gap(wide, y, alpha=0.05)
    assert gap_wide > 0
    # PI trop étroit → sous-couverture.
    narrow = (np.full_like(y, -0.1), np.full_like(y, 0.1))
    gap_narrow = coverage_gap(narrow, y, alpha=0.05)
    assert gap_narrow < 0


# ---------------------------------------------------------------------------
# Mondrian CQR (Boström 2020) — couverture par strate
# ---------------------------------------------------------------------------


def test_mondrian_cqr_per_stratum_coverage():
    """3 strates Gaussiennes d'écarts-types différents : Mondrian rétablit
    la couverture >= 92 % strate par strate, là où un q_hat marginal
    sous-couvrirait la strate à forte variance."""
    rng = np.random.default_rng(123)
    alpha = 0.05
    sigmas = {0: 0.5, 1: 1.0, 2: 2.0}

    n_per = 400  # 400 calib + 1000 test par strate
    q_low_c_parts, q_high_c_parts, y_c_parts, strata_c_parts = [], [], [], []
    q_low_t_parts, q_high_t_parts, y_t_parts, strata_t_parts = [], [], [], []
    for label, sigma in sigmas.items():
        # Prédicteur unique mal calibré (largeur fixe ±1) : trop large pour
        # la strate sigma=0.5, trop étroit pour sigma=2.0.
        # Calibration.
        y_c = rng.normal(0.0, sigma, size=n_per)
        q_low_c_parts.append(np.full(n_per, -1.0))
        q_high_c_parts.append(np.full(n_per, +1.0))
        y_c_parts.append(y_c)
        strata_c_parts.append(np.full(n_per, label, dtype=int))
        # Test.
        y_t = rng.normal(0.0, sigma, size=1000)
        q_low_t_parts.append(np.full(1000, -1.0))
        q_high_t_parts.append(np.full(1000, +1.0))
        y_t_parts.append(y_t)
        strata_t_parts.append(np.full(1000, label, dtype=int))

    q_low_c = np.concatenate(q_low_c_parts)
    q_high_c = np.concatenate(q_high_c_parts)
    y_c = np.concatenate(y_c_parts)
    strata_c = np.concatenate(strata_c_parts)

    q_hats = mondrian_cqr_calibrate(
        q_low_c, q_high_c, y_c, strata_c, alpha=alpha, min_per_stratum=20
    )
    assert set(q_hats) == {0, 1, 2}
    # Strate haute variance demande plus de correction.
    assert q_hats[2] > q_hats[0]

    q_low_t = np.concatenate(q_low_t_parts)
    q_high_t = np.concatenate(q_high_t_parts)
    strata_t = np.concatenate(strata_t_parts)
    y_t = np.concatenate(y_t_parts)
    lower, upper = mondrian_cqr_predict(q_low_t, q_high_t, strata_t, q_hats)

    for label in sigmas:
        mask = strata_t == label
        covered = (y_t[mask] >= lower[mask]) & (y_t[mask] <= upper[mask])
        empirical = covered.mean()
        assert empirical >= 0.92, (
            f"sous-couverture sur strate {label} : {empirical:.3f} "
            f"(sigma={sigmas[label]}, q_hat={q_hats[label]:.3f})"
        )


def test_mondrian_cqr_rejects_tiny_stratum():
    rng = np.random.default_rng(7)
    q_low = rng.normal(-1.0, 0.01, size=100)
    q_high = rng.normal(+1.0, 0.01, size=100)
    y = rng.normal(0.0, 1.0, size=100)
    strata = np.zeros(100, dtype=int)
    strata[-5:] = 1  # une strate de 5 éléments seulement
    with pytest.raises(ValueError, match="stratum 1"):
        mondrian_cqr_calibrate(q_low, q_high, y, strata, alpha=0.05, min_per_stratum=20)


def test_split_cqr_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        split_cqr_calibrate(np.zeros(10), np.zeros(10), np.zeros(9), alpha=0.05)
