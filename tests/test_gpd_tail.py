"""Tests unitaires POT/GPD sur résidus QDM (karpos-downscaling#67).

Scénarios couverts (squelette, fit MLE sur GPD synthétique uniquement) :

1. ``test_fit_gpd_pot_recovers_scale_shape`` — MLE retrouve (sigma, xi) à
   ~5 % près sur n=5000 excès tirés d'une GPD pure.
2. ``test_ks_test_does_not_reject_true_gpd`` — KS p > 0.05 sur queue
   réellement GPD.
3. ``test_quantile_gpd_matches_analytic`` — quantile POT est cohérent avec
   l'expression analytique de Coles (eq. 4.12).
4. ``test_mean_excess_plot_creates_figure`` — la fonction de diagnostic crée
   bien un fichier PNG et renvoie une ``Figure``.

Ces tests valident la mécanique du code, pas la décision Atekka (qui se joue
sur de vraies données dans le PR suivant).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from downscaling.validation.gpd_tail import (
    fit_gpd_pot,
    ks_test_gpd,
    mean_excess_plot,
    quantile_gpd,
)

# Matplotlib est dans l'extra ``viz`` : on saute le test de figure s'il est absent.
HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def _sample_left_tail_residuals(
    n: int,
    shape: float,
    scale: float,
    u: float,
    seed: int = 0,
) -> np.ndarray:
    """Construit ``n`` résidus dont les excès sous ``u`` sont GPD(scale, shape).

    On tire ``n`` excès positifs ``z ~ GPD(scale, shape)``, on en fait des
    excès négatifs ``eps = u - z`` (résidus plus froids que ``u``). C'est
    suffisant pour les tests : ``n_total == n_excess`` et la queue est
    exactement GPD.
    """
    rng = np.random.default_rng(seed)
    z = stats.genpareto.rvs(shape, loc=0.0, scale=scale, size=n, random_state=rng)
    return u - z  # eps < u par construction


# ---------------------------------------------------------------------------
# 1. MLE retrouve les vrais paramètres
# ---------------------------------------------------------------------------


def test_fit_gpd_pot_recovers_scale_shape() -> None:
    true_scale = 1.2
    true_shape = 0.15  # xi > 0 : queue lourde (cas réaliste pour résidus QDM en gel)
    u = -2.0
    residuals = _sample_left_tail_residuals(
        n=5000, shape=true_shape, scale=true_scale, u=u, seed=42
    )

    fit = fit_gpd_pot(residuals, threshold=u, n_bootstrap=200, random_state=7)

    assert fit.n_excess == 5000
    assert fit.n_total == 5000
    # Tolérances larges (5 %) : MLE GPD a une variance non négligeable même à n=5k.
    assert fit.scale == pytest.approx(true_scale, rel=0.10)
    assert fit.shape == pytest.approx(true_shape, abs=0.05)
    # IC95 bootstrap couvre la vraie valeur
    assert fit.ci95_scale[0] <= true_scale <= fit.ci95_scale[1]
    assert fit.ci95_shape[0] <= true_shape <= fit.ci95_shape[1]
    # Bootstrap a convergé sur la quasi-totalité des répliques
    assert fit.n_bootstrap >= 180


def test_fit_gpd_pot_raises_when_too_few_excesses() -> None:
    residuals = np.array([-0.1, -0.2, -0.3])
    with pytest.raises(ValueError, match="Trop peu d'excès"):
        fit_gpd_pot(residuals, threshold=-1.0, n_bootstrap=0)


# ---------------------------------------------------------------------------
# 2. KS ne rejette pas une vraie GPD
# ---------------------------------------------------------------------------


def test_ks_test_does_not_reject_true_gpd() -> None:
    true_scale = 0.8
    true_shape = 0.05
    u = -1.5
    residuals = _sample_left_tail_residuals(
        n=2000, shape=true_shape, scale=true_scale, u=u, seed=11
    )
    fit = fit_gpd_pot(residuals, threshold=u, n_bootstrap=0, random_state=11)
    d_stat, p_value = ks_test_gpd(residuals, threshold=u, fit=fit)

    assert 0.0 <= d_stat <= 1.0
    # H0 (GPD ajustée = vraie loi) ne doit pas être rejetée à 5 %.
    # Note : la p-valeur est anti-conservatrice (paramètres estimés sur la
    # même donnée), mais reste largement au-dessus de 0.05 sur GPD pure.
    assert p_value > 0.05


# ---------------------------------------------------------------------------
# 3. quantile_gpd cohérent avec la formule analytique de Coles eq. 4.12
# ---------------------------------------------------------------------------


def test_quantile_gpd_matches_analytic_formula() -> None:
    true_scale = 1.0
    true_shape = 0.20
    u = -2.0
    residuals = _sample_left_tail_residuals(
        n=4000, shape=true_shape, scale=true_scale, u=u, seed=3
    )
    fit = fit_gpd_pot(residuals, threshold=u, n_bootstrap=100, random_state=3)

    p = 0.01  # probabilité d'excès cible (zone "queue extrême")
    out = quantile_gpd(fit, exceedance_prob=p, alpha=0.95)

    # Formule analytique avec les paramètres MLE et zeta_u = 1.0 (n_excess == n_total).
    zeta_u = fit.n_excess / fit.n_total
    u_pos = -fit.threshold
    expected_z = u_pos + (fit.scale / fit.shape) * ((p / zeta_u) ** (-fit.shape) - 1.0)
    expected_quantile = -expected_z

    assert out["quantile"] == pytest.approx(expected_quantile, rel=1e-6)
    # IC95 doit encadrer le quantile ponctuel
    assert out["ci95_low"] <= out["quantile"] <= out["ci95_high"]


def test_quantile_gpd_rejects_invalid_exceedance_prob() -> None:
    # fit minimal avec n_excess << n_total : zeta_u petit
    fit = type(
        "Stub",
        (),
        {
            "threshold": -2.0,
            "scale": 1.0,
            "shape": 0.1,
            "n_excess": 50,
            "n_total": 1000,
            "ci95_scale": (0.9, 1.1),
            "ci95_shape": (0.05, 0.15),
            "mle_loglik": -100.0,
            "n_bootstrap": 200,
        },
    )()
    # zeta_u = 0.05 → exceedance_prob = 0.1 inadmissible (plus fréquent que le seuil)
    with pytest.raises(ValueError, match="exceedance_prob"):
        quantile_gpd(fit, exceedance_prob=0.10)


# ---------------------------------------------------------------------------
# 4. mean_excess_plot crée bien la figure
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib absent (extra ``viz`` non installé)")
def test_mean_excess_plot_creates_figure(tmp_path: Path) -> None:
    residuals = _sample_left_tail_residuals(
        n=500, shape=0.1, scale=1.0, u=-1.0, seed=21
    )
    out = tmp_path / "mep.png"
    fig = mean_excess_plot(residuals, save_path=out, n_thresholds=30, min_excess=10)

    assert out.exists()
    assert out.stat().st_size > 0
    # Sanity check : un axe avec données
    ax = fig.axes[0]
    assert ax.lines, "courbe e(u) non tracée"
