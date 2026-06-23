"""POT/GPD sur résidus QDM, queue gauche froide.

Référence
---------
Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*,
Springer Series in Statistics, chapitres 4 (Peaks-Over-Threshold, threshold
selection) et 5 (inférence GPD par maximum de vraisemblance).

Contexte Karpos
---------------
Le QDM (Cannon et al. 2015, cf. `downscaling.statistical.quantile_mapping`) ajuste
la distribution marginale de Tn par appariement de quantiles empiriques. Hors de
la plage de calibration, l'extrapolation linéaire de la fonction de transfert
revient implicitement à supposer une queue gaussienne. Pour défendre un produit
paramétrique gel devant un actuaire, on borne ce biais de queue par un fit GPD
sur les excès en dessous d'un seuil ``u`` (POT) et on compare le quantile
extrême gaussien au quantile extrême GPD.

Conventions de signe
--------------------
Les fonctions de ce module travaillent sur les **résidus négatifs** ``eps_i =
Tn_obs - Tn_QDM`` lorsque l'observation est plus froide que la prédiction
(``eps_i < 0``). Pour appliquer la théorie POT classique (excès au-dessus d'un
seuil haut), on étudie la variable transformée ``y_i = -eps_i`` et on cherche
les excès au-dessus d'un seuil ``u_pos = -u`` avec ``u < 0``. Toutes les
fonctions publiques exposent le seuil ``threshold`` dans l'espace original des
résidus (donc ``threshold < 0`` est attendu) et renvoient les quantiles
également dans l'espace original (donc négatifs, plus le froid est extrême,
plus la valeur est basse).

Statut
------
Squelette posé pour la sub-issue karpos-downscaling#67. Pas encore branché sur
un vrai run QDM Sencrop Baronnies : les premières exécutions sur données
réelles arrivent dans le PR suivant (cf. ``reports/drome/data/qdm_tail_decision.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Fit POT/GPD
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPDFit:
    """Résultat d'un fit POT/GPD sur les résidus QDM.

    Tous les champs sont exprimés dans l'espace **transformé** ``y = -eps``,
    seule représentation où ``threshold`` joue le rôle d'un seuil haut au sens
    de Coles 2001 (équation 4.7).

    Attributes
    ----------
    threshold:
        Seuil ``u`` dans l'espace des résidus originaux (typiquement < 0).
    scale:
        Paramètre d'échelle ``sigma`` de la GPD ajustée sur les excès.
    shape:
        Paramètre de forme ``xi`` de la GPD. ``xi > 0`` indique une queue
        lourde (Pareto), ``xi = 0`` une queue exponentielle, ``xi < 0`` une
        queue bornée.
    n_excess:
        Nombre d'observations strictement en dessous de ``threshold`` (= au
        dessus de ``-threshold`` dans l'espace transformé), utilisées pour
        l'estimation MLE.
    n_total:
        Effectif total des résidus passés au fit (utile pour la probabilité
        d'excès ``zeta_u = n_excess / n_total``, cf. Coles eq. 4.12).
    mle_loglik:
        Log-vraisemblance MLE de la GPD au point ``(scale, shape)``.
    ci95_scale:
        Bornes inférieure et supérieure de l'IC à 95 % du paramètre d'échelle
        (bootstrap non-paramétrique).
    ci95_shape:
        Idem pour le paramètre de forme.
    n_bootstrap:
        Nombre de répliques bootstrap effectivement utilisées (peut être
        inférieur au nombre demandé si certaines répliques ne convergent pas).
    """

    threshold: float
    scale: float
    shape: float
    n_excess: int
    n_total: int
    mle_loglik: float
    ci95_scale: tuple[float, float]
    ci95_shape: tuple[float, float]
    n_bootstrap: int


def _excesses(residuals: np.ndarray, threshold: float) -> np.ndarray:
    """Retourne les excès ``y_i - u_pos`` (positifs) prêts pour ``genpareto.fit``.

    Travaille dans l'espace transformé ``y = -residuals`` afin de se ramener
    au cas standard "excès au-dessus d'un seuil haut" de Coles 2001 eq. 4.7.
    """
    y = -np.asarray(residuals, dtype=float)
    u_pos = -float(threshold)
    mask = np.isfinite(y) & (y > u_pos)
    return y[mask] - u_pos


def fit_gpd_pot(
    residuals: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> GPDFit:
    """Fit POT/GPD sur les résidus QDM en queue gauche froide.

    Parameters
    ----------
    residuals:
        Résidus ``eps_i = Tn_obs - Tn_QDM`` (deg C). Les valeurs ``NaN`` sont
        ignorées.
    threshold:
        Seuil ``u`` (deg C), typiquement négatif. Seules les observations
        ``eps_i < threshold`` contribuent à l'estimation.
    n_bootstrap:
        Nombre de répliques bootstrap non-paramétrique pour les IC95 sur
        ``(scale, shape)``. Mettre à 0 désactive le bootstrap (IC = NaN).
    random_state:
        Graine ou ``np.random.Generator`` pour le bootstrap (reproductibilité).

    Returns
    -------
    GPDFit
        Paramètres MLE et IC95 bootstrap dans l'espace transformé.

    Notes
    -----
    L'estimation MLE utilise ``scipy.stats.genpareto.fit`` avec la
    paramétrisation ``floc=0`` (origine des excès en 0, conforme à Coles eq.
    4.7). Le paramètre de forme ``c`` de SciPy est égal au ``xi`` de Coles.

    Le bootstrap est non-paramétrique avec rééchantillonnage des excès. Les
    répliques pour lesquelles ``genpareto.fit`` ne converge pas (rare) sont
    silencieusement écartées.
    """
    excesses = _excesses(residuals, threshold)
    n_excess = int(excesses.size)
    n_total = int(np.isfinite(residuals).sum())

    if n_excess < 10:
        raise ValueError(
            f"Trop peu d'excès pour un fit MLE GPD stable (n_excess={n_excess} < 10). "
            "Abaisser ``threshold`` (le rendre plus proche de zéro) ou collecter "
            "davantage de résidus."
        )

    # MLE GPD : floc=0 car les excès sont centrés en zéro par construction.
    shape, _, scale = stats.genpareto.fit(excesses, floc=0.0)
    loglik = float(np.sum(stats.genpareto.logpdf(excesses, shape, loc=0.0, scale=scale)))

    # Bootstrap non-paramétrique sur les excès.
    rng = np.random.default_rng(random_state)
    scales: list[float] = []
    shapes: list[float] = []
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(excesses, size=n_excess, replace=True)
        try:
            xi_b, _, sigma_b = stats.genpareto.fit(sample, floc=0.0)
        except Exception:  # noqa: BLE001 — convergence MLE non garantie sur réplique dégénérée
            continue
        if not (np.isfinite(xi_b) and np.isfinite(sigma_b) and sigma_b > 0):
            continue
        scales.append(float(sigma_b))
        shapes.append(float(xi_b))

    if scales:
        ci95_scale = (float(np.percentile(scales, 2.5)), float(np.percentile(scales, 97.5)))
        ci95_shape = (float(np.percentile(shapes, 2.5)), float(np.percentile(shapes, 97.5)))
    else:
        ci95_scale = (float("nan"), float("nan"))
        ci95_shape = (float("nan"), float("nan"))

    return GPDFit(
        threshold=float(threshold),
        scale=float(scale),
        shape=float(shape),
        n_excess=n_excess,
        n_total=n_total,
        mle_loglik=loglik,
        ci95_scale=ci95_scale,
        ci95_shape=ci95_shape,
        n_bootstrap=len(scales),
    )


# ---------------------------------------------------------------------------
# Quantiles extrêmes
# ---------------------------------------------------------------------------


def quantile_gpd(
    fit: GPDFit,
    exceedance_prob: float,
    alpha: float = 0.95,
) -> dict[str, float]:
    """Quantile extrême gauche par POT/GPD (Coles 2001 eq. 4.12).

    Parameters
    ----------
    fit:
        Résultat de ``fit_gpd_pot``.
    exceedance_prob:
        Probabilité marginale d'excès cible ``p`` dans l'espace transformé
        ``y = -eps`` : on cherche la valeur ``z_p`` telle que ``P(Y > z_p) = p``,
        ce qui correspond, dans l'espace des résidus originaux, à ``P(eps < -z_p) = p``.
        Doit vérifier ``0 < p < zeta_u`` où ``zeta_u = n_excess / n_total`` est
        la probabilité empirique de dépasser le seuil.
    alpha:
        Niveau nominal de l'IC (défaut 0.95). L'IC est dérivé du bootstrap sur
        ``(scale, shape)`` (méthode delta non utilisée ici).

    Returns
    -------
    dict
        Clés :

        - ``quantile`` (float) — quantile dans l'espace **original** des
          résidus (négatif quand on tape la queue froide).
        - ``ci95_low`` (float) — borne basse de l'IC (= résidu le plus froid).
        - ``ci95_high`` (float) — borne haute de l'IC (= résidu le moins froid).

    Notes
    -----
    Formule POT (Coles eq. 4.12, transposée à un seuil haut sur ``y = -eps``) :

        z_p = u_pos + (sigma / xi) * ( (p / zeta_u)^(-xi) - 1 )   si xi != 0
        z_p = u_pos + sigma * log(zeta_u / p)                     si xi  = 0

    avec ``u_pos = -threshold``, ``zeta_u = n_excess / n_total``. Le quantile
    dans l'espace original des résidus est ensuite ``q = -z_p``.

    Cette fonction renvoie un IC bootstrap-naïf : elle calcule l'intervalle
    [percentile 2.5 %, percentile 97.5 %] de ``q`` propagé via les bornes IC95
    de ``scale`` et ``shape`` stockées dans ``fit``. Ce n'est qu'un proxy ; un
    bootstrap complet sur le quantile lui-même est une amélioration prévue.
    """
    if not 0.0 < exceedance_prob < 1.0:
        raise ValueError("exceedance_prob doit être dans (0, 1).")
    zeta_u = fit.n_excess / fit.n_total if fit.n_total else 0.0
    if zeta_u <= 0.0:
        raise ValueError("zeta_u nul : le fit ne contient aucun excès.")
    if exceedance_prob >= zeta_u:
        raise ValueError(
            f"exceedance_prob={exceedance_prob:.4f} >= zeta_u={zeta_u:.4f} : "
            "la probabilité cible est plus fréquente que le seuil, le quantile "
            "POT n'est pas extrapolable. Abaisser threshold ou augmenter "
            "exceedance_prob."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha doit être dans (0, 1).")

    u_pos = -fit.threshold

    def _z(sigma: float, xi: float) -> float:
        ratio = exceedance_prob / zeta_u
        if abs(xi) < 1e-8:
            return u_pos + sigma * float(np.log(1.0 / ratio))
        return u_pos + (sigma / xi) * (ratio ** (-xi) - 1.0)

    z_point = _z(fit.scale, fit.shape)
    quantile = -z_point  # retour dans l'espace original des résidus

    # IC95 proxy : on évalue z aux 4 coins (scale_lo/hi x shape_lo/hi) et on
    # prend min/max. Cohérent tant que les IC bootstrap de scale et shape sont
    # finis, sinon NaN.
    if all(np.isfinite([*fit.ci95_scale, *fit.ci95_shape])):
        corners = [
            _z(s, xi)
            for s in fit.ci95_scale
            for xi in fit.ci95_shape
        ]
        z_lo, z_hi = min(corners), max(corners)
        ci95_low = -z_hi  # plus z est grand, plus le résidu original est froid (négatif)
        ci95_high = -z_lo
    else:
        ci95_low = float("nan")
        ci95_high = float("nan")

    return {
        "quantile": float(quantile),
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
    }


# ---------------------------------------------------------------------------
# Diagnostic : Mean Excess Plot
# ---------------------------------------------------------------------------


def mean_excess_plot(
    residuals: np.ndarray,
    save_path: str | Path,
    n_thresholds: int = 50,
    min_excess: int = 10,
) -> Figure:
    """Mean Excess Plot pour le choix du seuil ``u`` (Coles 2001 §4.3.1).

    L'idée : si la GPD est appropriée au-dessus de ``u``, alors la fonction
    ``e(u) = E[Y - u | Y > u]`` est linéaire en ``u`` au-dessus du vrai seuil
    asymptotique. On choisit visuellement ``u`` à partir duquel la courbe
    devient linéaire.

    Parameters
    ----------
    residuals:
        Résidus QDM ``eps_i = Tn_obs - Tn_QDM`` (deg C). Travaille
        automatiquement dans l'espace transformé ``y = -residuals``.
    save_path:
        Chemin de sortie de la figure (extension PNG/PDF/SVG reconnue par
        matplotlib).
    n_thresholds:
        Nombre de seuils candidats balayés entre le minimum et le
        ``(1 - min_excess/n)``-quantile des ``y_i``.
    min_excess:
        Seuil minimal d'excès au-delà duquel la moyenne empirique est
        considérée comme stable.

    Returns
    -------
    matplotlib.figure.Figure
        Figure sauvegardée et retournée (pour usage interactif/test).

    Notes
    -----
    L'axe X est gradué dans l'espace des résidus originaux (négatif). Plus on
    va à gauche, plus le seuil est froid.
    """
    # Import paresseux : matplotlib est dans l'extra ``viz`` (cf. pyproject.toml).
    import matplotlib.pyplot as plt

    y = -np.asarray(residuals, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < min_excess + 5:
        raise ValueError(
            f"Pas assez de résidus finis (n={y.size}) pour balayer {min_excess} excès min."
        )

    y_sorted = np.sort(y)
    # Borne haute : on laisse au moins ``min_excess`` points strictement au-dessus.
    u_max = y_sorted[-min_excess - 1]
    u_min = y_sorted[0]
    grid = np.linspace(u_min, u_max, int(n_thresholds))

    means: list[float] = []
    stderr: list[float] = []
    for u_pos in grid:
        excess = y[y > u_pos] - u_pos
        if excess.size < min_excess:
            means.append(np.nan)
            stderr.append(np.nan)
            continue
        means.append(float(np.mean(excess)))
        stderr.append(float(np.std(excess, ddof=1) / np.sqrt(excess.size)))

    means_arr = np.asarray(means)
    stderr_arr = np.asarray(stderr)
    u_orig = -grid  # affichage dans l'espace original des résidus

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(u_orig, means_arr, color="C0", lw=1.6, label="e(u) = E[Y - u | Y > u]")
    ax.fill_between(
        u_orig,
        means_arr - 1.96 * stderr_arr,
        means_arr + 1.96 * stderr_arr,
        color="C0",
        alpha=0.2,
        label="IC95 (normal approx)",
    )
    ax.set_xlabel("Seuil u (deg C, espace résidus originaux)")
    ax.set_ylabel("Moyenne des excès (deg C)")
    ax.set_title("Mean Excess Plot — Coles 2001 §4.3.1")
    ax.invert_xaxis()  # on lit de la queue froide (gauche) vers le corps
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Test d'adéquation : Kolmogorov-Smirnov sur la queue
# ---------------------------------------------------------------------------


def ks_test_gpd(
    residuals: np.ndarray,
    threshold: float,
    fit: GPDFit,
) -> tuple[float, float]:
    """Test KS d'adéquation des excès à la GPD ajustée.

    Parameters
    ----------
    residuals:
        Résidus QDM (mêmes que ceux passés à ``fit_gpd_pot``).
    threshold:
        Même seuil que celui passé à ``fit_gpd_pot``.
    fit:
        Fit GPD à tester.

    Returns
    -------
    (D, p_value)
        Statistique de Kolmogorov-Smirnov et p-valeur asymptotique. ``H0`` : les
        excès sont issus de ``GPD(scale=fit.scale, shape=fit.shape)``. Si
        ``p_value < 0.05``, on rejette l'adéquation au seuil 5 % et la queue
        n'est pas correctement modélisée par cette GPD (revoir le choix de
        ``threshold`` ou stratifier le fit).

    Notes
    -----
    La p-valeur de ``scipy.stats.kstest`` est asymptotique et **non corrigée**
    pour le fait que ``(scale, shape)`` ont été estimés sur les mêmes données :
    elle est donc anti-conservatrice. Pour un test rigoureux, utiliser un
    bootstrap paramétrique (prévu dans une itération ultérieure).
    """
    excesses = _excesses(residuals, threshold)
    if excesses.size < 10:
        raise ValueError(
            f"Trop peu d'excès pour KS (n={excesses.size} < 10)."
        )
    cdf = lambda x: stats.genpareto.cdf(x, fit.shape, loc=0.0, scale=fit.scale)  # noqa: E731
    result = stats.kstest(excesses, cdf)
    return float(result.statistic), float(result.pvalue)
