#!/usr/bin/env python3
"""Valeur Économique Relative (REV) — modèle coût-perte pour l'alerte gel (#230).

Transforme une table de contingence produit-vs-Sencrop en **euros** et en
**enveloppe de valeur** ``V(α)``, sans donnée nouvelle au-delà de ce dont on
dispose déjà (obs Sencrop et prédiction downscalée, °C). Cadre standard de la
value-of-information météo (Richardson 2000 ; Wilks), lentille A de la méthodo
« Valeur économique des prévisions gel (REV + risque de base) ».

Modèle de décision
------------------
À chaque nuit à risque, le décideur choisit **protéger** (coût ``C`` : aspersion,
tour, bougies…) ou **ne pas protéger** (perte ``L`` de récolte si le gel
survient). On normalise par ``L`` (perte = 1) ; le seul paramètre économique est
le ratio coût-perte ``α = C/L``. On introduit l'efficacité de protection
``η ∈ ]0, 1]`` : une protection déclenchée n'évite qu'une fraction ``η`` de la
perte (le gel advectif / les épisodes répétés ne sont couverts que partiellement).

Convention gel : événement réel = ``obs <= threshold_c``. Décision « protéger »
= ``pred <= τ`` où ``τ`` est le seuil de décision balayé sur les valeurs de
``pred``. Pour chaque ``τ`` on a la table de contingence :

- ``a`` = hits    (``pred<=τ & obs<=thr``)
- ``b`` = fausses alertes (``pred<=τ & obs>thr``)
- ``c`` = ratés   (``pred>τ  & obs<=thr``)
- ``d`` = corrects négatifs (``pred>τ & obs>thr``)
- ``N = a+b+c+d`` ; base rate ``s = (a+c)/N`` (constant en τ).

Dépenses espérées (unités de ``L``)
-----------------------------------
Coût par nuit selon la décision et la réalité :

- protéger + gel      → ``α + (1-η)``  (coût de protection + perte résiduelle)
- protéger + pas gel  → ``α``
- ne pas protéger + gel     → ``1``    (perte pleine)
- ne pas protéger + pas gel → ``0``

d'où, en agrégeant sur la table :

    E_f   = [α·(a+b) + (1-η)·a + c] / N          (prévision Karpos)
    E_clim = min( α + (1-η)·s , s )              (climatologie : toujours / jamais)
    E_perf = s·(α + 1-η)                         (prévision parfaite : protège ssi gel)

Cas ``η = 1`` (protection parfaite) — la forme classique :

    E_f    = [α·(a+b) + c] / N
    E_clim = min(α, s)
    E_perf = s·α

La métrique
-----------
    V(α, τ) = (E_clim - E_f) / (E_clim - E_perf)   ∈ ]-∞, 1]
    V_env(α) = max_τ  V(α, τ)                       (enveloppe de valeur)

``V = 1`` : vaut autant que la prévision parfaite. ``V = 0`` : ne vaut pas mieux
que la climatologie. L'enveloppe inclut les stratégies triviales (protéger
jamais / toujours), donc ``V_env(α) ≥ 0`` : on peut toujours ignorer la
prévision et retomber sur la climatologie.

**Ancrage théorique** : au point ``α = s`` et ``η = 1``, on a exactement
``V(τ) = POD(τ) − POFD(τ)`` pour tout τ, donc ``V_env(s) = max_τ (POD − POFD)``
= score de Peirce / Hanssen-Kuipers. La REV intègre tous les seuils de décision,
ce qui règle la limite « seuil fixe −2,2 °C » du suivi POD/FAR/CSI actuel.

Rigueur (cf. méthodo §Lentille A) : alimenter cette fonction avec des
contingences **LOO station-out** (jamais in-sample), et tracer sur la **grille
(C, L)** plutôt qu'un point unique (``C`` et ``L`` sont des fourchettes).
"""

from __future__ import annotations

import numpy as np


def _finite_pairs(obs: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Garde uniquement les paires (obs, pred) toutes deux finies."""
    obs = np.asarray(obs, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if obs.shape != pred.shape:
        raise ValueError(f"obs et pred de tailles différentes : {obs.shape} vs {pred.shape}")
    mask = np.isfinite(obs) & np.isfinite(pred)
    return obs[mask], pred[mask]


def _operating_points(
    obs: np.ndarray, pred: np.ndarray, threshold_c: float
) -> dict[str, np.ndarray]:
    """Tables de contingence (a, b, c, d) le long du balayage de τ sur pred.

    On balaye ``τ`` sur les valeurs distinctes de ``pred`` (protéger = ``pred<=τ``)
    en gérant les ex-æquo (une valeur distincte protège tout son groupe), et on
    ajoute la stratégie triviale « ne jamais protéger » (τ = −∞). La stratégie
    « toujours protéger » est le plus grand τ distinct. Retourne des vecteurs
    alignés ``a, b, c, d, tau`` (un opérateur par colonne).
    """
    frost = (obs <= threshold_c).astype(float)
    n1 = float(frost.sum())  # nombre d'événements gel
    N = int(obs.size)

    order = np.argsort(pred, kind="mergesort")
    pred_s = pred[order]
    frost_s = frost[order]
    cum_frost = np.cumsum(frost_s)  # gels parmi les k plus petits pred

    # Indice de dernière occurrence de chaque valeur distincte de pred (gère les ex-æquo).
    uniq, first_idx = np.unique(pred_s, return_index=True)
    last_idx = np.append(first_idx[1:] - 1, N - 1)

    a = cum_frost[last_idx]  # hits parmi les protégés
    prot = last_idx + 1.0  # nombre de protégés (pred<=τ)
    b = prot - a  # fausses alertes
    c = n1 - a  # ratés
    d = (N - prot) - c  # corrects négatifs
    tau = uniq.astype(float)

    # Stratégie triviale « ne jamais protéger » (τ = −∞) : a=b=0, c=n1, d=N-n1.
    a = np.concatenate([[0.0], a])
    b = np.concatenate([[0.0], b])
    c = np.concatenate([[n1], c])
    d = np.concatenate([[float(N) - n1], d])
    tau = np.concatenate([[-np.inf], tau])

    return {"a": a, "b": b, "c": c, "d": d, "tau": tau, "N": float(N), "n1": n1}


def _rev_over_alphas(
    ops: dict[str, np.ndarray], alphas: np.ndarray, eta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule V(α, τ) pour tous les opérateurs et renvoie l'enveloppe.

    Retourne ``(V_env, tau_star, arg_star)`` : enveloppe max_τ V par α, le τ
    atteignant le max, et l'indice de l'opérateur optimal (dans ``ops``).
    """
    a, b, c = ops["a"], ops["b"], ops["c"]
    N, n1 = ops["N"], ops["n1"]
    s = n1 / N

    alphas = np.asarray(alphas, dtype=float)
    # E_f : (A, T)
    E_f = (alphas[:, None] * (a + b)[None, :] + (1.0 - eta) * a[None, :] + c[None, :]) / N
    # E_clim, E_perf : (A,)
    E_clim = np.minimum(alphas + (1.0 - eta) * s, s)
    E_perf = s * (alphas + 1.0 - eta)
    denom = E_clim - E_perf  # (A,)

    with np.errstate(divide="ignore", invalid="ignore"):
        V = (E_clim[:, None] - E_f) / denom[:, None]
    # Dénominateur ~ 0 (α → limites dégénérées) : V indéfini → -inf pour l'argmax.
    bad = np.abs(denom) < 1e-12
    V[bad, :] = -np.inf

    arg_star = np.argmax(V, axis=1)
    V_env = V[np.arange(V.shape[0]), arg_star]
    tau_star = ops["tau"][arg_star]
    # Là où le dénom est dégénéré, l'enveloppe n'a pas de sens → NaN.
    V_env = np.where(bad, np.nan, V_env)
    return V_env, tau_star, arg_star


def _degenerate_result(alphas: np.ndarray, base_rate: float, reason: str) -> dict:
    """Dict bien formé quand la REV est indéfinie (n=0, pas de gel, tout gel)."""
    A = int(np.asarray(alphas).size)
    return {
        "alphas": [float(x) for x in np.asarray(alphas)],
        "V_env": [None] * A,
        "tau_star": [None] * A,
        "alpha_positive_range": None,
        "V_max": None,
        "alpha_at_peak": None,
        "base_rate": float(base_rate),
        "peirce_at_s": None,
        "V_env_at_s": None,
        "degenerate": True,
        "reason": reason,
        "eta": None,
    }


def relative_economic_value(
    obs,
    pred,
    threshold_c: float = -2.2,
    alphas=None,
    cost_grid: dict | None = None,
    eta: float = 1.0,
) -> dict:
    """Valeur Économique Relative (REV) coût-perte de l'alerte gel.

    Parameters
    ----------
    obs, pred : array-like
        Tmin Sencrop observée et prédite (°C), appariées station-nuit. Les paires
        non finies sont ignorées.
    threshold_c : float
        Seuil de gel (événement réel = ``obs <= threshold_c``). Défaut −2,2 °C
        (BBCH floraison abricot).
    alphas : array-like, optional
        Grille du ratio coût-perte ``α = C/L``. Défaut
        ``np.logspace(-3, log10(0.5), 60)``.
    cost_grid : dict, optional
        ``{"C": [...], "L": [...]}`` en €/ha (coût de protection, valeur de
        récolte à risque). Défaut C∈{600,900,2500}, L∈{15000,25000,40000}. Active
        la sortie ``euros_per_ha``.
    eta : float
        Efficacité de protection ``η ∈ ]0, 1]`` (fraction de perte évitée quand on
        protège). Défaut 1.0 (protection parfaite, forme classique de la REV).

    Returns
    -------
    dict JSON-sérialisable — voir docstring du module pour la méthode. Clés :
    ``alphas``, ``V_env``, ``tau_star`` (par α), ``alpha_positive_range``,
    ``V_max``, ``alpha_at_peak``, ``base_rate``, ``peirce_at_s`` (POD−POFD au τ
    optimal pour α=s), ``V_env_at_s`` (ancrage : ≈ ``peirce_at_s`` pour η=1), et,
    si ``cost_grid``, ``euros_per_ha`` = liste de ``{C, L, alpha, euros_per_ha,
    tau_star}`` avec ``euros = (E_clim − E_f_opt)·L`` au τ optimal pour α=C/L.
    """
    if alphas is None:
        alphas = np.logspace(-3, np.log10(0.5), 60)
    alphas = np.asarray(alphas, dtype=float)
    if cost_grid is None:
        cost_grid = {"C": [600.0, 900.0, 2500.0], "L": [15000.0, 25000.0, 40000.0]}
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta doit être dans ]0, 1], reçu {eta}")

    obs, pred = _finite_pairs(obs, pred)
    N = int(obs.size)
    if N == 0:
        return _degenerate_result(alphas, base_rate=float("nan"), reason="no finite pairs")

    frost = obs <= threshold_c
    n1 = int(frost.sum())
    base_rate = n1 / N
    if n1 == 0:
        return _degenerate_result(alphas, base_rate, reason="no frost events (s=0)")
    if n1 == N:
        return _degenerate_result(alphas, base_rate, reason="all frost (s=1)")

    ops = _operating_points(obs, pred, threshold_c)
    s = base_rate

    V_env, tau_star, _ = _rev_over_alphas(ops, alphas, eta)

    # Enveloppe et pic.
    finite = np.isfinite(V_env)
    if finite.any():
        i_peak = int(np.nanargmax(np.where(finite, V_env, -np.inf)))
        V_max = float(V_env[i_peak])
        alpha_at_peak = float(alphas[i_peak])
    else:
        V_max = None
        alpha_at_peak = None

    # Plage de α où V_env > 0 (là où le produit vaut de l'argent).
    pos = finite & (V_env > 1e-9)
    if pos.any():
        alpha_positive_range = [float(alphas[pos].min()), float(alphas[pos].max())]
    else:
        alpha_positive_range = None

    # Ancrage : α = s exactement (calculé hors grille).
    V_env_s, tau_s, arg_s = _rev_over_alphas(ops, np.array([s]), eta)
    V_env_at_s = float(V_env_s[0]) if np.isfinite(V_env_s[0]) else None
    a_s, b_s, c_s, d_s = (
        ops["a"][arg_s[0]],
        ops["b"][arg_s[0]],
        ops["c"][arg_s[0]],
        ops["d"][arg_s[0]],
    )
    pod = a_s / (a_s + c_s) if (a_s + c_s) > 0 else float("nan")
    pofd = b_s / (b_s + d_s) if (b_s + d_s) > 0 else float("nan")
    peirce_at_s = float(pod - pofd)

    result = {
        "alphas": [float(x) for x in alphas],
        "V_env": [float(v) if np.isfinite(v) else None for v in V_env],
        "tau_star": [float(t) if np.isfinite(t) else None for t in tau_star],
        "alpha_positive_range": alpha_positive_range,
        "V_max": V_max,
        "alpha_at_peak": alpha_at_peak,
        "base_rate": float(base_rate),
        "peirce_at_s": peirce_at_s,
        "V_env_at_s": V_env_at_s,
        "n_pairs": N,
        "n_frost": n1,
        "threshold_c": float(threshold_c),
        "eta": float(eta),
        "degenerate": False,
    }

    # €/ha évités sur la grille (C, L) : au τ optimal pour α = C/L.
    if cost_grid is not None:
        Cs = [float(x) for x in cost_grid.get("C", [])]
        Ls = [float(x) for x in cost_grid.get("L", [])]
        euros: list[dict] = []
        for L in Ls:
            for C in Cs:
                alpha = C / L
                V_a, tau_a, arg_a = _rev_over_alphas(ops, np.array([alpha]), eta)
                k = arg_a[0]
                a_k, b_k, c_k = ops["a"][k], ops["b"][k], ops["c"][k]
                E_f = (alpha * (a_k + b_k) + (1.0 - eta) * a_k + c_k) / ops["N"]
                E_clim = min(alpha + (1.0 - eta) * s, s)
                saved = (E_clim - E_f) * L  # €/ha·nuit évités vs climatologie
                euros.append(
                    {
                        "C": C,
                        "L": L,
                        "alpha": float(alpha),
                        "euros_per_ha": float(saved),
                        "V": float(V_a[0]) if np.isfinite(V_a[0]) else None,
                        "tau_star": float(tau_a[0]) if np.isfinite(tau_a[0]) else None,
                    }
                )
        result["euros_per_ha"] = euros

    return result


__all__ = ["relative_economic_value"]
