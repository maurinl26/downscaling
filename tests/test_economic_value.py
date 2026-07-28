"""Tests de la Valeur Économique Relative (REV) — issue #230.

Propriétés vérifiées :
- prévision parfaite (pred == obs) → V_env ≈ 1 sur la plage utile ;
- climatologie (pred constant) → V_env ≈ 0 partout ;
- ancrage théorique : V_env_at_s ≈ peirce_at_s (POD − POFD au τ optimal, α=s) ;
- V ≤ 1 toujours, robustesse aux cas dégénérés (n=0, pas de gel, NaN).
"""

from __future__ import annotations

import numpy as np
import pytest

from downscaling.scripts.economic_value import relative_economic_value


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _synthetic(rng, n=2000, base_rate=0.08, noise=1.0, thr=-2.2):
    """Génère obs (Tmin) avec un base rate gel ~ donné, et pred = obs + bruit."""
    # Tmin centré au-dessus du seuil, queue froide pour créer des gels.
    obs = rng.normal(loc=2.0, scale=3.5, size=n)
    # Ajuste grossièrement la fréquence de gel visée en décalant la moyenne.
    for _ in range(50):
        s = np.mean(obs <= thr)
        if abs(s - base_rate) < 0.005:
            break
        obs += 0.1 if s > base_rate else -0.1
    pred = obs + rng.normal(0.0, noise, size=n)
    return obs, pred


def test_perfect_forecast_V_env_near_one(rng):
    """pred == obs → V_env ≈ 1 sur la plage utile."""
    obs, pred = _synthetic(rng, noise=0.0)
    res = relative_economic_value(obs, pred)
    assert not res["degenerate"]
    V_env = np.array([v for v in res["V_env"]], dtype=float)
    alphas = np.array(res["alphas"])
    # Plage utile : α autour du base rate, à l'écart des bords dégénérés.
    s = res["base_rate"]
    useful = (alphas > s / 3) & (alphas < min(0.4, 3 * s))
    assert useful.any()
    assert np.nanmin(V_env[useful]) > 0.98
    assert res["V_max"] == pytest.approx(1.0, abs=1e-6)


def test_climatology_forecast_V_env_near_zero(rng):
    """pred constant (aucune information) → V_env ≈ 0 partout."""
    obs, _ = _synthetic(rng)
    pred = np.full_like(obs, 5.0)  # constant → jamais de séparation
    res = relative_economic_value(obs, pred)
    assert not res["degenerate"]
    V_env = np.array(res["V_env"], dtype=float)
    assert np.nanmax(np.abs(V_env)) < 1e-6
    assert res["alpha_positive_range"] is None


def test_anchor_V_env_at_s_equals_peirce(rng):
    """Ancrage : à α = s et η = 1, V_env(s) = max_τ (POD − POFD)."""
    obs, pred = _synthetic(rng, noise=1.5)
    res = relative_economic_value(obs, pred)
    assert res["V_env_at_s"] == pytest.approx(res["peirce_at_s"], abs=1e-9)


def test_V_never_exceeds_one(rng):
    """V_env ≤ 1 pour toute configuration (borne théorique)."""
    for noise in (0.0, 0.5, 1.0, 2.0, 5.0):
        obs, pred = _synthetic(rng, noise=noise)
        res = relative_economic_value(obs, pred)
        V_env = np.array([v for v in res["V_env"] if v is not None], dtype=float)
        assert np.all(V_env <= 1.0 + 1e-9)
        # Enveloppe ≥ 0 : on peut toujours retomber sur la climatologie.
        assert np.all(V_env >= -1e-9)


def test_euros_per_ha_grid(rng):
    """La grille (C, L) produit des €/ha finis et cohérents (perfect → gros gains)."""
    obs, pred = _synthetic(rng, noise=0.0)
    res = relative_economic_value(obs, pred)
    assert "euros_per_ha" in res
    assert len(res["euros_per_ha"]) == 9  # 3 C × 3 L
    for row in res["euros_per_ha"]:
        assert np.isfinite(row["euros_per_ha"])
        assert row["euros_per_ha"] >= -1e-6  # enveloppe ≥ climatologie
        assert row["alpha"] == pytest.approx(row["C"] / row["L"])


def test_degenerate_empty():
    """n = 0 → dict bien formé, pas de crash."""
    res = relative_economic_value(np.array([]), np.array([]))
    assert res["degenerate"]
    assert res["V_max"] is None
    assert all(v is None for v in res["V_env"])


def test_degenerate_no_frost():
    """Aucun gel (s = 0) → dégénéré, base_rate = 0."""
    obs = np.full(100, 5.0)
    pred = np.full(100, 5.0)
    res = relative_economic_value(obs, pred, threshold_c=-2.2)
    assert res["degenerate"]
    assert res["base_rate"] == pytest.approx(0.0)


def test_nan_robustness(rng):
    """Les paires NaN sont ignorées sans fausser le résultat."""
    obs, pred = _synthetic(rng, noise=1.0)
    obs2 = obs.copy()
    pred2 = pred.copy()
    obs2[::10] = np.nan
    pred2[5::10] = np.nan
    res = relative_economic_value(obs2, pred2)
    assert not res["degenerate"]
    assert res["n_pairs"] < obs.size
    V_env = np.array([v for v in res["V_env"] if v is not None], dtype=float)
    assert np.all(V_env <= 1.0 + 1e-9)


def test_eta_partial_protection_valid(rng):
    """η < 1 (protection partielle) reste bien défini et borné (V ≤ 1)."""
    obs, pred = _synthetic(rng, noise=1.0)
    partial = relative_economic_value(obs, pred, eta=0.6)
    assert not partial["degenerate"]
    assert partial["eta"] == pytest.approx(0.6)
    V_env = np.array([v for v in partial["V_env"] if v is not None], dtype=float)
    assert np.all(V_env <= 1.0 + 1e-9)
    assert np.all(V_env >= -1e-9)


def test_invalid_eta_raises(rng):
    """η hors ]0, 1] → ValueError."""
    obs, pred = _synthetic(rng, noise=1.0)
    with pytest.raises(ValueError):
        relative_economic_value(obs, pred, eta=0.0)
    with pytest.raises(ValueError):
        relative_economic_value(obs, pred, eta=1.5)
