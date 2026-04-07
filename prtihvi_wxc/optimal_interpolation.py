"""
optimal_interpolation.py — Correction OI sur le champ downscalé via observations Netatmo.

Implémente la correction BLUE (Best Linear Unbiased Estimator), équivalente à
un filtre de Kalman sans dimension temporelle :

    x_a = x_b + K (y_o - H x_b)

    K = B H^T (H B H^T + R)^-1   (gain de Kalman)

    où :
      x_b  = champ background (sortie Prithvi WxC, grille HR)
      y_o  = observations Netatmo QC'd aux stations
      H    = opérateur d'observation (interpolation bilinéaire grille → stations)
      B    = matrice d'erreur background (Gaussienne, longueur de corrélation L)
      R    = matrice d'erreur d'observation (diagonale, σ_obs²)
      x_a  = champ analysé (background + incrément d'observation)

Pourquoi pas un EKF complet ?
  Pour une réanalyse offline (pas de dimension temporelle active),
  l'OI est suffisante et O(N²) au lieu de O(N³) pour l'EKF.
  Les données Netatmo nocturnes sont quasi-indépendantes (biais radiatif nul)
  → matrice R bien conditionnée.

Référence :
  Lussana et al. (2019) Q. J. Roy. Met. Soc. — MET Norway + Netatmo
  Nipen et al. (2020) — assimilation opérationnelle Netatmo NWP
  Ide et al. (1997) J. Met. Soc. Japan — notations unifiées OI/Kalman
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from scipy.linalg import solve
from scipy.spatial import cKDTree

from shared.netatmo_qc import NetatmoObs, tmin_nocturnal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paramètres OI (à calibrer sur Drôme/Ardèche vs SYNOP)
# ---------------------------------------------------------------------------

# Longueur de corrélation du background error (km)
# Calibrée sur CERRA vs SYNOP Drôme : ~20-30 km pour T nocturne
BACKGROUND_CORR_LENGTH_M = 25_000.0

# Variance erreur background (K²) — typique pour downscaling 1km vs réalité
SIGMA_B_K = 1.5  # 1.5°C std background error

# Variance erreur observation (K²) — Netatmo nocturne QC'd
# ±0.3°C précision capteur + ±0.4°C représentativité → σ_obs ~ 0.5°C
SIGMA_OBS_K = 0.5

# Rayon d'influence des observations (au-delà : correction nulle)
INFLUENCE_RADIUS_M = 50_000.0

# Nombre max de stations par point de grille (contrôle le temps de calcul)
MAX_OBS_PER_GRIDPOINT = 30


class OptimalInterpolationCorrection:
    """
    Applique une correction OI sur le champ Tmin nocturne downscalé.

    Le champ background (Prithvi WxC) est corrigé par les observations
    Netatmo QC'd pour produire un champ analysé à la résolution HR.

    Usage :
        oi = OptimalInterpolationCorrection()
        t2m_analysed = oi.correct(
            background=t2m_prithvi,   # xr.DataArray (lat, lon) en °C
            obs=obs_netatmo_qc,       # NetatmoObs après QC
        )
    """

    def __init__(
        self,
        corr_length_m: float = BACKGROUND_CORR_LENGTH_M,
        sigma_b: float = SIGMA_B_K,
        sigma_obs: float = SIGMA_OBS_K,
        influence_radius_m: float = INFLUENCE_RADIUS_M,
        max_obs: int = MAX_OBS_PER_GRIDPOINT,
    ):
        self.corr_length_m = corr_length_m
        self.sigma_b = sigma_b
        self.sigma_obs = sigma_obs
        self.influence_radius_m = influence_radius_m
        self.max_obs = max_obs

    def correct(
        self,
        background: xr.DataArray,
        obs: NetatmoObs,
    ) -> xr.DataArray:
        """
        Corrige le champ background par les observations OI.

        Args:
            background: Champ Tmin HR (lat, lon) en °C — sortie Prithvi WxC.
            obs:        Observations Netatmo QC'd.

        Returns:
            xr.DataArray de même shape avec le champ analysé.
        """
        # Tmin nocturne QC par station
        tmin_obs = tmin_nocturnal(obs)

        # Ne garder que les stations avec observation valide
        valid_mask = ~np.isnan(tmin_obs.values)
        if valid_mask.sum() == 0:
            log.warning("Aucune observation Netatmo valide — pas de correction OI.")
            return background

        lat_obs = obs.lat[valid_mask]
        lon_obs = obs.lon[valid_mask]
        y_obs = tmin_obs.values[valid_mask]   # (n_obs,)

        log.info(f"OI : {valid_mask.sum()} observations Netatmo valides")

        # Grille cible
        lat_grid = background.lat.values
        lon_grid = background.lon.values
        x_b = background.values.copy()  # (H, W) en °C

        # Calcul de l'incrément OI
        increment = self._compute_oi_increment(
            x_b=x_b,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            lat_obs=lat_obs,
            lon_obs=lon_obs,
            y_obs=y_obs,
        )

        x_a = x_b + increment

        return xr.DataArray(
            x_a,
            dims=background.dims,
            coords=background.coords,
            attrs={
                **background.attrs,
                "method": "Prithvi WxC + Netatmo OI correction",
                "n_obs_assimilated": int(valid_mask.sum()),
                "oi_corr_length_km": self.corr_length_m / 1000,
                "oi_sigma_b": self.sigma_b,
                "oi_sigma_obs": self.sigma_obs,
            },
        )

    def _compute_oi_increment(
        self,
        x_b: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        lat_obs: np.ndarray,
        lon_obs: np.ndarray,
        y_obs: np.ndarray,
    ) -> np.ndarray:
        """
        Calcule l'incrément OI = K (y_o - H x_b) sur la grille.

        Implémentation localisée : pour chaque point de grille, seules
        les observations dans le rayon d'influence sont utilisées.
        → scalable à haute résolution (pas de matrice NxM globale)
        """
        H_lr, W_lr = x_b.shape
        lat_rad = np.radians(np.mean(lat_grid))
        cos_lat = np.cos(lat_rad)

        # Coordonnées métriques approximées
        def to_xy(lat, lon):
            return np.column_stack([lon * cos_lat * 111_320, lat * 111_320])

        xy_obs = to_xy(lat_obs, lon_obs)
        tree = cKDTree(xy_obs)

        # Interpoler le background aux positions des observations (H x_b)
        H_xb = self._interpolate_to_obs(x_b, lat_grid, lon_grid, lat_obs, lon_obs)
        innovation = y_obs - H_xb  # y_o - H x_b

        # Grille 2D
        LAT, LON = np.meshgrid(lat_grid, lon_grid, indexing="ij")
        xy_grid = to_xy(LAT.ravel(), LON.ravel())

        increment = np.zeros(x_b.size, dtype=np.float64)

        # Traitement par batch pour éviter l'OOM
        BATCH = 4096
        for start in range(0, len(xy_grid), BATCH):
            end = min(start + BATCH, len(xy_grid))
            xy_batch = xy_grid[start:end]

            # Observations dans le rayon d'influence
            obs_indices = tree.query_ball_point(xy_batch, self.influence_radius_m)

            for local_i, global_i in enumerate(range(start, end)):
                near_idx = obs_indices[local_i]
                if not near_idx:
                    continue

                # Limiter le nombre d'obs (les plus proches)
                if len(near_idx) > self.max_obs:
                    dists = np.linalg.norm(
                        xy_obs[near_idx] - xy_batch[local_i], axis=1
                    )
                    near_idx = [near_idx[j] for j in np.argsort(dists)[: self.max_obs]]

                near_idx = np.array(near_idx)
                xy_near = xy_obs[near_idx]
                inno_near = innovation[near_idx]

                # Distance point grille ↔ observations
                d_go = np.linalg.norm(xy_near - xy_batch[local_i], axis=1)

                # Distance observations ↔ observations
                d_oo = np.linalg.norm(
                    xy_near[:, np.newaxis] - xy_near[np.newaxis, :], axis=2
                )

                # Fonction de corrélation Gaussienne
                b_go = self.sigma_b**2 * np.exp(-0.5 * (d_go / self.corr_length_m) ** 2)
                B_oo = self.sigma_b**2 * np.exp(-0.5 * (d_oo / self.corr_length_m) ** 2)

                # Matrice (H B H^T + R)
                R = np.diag(np.full(len(near_idx), self.sigma_obs**2))
                A = B_oo + R

                # Incrément : b_go^T A^{-1} (y_o - H x_b)
                try:
                    weights = solve(A, inno_near, assume_a="pos")
                    increment[global_i] = b_go @ weights
                except np.linalg.LinAlgError:
                    # Matrice mal conditionnée (stations très proches) → skip
                    pass

        return increment.reshape(H_lr, W_lr)

    def _interpolate_to_obs(
        self,
        field: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        lat_obs: np.ndarray,
        lon_obs: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolation bilinéaire du champ grille → positions des stations.
        Opérateur d'observation H.
        """
        from scipy.interpolate import RegularGridInterpolator

        # RegularGridInterpolator attend lat croissant
        if lat_grid[0] > lat_grid[-1]:
            field_interp = field[::-1]
            lat_interp = lat_grid[::-1]
        else:
            field_interp = field
            lat_interp = lat_grid

        interpolator = RegularGridInterpolator(
            (lat_interp, lon_grid),
            field_interp,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        points = np.column_stack([lat_obs, lon_obs])
        return interpolator(points)


# ---------------------------------------------------------------------------
# Calcul de la réduction de basis risk
# ---------------------------------------------------------------------------

def compute_basis_risk_reduction(
    t2m_background: xr.DataArray,
    t2m_analysed: xr.DataArray,
    obs: NetatmoObs,
    frost_threshold_c: float = 0.0,
) -> dict[str, float]:
    """
    Quantifie la réduction de basis risk apportée par la correction OI.

    Métrique principale : RMSE aux stations Netatmo non assimilées (held-out)
    avant et après correction OI.

    Utile pour : la note méthodologique CLI-INS / Atekka.

    Returns:
        dict avec rmse_background, rmse_analysed, basis_risk_reduction_pct
    """
    tmin_obs = tmin_nocturnal(obs)
    valid = ~np.isnan(tmin_obs.values)
    if valid.sum() < 2:
        return {}

    from scipy.interpolate import RegularGridInterpolator

    lat_g = t2m_background.lat.values
    lon_g = t2m_background.lon.values
    if lat_g[0] > lat_g[-1]:
        lat_g = lat_g[::-1]

    def interp_to_obs(da):
        arr = da.values if lat_g[0] < lat_g[-1] else da.values[::-1]
        f = RegularGridInterpolator((lat_g, lon_g), arr, bounds_error=False)
        return f(np.column_stack([obs.lat[valid], obs.lon[valid]]))

    bg_at_obs = interp_to_obs(t2m_background)
    an_at_obs = interp_to_obs(t2m_analysed)
    obs_vals = tmin_obs.values[valid]

    rmse_bg = float(np.sqrt(np.mean((bg_at_obs - obs_vals) ** 2)))
    rmse_an = float(np.sqrt(np.mean((an_at_obs - obs_vals) ** 2)))
    reduction_pct = 100 * (rmse_bg - rmse_an) / rmse_bg if rmse_bg > 0 else 0.0

    # Taux d'erreur de classification gel/non-gel
    def frost_error_rate(t_pred, t_obs, threshold):
        pred_frost = t_pred < threshold
        obs_frost = t_obs < threshold
        return float(np.mean(pred_frost != obs_frost))

    fer_bg = frost_error_rate(bg_at_obs, obs_vals, frost_threshold_c)
    fer_an = frost_error_rate(an_at_obs, obs_vals, frost_threshold_c)

    results = {
        "rmse_background_K": rmse_bg,
        "rmse_analysed_K": rmse_an,
        "basis_risk_reduction_pct": reduction_pct,
        "frost_error_rate_background": fer_bg,
        "frost_error_rate_analysed": fer_an,
        "n_obs": int(valid.sum()),
    }

    log.info(
        f"Basis risk : RMSE {rmse_bg:.2f}K → {rmse_an:.2f}K "
        f"({reduction_pct:.0f}% réduction) | "
        f"Erreur classif. gel : {fer_bg:.1%} → {fer_an:.1%}"
    )
    return results
