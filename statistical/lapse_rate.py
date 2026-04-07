"""
Correction de température par le gradient adiabatique (lapse-rate).

Principe
--------
La température décroît avec l'altitude. ERA5 (31 km) et CERRA (5.5 km) ne
représentent pas fidèlement l'altitude locale des pixels kilométriques. On corrige :

    T_fin(x, y) = T_grossier(x, y) + γ(mois) × [z_fin(x, y) – z_grossier(x, y)]

avec γ ≈ −6.5 K/km (lapse-rate standard), calibrable mensuellement sur stations.

Classes
-------
- LapseRateCorrector   : correction simple élévation → température
- MonthlyLapseRate     : estimation du gradient depuis un réseau de stations
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import xarray as xr


#: Lapse-rate standard ICAO (K m⁻¹) — valeur négative = refroidissement en altitude
STANDARD_LAPSE_RATE = -6.5e-3  # K/m


class LapseRateCorrector:
    """
    Correction de température par différence d'altitude entre la source (ERA5/CERRA)
    et le MNT haute résolution.

    Parameters
    ----------
    lapse_rate:
        Gradient thermique vertical (K m⁻¹). Peut être un scalaire unique,
        un tableau (12,) pour des valeurs mensuelles, ou un DataArray (time, y, x)
        pour une correction spatialement et temporellement variable.
        Défaut : -6.5e-3 K/m (lapse-rate standard ICAO).
    """

    def __init__(self, lapse_rate: float | np.ndarray | xr.DataArray = STANDARD_LAPSE_RATE):
        self.lapse_rate = lapse_rate

    def correct(
        self,
        t_coarse: xr.DataArray,
        z_coarse: xr.DataArray,
        z_fine: xr.DataArray,
    ) -> xr.DataArray:
        """
        Applique la correction lapse-rate.

        Parameters
        ----------
        t_coarse:
            Température ERA5 / CERRA regrillée sur la grille fine (K).
            Dims : (time, y, x) ou (time, lat, lon).
        z_coarse:
            Altitude de la réanalyse regrillée sur la grille fine (m).
            Dims : (y, x) ou (lat, lon).
        z_fine:
            Altitude du MNT haute résolution (m). Mêmes dims spatiales que z_coarse.

        Returns
        -------
        xr.DataArray de même shape que t_coarse, températures corrigées (K).
        """
        dz = z_fine - z_coarse  # différence d'altitude (m), positif = MNT plus haut

        gamma = self._get_gamma(t_coarse)
        correction = gamma * dz

        t_corrected = t_coarse + correction
        t_corrected.attrs.update(t_coarse.attrs)
        t_corrected.attrs["lapse_rate_correction"] = "applied"
        return t_corrected.rename(t_coarse.name or "t2m_corrected")

    def _get_gamma(self, t_coarse: xr.DataArray) -> float | xr.DataArray:
        """Retourne le lapse-rate adapté (scalaire, mensuel ou DataArray)."""
        if isinstance(self.lapse_rate, (int, float)):
            return self.lapse_rate

        if isinstance(self.lapse_rate, np.ndarray) and self.lapse_rate.shape == (12,):
            # Valeurs mensuelles → aligner sur l'axe temporel
            months = t_coarse.time.dt.month.values - 1  # 0-indexed
            gamma_series = self.lapse_rate[months]
            # Broadcaster sur les dims spatiales
            return xr.DataArray(
                gamma_series,
                dims=["time"],
                coords={"time": t_coarse.time},
            )

        if isinstance(self.lapse_rate, xr.DataArray):
            return self.lapse_rate

        raise TypeError(f"lapse_rate de type non supporté : {type(self.lapse_rate)}")


class MonthlyLapseRate:
    """
    Estime le gradient thermique vertical mensuel depuis un réseau de stations.

    Utilise une régression linéaire simple T ~ a + γ·z pour chaque mois.
    Applicable au domaine Drôme-Ardèche avec les stations Météo-France SYNOP.

    Parameters
    ----------
    station_altitudes:
        Altitudes des stations (m). Shape (n_stations,).
    station_temps:
        Température mensuelle moyenne par station (K ou °C). Shape (n_months, n_stations).
        n_months = 12 pour une climatologie.
    """

    def __init__(
        self,
        station_altitudes: np.ndarray,
        station_temps: np.ndarray,
    ):
        self.z = np.asarray(station_altitudes, dtype=float)
        self.T = np.asarray(station_temps, dtype=float)
        if self.T.shape[-1] != len(self.z):
            raise ValueError("station_temps doit avoir len(station_altitudes) colonnes.")

    def fit(self) -> np.ndarray:
        """
        Ajuste la régression T(z) par moindres carrés pour chaque mois.

        Returns
        -------
        gamma: np.ndarray shape (12,) — gradient K/m (négatif si T décroît avec z).
        """
        from scipy.stats import linregress

        n_months = self.T.shape[0]
        gamma = np.zeros(n_months)
        r2 = np.zeros(n_months)

        for m in range(n_months):
            mask = np.isfinite(self.T[m])
            if mask.sum() < 3:
                warnings.warn(f"Mois {m + 1}: moins de 3 stations valides, γ standard utilisé.")
                gamma[m] = STANDARD_LAPSE_RATE
                continue
            slope, intercept, r, p, se = linregress(self.z[mask], self.T[m][mask])
            gamma[m] = slope
            r2[m] = r**2
            if r2[m] < 0.5:
                warnings.warn(
                    f"Mois {m + 1}: R²={r2[m]:.2f} faible pour la régression T–z "
                    f"(N={mask.sum()} stations)."
                )

        self.gamma_ = gamma
        self.r2_ = r2
        return gamma

    def to_corrector(self) -> LapseRateCorrector:
        """Retourne un LapseRateCorrector avec les gradients mensuels ajustés."""
        if not hasattr(self, "gamma_"):
            self.fit()
        return LapseRateCorrector(lapse_rate=self.gamma_)


# ---------------------------------------------------------------------------
# Correction de la pression de surface (hydrostatique)
# ---------------------------------------------------------------------------

def correct_surface_pressure(
    sp_coarse: xr.DataArray,
    z_coarse: xr.DataArray,
    z_fine: xr.DataArray,
    t_mean_k: float = 288.15,
) -> xr.DataArray:
    """
    Corrige la pression de surface pour la différence d'altitude (loi hypsométrique).

    sp_fine = sp_coarse × exp(–g × dz / (R_d × T_mean))

    Parameters
    ----------
    sp_coarse:
        Pression de surface regrillée sur la grille fine (Pa).
    z_coarse, z_fine:
        Altitudes sources et cibles (m).
    t_mean_k:
        Température moyenne de référence (K). Défaut : 288.15 K (ISA).
    """
    g = 9.80665   # m s⁻²
    Rd = 287.05   # J kg⁻¹ K⁻¹
    dz = z_fine - z_coarse
    sp_fine = sp_coarse * np.exp(-g * dz / (Rd * t_mean_k))
    sp_fine.attrs.update(sp_coarse.attrs)
    sp_fine.attrs["hypsometric_correction"] = "applied"
    return sp_fine.rename(sp_coarse.name or "sp_corrected")
