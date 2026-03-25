"""
Correction de biais par cartographie des quantiles.

Deux méthodes implémentées
--------------------------
EQM — Empirical Quantile Mapping
    Corrige le biais en appliquant la fonction de transfert empirique
    F_obs⁻¹(F_mod(x)).  Simple, robuste, mais ne préserve pas le signal
    de tendance climatique.

QDM — Quantile Delta Mapping  (Cannon et al. 2015)
    Préserve les variations relatives (précipitations) ou absolues (température)
    autour du quantile climatologique. Recommandé pour les projections climatiques
    et la correction de réanalyses avec dérive temporelle.

Usage typique
-------------
    # Calibration sur la période 1985-2010 (ERA5 → CERRA comme "observations")
    eqm = EmpiricalQuantileMapping(n_quantiles=100)
    eqm.fit(era5_hist, cerra_hist)
    era5_corrected = eqm.transform(era5_future)

    # QDM pour les précipitations
    qdm = QuantileDeltaMapping(kind="ratio", n_quantiles=100)
    qdm.fit(era5_hist, cerra_hist)
    era5_corrected = qdm.transform(era5_future)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


class EmpiricalQuantileMapping:
    """
    Empirical Quantile Mapping (EQM).

    La correction est apprise séparément pour chaque mois calendaire afin
    de conserver le cycle saisonnier.

    Parameters
    ----------
    n_quantiles:
        Nombre de quantiles (défaut 100).
    extrapolation:
        Méthode d'extrapolation hors de la plage de calibration :
        'constant' (valeurs extrêmes clampées) ou 'linear'.
    by_month:
        Si True (défaut), ajuste une fonction de transfert par mois calendaire.
    """

    def __init__(
        self,
        n_quantiles: int = 100,
        extrapolation: str = "constant",
        by_month: bool = True,
    ):
        self.n_quantiles = n_quantiles
        self.extrapolation = extrapolation
        self.by_month = by_month
        self._transfer: dict[int, interp1d] = {}  # clé = mois (1-12) ou 0 = global

    # ------------------------------------------------------------------
    def fit(self, modeled: xr.DataArray, observed: xr.DataArray) -> "EmpiricalQuantileMapping":
        """
        Calibre la fonction de transfert.

        Parameters
        ----------
        modeled:
            Champ modélisé (ERA5 ou CERRA basse résolution) — période historique.
            Dim temporelle : 'time'.
        observed:
            Champ de référence (CERRA haute résolution, SAFRAN…) — même période.
        """
        months = list(range(1, 13)) if self.by_month else [0]
        quantiles = np.linspace(0, 100, self.n_quantiles + 1)

        for m in months:
            if m == 0:
                mod_vals = modeled.values.ravel()
                obs_vals = observed.values.ravel()
            else:
                mod_vals = modeled.sel(time=modeled.time.dt.month == m).values.ravel()
                obs_vals = observed.sel(time=observed.time.dt.month == m).values.ravel()

            mask = np.isfinite(mod_vals) & np.isfinite(obs_vals)
            mod_q = np.percentile(mod_vals[mask], quantiles)
            obs_q = np.percentile(obs_vals[mask], quantiles)

            fill_val = (obs_q[0], obs_q[-1]) if self.extrapolation == "constant" else "extrapolate"
            self._transfer[m] = interp1d(
                mod_q, obs_q, kind="linear", bounds_error=False, fill_value=fill_val
            )
        return self

    def transform(self, modeled: xr.DataArray) -> xr.DataArray:
        """
        Applique la correction de biais.

        Parameters
        ----------
        modeled:
            Champ à corriger (même variable que lors du fit).

        Returns
        -------
        xr.DataArray corrigé, mêmes dims/coords que `modeled`.
        """
        if not self._transfer:
            raise RuntimeError("Appeler fit() avant transform().")

        result = modeled.copy(deep=True)
        months = list(range(1, 13)) if self.by_month else [0]

        for m in months:
            if m == 0:
                idx = slice(None)
            else:
                idx = modeled.time.dt.month == m
            block = modeled.sel(time=idx)
            tf = self._transfer[m]
            corrected = xr.apply_ufunc(
                tf, block, dask="parallelized", output_dtypes=[float]
            )
            result.loc[{"time": idx}] = corrected.values

        result.attrs.update(modeled.attrs)
        result.attrs["bias_correction"] = "EQM"
        return result


class QuantileDeltaMapping:
    """
    Quantile Delta Mapping (QDM — Cannon et al. 2015).

    Décompose le signal en :
        - un biais climatologique (corrigé par EQM sur la période de référence)
        - une anomalie (delta) préservée telle quelle

    Pour les précipitations (kind='ratio') :
        x_corr = F_obs_ref⁻¹(F_mod_ref(x_fut)) × (x_fut / F_mod_ref⁻¹(F_mod_ref(x_fut)))

    Pour la température (kind='delta') :
        x_corr = F_obs_ref⁻¹(F_mod_ref(x_fut)) + (x_fut − F_mod_ref⁻¹(F_mod_ref(x_fut)))

    Parameters
    ----------
    kind:
        'delta' pour variables additives (température), 'ratio' pour multiplicatives
        (précipitations).
    n_quantiles:
        Nombre de quantiles.
    by_month:
        Ajustement mensuel (défaut True).
    wet_threshold:
        Seuil (mm) sous lequel un jour est considéré sec — utilisé uniquement
        si kind='ratio' pour éviter la division par zéro.
    """

    def __init__(
        self,
        kind: str = "delta",
        n_quantiles: int = 100,
        by_month: bool = True,
        wet_threshold: float = 0.1,
    ):
        if kind not in ("delta", "ratio"):
            raise ValueError("kind doit être 'delta' ou 'ratio'.")
        self.kind = kind
        self.n_quantiles = n_quantiles
        self.by_month = by_month
        self.wet_threshold = wet_threshold

        # Fonctions de transfert calibrées
        self._mod_cdf: dict[int, interp1d] = {}   # F_mod_ref(x)  → quantile
        self._obs_ppf: dict[int, interp1d] = {}   # F_obs_ref⁻¹(q) → valeur

    # ------------------------------------------------------------------
    def fit(
        self, modeled_ref: xr.DataArray, observed_ref: xr.DataArray
    ) -> "QuantileDeltaMapping":
        """
        Calibre les distributions de référence.

        Parameters
        ----------
        modeled_ref:
            ERA5/CERRA basse résolution sur la période de référence.
        observed_ref:
            Référence haute résolution (CERRA fine, SAFRAN…) — même période.
        """
        months = list(range(1, 13)) if self.by_month else [0]
        quantiles = np.linspace(0, 100, self.n_quantiles + 1)
        q01 = quantiles / 100.0  # [0, 1]

        for m in months:
            if m == 0:
                mod_vals = modeled_ref.values.ravel()
                obs_vals = observed_ref.values.ravel()
            else:
                mod_vals = modeled_ref.sel(time=modeled_ref.time.dt.month == m).values.ravel()
                obs_vals = observed_ref.sel(time=observed_ref.time.dt.month == m).values.ravel()

            if self.kind == "ratio":
                mod_vals = mod_vals[mod_vals > self.wet_threshold]
                obs_vals = obs_vals[obs_vals > self.wet_threshold]

            mask_m = np.isfinite(mod_vals)
            mask_o = np.isfinite(obs_vals)
            mod_q = np.percentile(mod_vals[mask_m], quantiles)
            obs_q = np.percentile(obs_vals[mask_o], quantiles)

            # CDF modèle : valeur → quantile
            self._mod_cdf[m] = interp1d(
                mod_q, q01, kind="linear", bounds_error=False,
                fill_value=(q01[0], q01[-1])
            )
            # PPF observations : quantile → valeur
            self._obs_ppf[m] = interp1d(
                q01, obs_q, kind="linear", bounds_error=False,
                fill_value=(obs_q[0], obs_q[-1])
            )
        return self

    def transform(self, modeled_future: xr.DataArray) -> xr.DataArray:
        """
        Applique QDM.

        Parameters
        ----------
        modeled_future:
            Champ à corriger (peut être la même période que ref ou une période future).

        Returns
        -------
        xr.DataArray corrigé.
        """
        if not self._mod_cdf:
            raise RuntimeError("Appeler fit() avant transform().")

        result = modeled_future.copy(deep=True)
        months = list(range(1, 13)) if self.by_month else [0]

        for m in months:
            idx = slice(None) if m == 0 else (modeled_future.time.dt.month == m)
            block = modeled_future.sel(time=idx).values  # numpy

            tau = self._mod_cdf[m](block)           # quantile dans distribution de ref
            x_ref = self._obs_ppf[m](tau)           # valeur correspondante dans obs_ref

            if self.kind == "delta":
                delta = block - self._obs_ppf[m](tau)
                corrected = x_ref + delta
            else:
                # Ratio : évite division par zéro
                denom = np.where(np.abs(x_ref) > self.wet_threshold, x_ref, self.wet_threshold)
                corrected = x_ref * np.where(np.abs(denom) > 0, block / denom, 1.0)
                corrected = np.maximum(corrected, 0.0)

            result.loc[{"time": idx}] = corrected

        result.attrs.update(modeled_future.attrs)
        result.attrs["bias_correction"] = f"QDM-{self.kind}"
        return result


# ---------------------------------------------------------------------------
# BCSD — Bias Correction & Spatial Disaggregation (version simplifiée)
# ---------------------------------------------------------------------------

def bcsd_temperature(
    t_coarse: xr.DataArray,
    t_ref_coarse: xr.DataArray,
    t_ref_fine: xr.DataArray,
    n_quantiles: int = 100,
) -> xr.DataArray:
    """
    BCSD pour la température :
    1. QDM en basse résolution (ERA5 → CERRA coarse)
    2. Désagrégation spatiale par interpolation bilinéaire +
       anomalie haute fréquence de la référence climatologique.

    Parameters
    ----------
    t_coarse:
        Température ERA5 à corriger/désagréger.
    t_ref_coarse:
        Climatologie de référence basse résolution (même grille que t_coarse).
    t_ref_fine:
        Climatologie de référence haute résolution (grille cible).

    Returns
    -------
    xr.DataArray température haute résolution corrigée.
    """
    # Étape 1 : QDM basse résolution
    qdm = QuantileDeltaMapping(kind="delta", n_quantiles=n_quantiles)
    qdm.fit(t_coarse, t_ref_coarse)
    t_bc = qdm.transform(t_coarse)

    # Étape 2 : interpolation bilinéaire de t_bc sur la grille fine
    lat_fine = t_ref_fine.coords.get("lat", t_ref_fine.coords.get("latitude"))
    lon_fine = t_ref_fine.coords.get("lon", t_ref_fine.coords.get("longitude"))
    lat_name = "latitude" if "latitude" in t_bc.coords else "lat"
    lon_name = "longitude" if "longitude" in t_bc.coords else "lon"

    t_bc_fine = t_bc.interp({lat_name: lat_fine, lon_name: lon_fine}, method="linear")

    # Étape 3 : ajout de l'anomalie spatiale de la référence fine
    # clim_ref_fine_monthly — la variabilité sub-maille
    clim_coarse_monthly = t_ref_coarse.groupby("time.month").mean("time")
    clim_fine_monthly = t_ref_fine.groupby("time.month").mean("time")
    clim_coarse_fine = clim_coarse_monthly.interp(
        {lat_name: lat_fine, lon_name: lon_fine}, method="linear"
    )
    spatial_anomaly = clim_fine_monthly - clim_coarse_fine  # (12, y, x)

    # Ajouter l'anomalie mensuelle correspondante
    months = t_bc_fine.time.dt.month
    anom = spatial_anomaly.sel(month=months).drop_vars("month")
    return (t_bc_fine + anom).rename("t2m_bcsd")
