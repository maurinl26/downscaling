"""
netatmo_qc.py — Contrôle qualité des observations Netatmo pour nuits de gel.

Pipeline en 4 niveaux (adapté de Meier et al. 2017 / CrowdQC+ / Nipen et al. 2020) :
  L1 : Plage climatologique (range check)
  L2 : Correction constante de temps du capteur (τ ≈ 13 min, Coney et al. 2022)
  L3 : Correction lapse-rate (ramène à altitude de référence commune)
  L4 : Buddy check spatial (cohérence avec voisins après normalisation altitude)

Pourquoi ce QC est plus simple la nuit :
  - Biais radiatif solaire = 0 → L4 suffit à détecter les outliers
  - Pas besoin de correction ombre/exposition (Clark scheme diurne)
  - Erreurs résiduelles ~ ±0.5°C au lieu de >1°C en journée (CrowdQC+, 2021)

Référence :
  Coney et al. (2022) Meteorological Applications 10.1002/met.2075
  Meier et al. (2017) Urban Climate 10.1016/j.uclim.2016.10.006
  CrowdQC+ (Fenner et al. 2021) Frontiers Env. Sci. 10.3389/fenvs.2021.720747
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes Netatmo (Coney et al. 2022, lab experiments)
# ---------------------------------------------------------------------------
TAU_SECONDS = 762.0          # Constante de temps capteur (τ = 12.7 min)
SENSOR_ACCURACY_K = 0.3      # Précision intrinsèque ±0.3°C
LAPSE_RATE_DEFAULT = -6.5e-3 # K/m (gradient atmosphérique standard)

# Plages climatologiques nocturnes Drôme/Ardèche (oct–mai)
T_MIN_PLAUSIBLE_C = -20.0    # °C — gel extrême (record Drôme : -18°C)
T_MAX_PLAUSIBLE_C = 25.0     # °C — nuit la plus chaude du printems

# Buddy check
BUDDY_RADIUS_M = 15_000      # 15 km de rayon de recherche
MIN_BUDDIES = 3              # Nombre minimum de voisins requis
BUDDY_SIGMA_THRESHOLD = 4.0  # Seuil en sigma pour rejet (conservateur la nuit)


# ---------------------------------------------------------------------------
# Structure de données
# ---------------------------------------------------------------------------

@dataclass
class NetatmoObs:
    """
    Observations Netatmo pour une nuit donnée.

    Args:
        station_id:  Identifiants uniques des stations
        lat, lon:    Coordonnées géographiques (degrés)
        elevation_m: Altitude des stations (m) — depuis API Netatmo ou SRTM
        t_raw:       Températures brutes (°C), shape (n_stations, n_hours)
        times:       Index temporel (heures)
        qc_flags:    Masque booléen (True = valide), shape (n_stations, n_hours)
    """
    station_id: np.ndarray           # (n,)
    lat: np.ndarray                  # (n,)
    lon: np.ndarray                  # (n,)
    elevation_m: np.ndarray          # (n,)
    t_raw: np.ndarray                # (n, t) en °C
    times: pd.DatetimeIndex
    qc_flags: np.ndarray = field(init=False)

    def __post_init__(self):
        self.qc_flags = np.ones(self.t_raw.shape, dtype=bool)

    @property
    def t_qc(self) -> np.ndarray:
        """T observée après masquage des outliers (NaN là où QC=False)."""
        out = self.t_raw.copy()
        out[~self.qc_flags] = np.nan
        return out

    @property
    def n_stations(self) -> int:
        return len(self.station_id)

    @property
    def n_valid_per_hour(self) -> np.ndarray:
        return self.qc_flags.sum(axis=0)


# ---------------------------------------------------------------------------
# Pipeline QC
# ---------------------------------------------------------------------------

class NetatmoNocturnalQC:
    """
    Contrôle qualité Netatmo optimisé pour les nuits de gel.

    Usage :
        obs = NetatmoObs(...)
        qc = NetatmoNocturnalQC()
        obs_clean = qc.run(obs)
        print(f"Taux de rétention : {obs_clean.qc_flags.mean():.1%}")
    """

    def __init__(
        self,
        lapse_rate: float = LAPSE_RATE_DEFAULT,
        reference_elevation_m: float = 200.0,   # Altitude de référence Drôme
        buddy_radius_m: float = BUDDY_RADIUS_M,
        buddy_sigma: float = BUDDY_SIGMA_THRESHOLD,
        correct_tau: bool = True,
    ):
        self.lapse_rate = lapse_rate
        self.ref_elevation_m = reference_elevation_m
        self.buddy_radius_m = buddy_radius_m
        self.buddy_sigma = buddy_sigma
        self.correct_tau = correct_tau

    def run(self, obs: NetatmoObs) -> NetatmoObs:
        """Exécute le pipeline QC complet. Modifie obs.qc_flags in-place."""
        n_init = obs.qc_flags.sum()

        # L1 — Plage climatologique
        self._range_check(obs)
        log.debug(f"Après L1 (range) : {obs.qc_flags.sum()}/{obs.t_raw.size} valeurs")

        # L2 — Correction constante de temps (lag thermique capteur)
        if self.correct_tau:
            self._tau_correction(obs)

        # L3 — Correction lapse-rate (normalisation altitude)
        t_normalized = self._lapse_rate_correction(obs)

        # L4 — Buddy check spatial
        self._buddy_check(obs, t_normalized)
        n_final = obs.qc_flags.sum()

        retention = n_final / n_init if n_init > 0 else 0.0
        log.info(
            f"QC terminé : {n_final}/{n_init} valeurs retenues "
            f"({retention:.1%}) — {obs.n_stations} stations"
        )
        return obs

    # ------------------------------------------------------------------
    # L1 — Plage climatologique
    # ------------------------------------------------------------------

    def _range_check(self, obs: NetatmoObs) -> None:
        """Rejette les valeurs hors plage physiquement plausible."""
        out_of_range = (obs.t_raw < T_MIN_PLAUSIBLE_C) | (obs.t_raw > T_MAX_PLAUSIBLE_C)
        obs.qc_flags[out_of_range] = False

    # ------------------------------------------------------------------
    # L2 — Correction constante de temps capteur (Coney et al. 2022)
    # ------------------------------------------------------------------

    def _tau_correction(self, obs: NetatmoObs) -> None:
        """
        Corrige le lag thermique du capteur Netatmo (τ ≈ 13 min).

        Formule (CrowdQC+, Miloshevich 2004) :
            T_corr[i] = T[i] + (T[i] - T[i-1]) * (e^(dt/τ) - 1)^-1 * (1 - e^(-dt/τ))

        Pour des données horaires (dt=3600s >> τ=762s) : effet marginal
        mais utile pour données 5 min ou si la nuit est très dynamique.
        """
        dt_s = 3600.0  # données horaires
        alpha = np.exp(-dt_s / TAU_SECONDS)

        # Appliquer uniquement sur les valeurs valides
        T = obs.t_raw.copy()
        T[~obs.qc_flags] = np.nan

        T_corr = T.copy()
        for i in range(1, T.shape[1]):
            delta = T[:, i] - T[:, i - 1]
            # Correction proportionnelle au taux de changement
            T_corr[:, i] = T[:, i] + delta * (1 - alpha) / alpha

        # Mettre à jour les données brutes avec la version corrigée
        # Conserver NaN là où le QC a déjà invalidé
        obs.t_raw[:] = np.where(obs.qc_flags, T_corr, obs.t_raw)

    # ------------------------------------------------------------------
    # L3 — Correction lapse-rate
    # ------------------------------------------------------------------

    def _lapse_rate_correction(self, obs: NetatmoObs) -> np.ndarray:
        """
        Ramène toutes les stations à une altitude de référence commune.
        Retourne T normalisée (même altitude) pour le buddy check.

        T_ref = T_obs - γ × (z_station - z_ref)
        """
        dz = obs.elevation_m[:, np.newaxis] - self.ref_elevation_m  # (n, 1)
        t_normalized = obs.t_raw - self.lapse_rate * dz  # broadcast sur les heures
        return t_normalized  # (n, t) — comparable entre altitudes

    # ------------------------------------------------------------------
    # L4 — Buddy check spatial
    # ------------------------------------------------------------------

    def _buddy_check(self, obs: NetatmoObs, t_normalized: np.ndarray) -> None:
        """
        Contrôle de cohérence spatiale après normalisation altitude.
        Pour chaque station, vérifie que sa T est dans ±sigma × std(voisins).

        Utilise un KD-tree en coordonnées cartésiennes approximées.
        """
        # Conversion lat/lon → coordonnées métriques approx (Drôme ~44-45°N)
        lat_rad = np.radians(np.mean(obs.lat))
        cos_lat = np.cos(lat_rad)

        xy = np.column_stack([
            obs.lon * cos_lat * 111_320,  # m
            obs.lat * 111_320,
        ])
        tree = cKDTree(xy)

        for t_idx in range(t_normalized.shape[1]):
            T_t = t_normalized[:, t_idx]
            valid = obs.qc_flags[:, t_idx]

            if valid.sum() < MIN_BUDDIES + 1:
                # Pas assez de stations pour un buddy check fiable
                continue

            for i in range(obs.n_stations):
                if not valid[i]:
                    continue

                # Voisins dans le rayon
                indices = tree.query_ball_point(xy[i], self.buddy_radius_m)
                neighbors = [j for j in indices if j != i and valid[j]]

                if len(neighbors) < MIN_BUDDIES:
                    # Isolée — ne pas rejeter (station peut être en fond de vallée)
                    continue

                T_neighbors = T_t[neighbors]
                T_mean = np.nanmean(T_neighbors)
                T_std = max(np.nanstd(T_neighbors), SENSOR_ACCURACY_K)

                z_score = abs(T_t[i] - T_mean) / T_std
                if z_score > self.buddy_sigma:
                    obs.qc_flags[i, t_idx] = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tmin_nocturnal(obs: NetatmoObs) -> pd.Series:
    """
    Calcule Tmin nocturne par station à partir des observations QC'd.
    Retourne un Series {station_id: Tmin_°C}.
    """
    T_qc = obs.t_qc  # NaN aux positions rejetées
    tmin = np.nanmin(T_qc, axis=1)  # min sur les heures nocturnes
    return pd.Series(tmin, index=obs.station_id, name="Tmin_C")


def load_netatmo_parquet(
    path: str,
    date: str,
    bbox: dict[str, float] | None = None,
) -> NetatmoObs:
    """
    Charge les données Netatmo depuis un fichier Parquet (format NDA Netatmo).

    Schéma attendu : station_id, lat, lon, elevation_m, timestamp, t_celsius
    """
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filtre temporel sur la nuit donnée (20h–08h)
    night_start = pd.Timestamp(date) + pd.Timedelta("20h")
    next_morning = pd.Timestamp(date) + pd.Timedelta("1D") + pd.Timedelta("8h")
    df = df[(df["timestamp"] >= night_start) & (df["timestamp"] < next_morning)]

    if bbox:
        df = df[
            (df["lat"] >= bbox["lat_min"]) & (df["lat"] <= bbox["lat_max"]) &
            (df["lon"] >= bbox["lon_min"]) & (df["lon"] <= bbox["lon_max"])
        ]

    if df.empty:
        raise ValueError(f"Aucune donnée Netatmo pour la nuit du {date}")

    # Pivot : stations × heures
    pivot = df.pivot_table(
        index="station_id", columns="timestamp", values="t_celsius", aggfunc="mean"
    )
    times = pd.DatetimeIndex(pivot.columns)

    meta = df.drop_duplicates("station_id").set_index("station_id")

    return NetatmoObs(
        station_id=pivot.index.values,
        lat=meta.loc[pivot.index, "lat"].values,
        lon=meta.loc[pivot.index, "lon"].values,
        elevation_m=meta.loc[pivot.index, "elevation_m"].values,
        t_raw=pivot.values.astype(np.float32),
        times=times,
    )
