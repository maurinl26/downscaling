#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "scipy>=1.10",
#   "xarray>=2023.1",
#   "netCDF4>=1.6",
#   "pandas>=2.0",
#   "pyyaml>=6.0",
#   "downscaling",
# ]
# ///
"""
Comparaison quantitative des méthodes de descente d'échelle.

Sources comparées
-----------------
  ERA5 brut      : référence basse résolution (31 km)
  CERRA brut     : référence intermédiaire (5.5 km)
  Stat (lapse+QDM) : descente d'échelle statistique (1 km)
  DL (U-Net FiLM)  : descente d'échelle deep learning (1 km) — optionnel
  PMAP ERA5        : descente d'échelle dynamique, IC ERA5 (1 km)
  PMAP CERRA       : descente d'échelle dynamique, IC CERRA (1 km)

Vérification
------------
  Stations SYNOP : T2m horaire, Tmin nocturne, vent
  Métriques      : RMSE, biais, Tmin error, hit rate gel, skill score

Sorties
-------
  output/verification/scores_april2021.csv      : métriques par nuit / méthode / station
  output/verification/tmin_maps_april2021.nc    : cartes Tmin par méthode (pour visualisation)
  output/verification/report_april2021.txt      : résumé texte

Usage
-----
    python runs/scripts/compare_downscaling.py \\
        --config     runs/april2021/config.yml \\
        --stat-out   output/stat/stat_downscaled_april2021.nc \\
        --pmap-dir   output/pmap/ \\
        --obs-synop  data/obs/synop_april2021.csv \\
        --out-dir    output/verification/
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

log = logging.getLogger(__name__)

_K0 = 273.15


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_pmap_tmin(pmap_dir: Path, forcing: str, night_date: str) -> xr.DataArray | None:
    """
    Charge le Tmin nocturne depuis les sorties PMAP.

    Les sorties PMAP sont des snapshots horaires (theta_total, exner_total).
    T = theta × exner — on reconstruit T2m depuis le premier niveau vertical.
    """
    night_compact = night_date.replace("-", "")[2:]   # "210427"
    run_dir = pmap_dir / forcing / f"nuit_{night_compact}"

    if not run_dir.exists():
        log.warning(f"Dossier PMAP manquant : {run_dir}")
        return None

    nc_files = sorted(run_dir.glob("theta_total_*.nc"))
    if not nc_files:
        log.warning(f"Aucun fichier theta_total_*.nc dans {run_dir}")
        return None

    temps = []
    for f in nc_files:
        try:
            ds = xr.open_dataset(f, engine="netcdf4")
            theta = ds["theta_total"]
            # Exner correspondant
            exner_f = f.parent / f.name.replace("theta_total", "exner_total")
            exner = xr.open_dataset(exner_f, engine="netcdf4")["exner_total"]
            t_k = (theta * exner).isel(z=0)   # premier niveau vertical ≈ 25 m
            temps.append(t_k)
        except Exception as e:
            log.debug(f"  Erreur lecture {f} : {e}")
    if not temps:
        return None

    t_series = xr.concat(temps, dim="time")
    return (t_series.min("time") - _K0).rename("tmin_c")


def load_era5_tmin(era5_file: str, night_date: str,
                   start_h: int = 18, end_h: int = 6) -> xr.DataArray | None:
    """Charge le Tmin nocturne depuis ERA5 single-level (mn2t ou t2m)."""
    try:
        ds = xr.open_dataset(era5_file, engine="netcdf4")
    except FileNotFoundError:
        return None

    var = "mn2t" if "mn2t" in ds else ("t2m" if "t2m" in ds else None)
    if var is None:
        return None

    night = ds[var].sel(time=ds.time.dt.date.astype(str) == night_date)
    if len(night.time) == 0:
        return None

    if var == "t2m":
        # Calculer Tmin depuis horaires
        night = night.where(
            (night.time.dt.hour >= start_h) | (night.time.dt.hour <= end_h)
        ).min("time")

    return (night.squeeze() - _K0).rename("tmin_c")


def load_synop(csv_path: str) -> dict[str, list[dict]]:
    """
    Charge les observations SYNOP depuis un CSV.

    Format attendu (colonnes) :
        station_name, wmo_id, datetime_utc (ISO), t2m_c, wind_ms

    Returns
    -------
    Dict {wmo_id: [{datetime_utc, t2m_c, wind_ms}, …]}
    """
    obs = {}
    path = Path(csv_path)
    if not path.exists():
        log.warning(f"Fichier SYNOP manquant : {csv_path}")
        return obs

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wmo = row.get("wmo_id", row.get("station_id", "?"))
            if wmo not in obs:
                obs[wmo] = []
            obs[wmo].append({
                "datetime_utc": row.get("datetime_utc"),
                "t2m_c": float(row.get("t2m_c", "nan")),
                "wind_ms": float(row.get("wind_ms", "nan")),
            })
    return obs


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(obs)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - obs[mask]) ** 2)))


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(obs)
    if not mask.any():
        return np.nan
    return float(np.mean(pred[mask] - obs[mask]))


def frost_hit_rate(pred_tmin: np.ndarray, obs_tmin: np.ndarray,
                   threshold_c: float = -2.0) -> tuple[float, float]:
    """Retourne (hit_rate, false_alarm_rate) pour le gel."""
    obs_frost = obs_tmin < threshold_c
    pred_frost = pred_tmin < threshold_c
    n_obs_frost = obs_frost.sum()
    if n_obs_frost == 0:
        return np.nan, np.nan
    hr = float((pred_frost & obs_frost).sum() / n_obs_frost)
    far = float((pred_frost & ~obs_frost).sum() / max((~obs_frost).sum(), 1))
    return hr, far


def skill_score_vs_reference(
    pred_tmin: np.ndarray,
    obs_tmin: np.ndarray,
    ref_tmin: np.ndarray,
) -> float:
    """
    Skill score de Murphy (1988) :
        SS = 1 - MSE(pred, obs) / MSE(ref, obs)
    SS > 0 : pred meilleur que ref
    SS = 1 : prédiction parfaite
    SS < 0 : pred pire que ref
    """
    mask = np.isfinite(pred_tmin) & np.isfinite(obs_tmin) & np.isfinite(ref_tmin)
    if not mask.any():
        return np.nan
    mse_pred = np.mean((pred_tmin[mask] - obs_tmin[mask]) ** 2)
    mse_ref = np.mean((ref_tmin[mask] - obs_tmin[mask]) ** 2)
    if mse_ref == 0:
        return np.nan
    return float(1.0 - mse_pred / mse_ref)


# ---------------------------------------------------------------------------
# Extraction spatiale au point d'une station
# ---------------------------------------------------------------------------

def extract_at_station(
    da: xr.DataArray,
    lat: float,
    lon: float,
    method: str = "nearest",
) -> float:
    """Extrait la valeur d'un DataArray au point (lat, lon) le plus proche."""
    try:
        # Tente via coordonnées lat/lon
        lat_name = "latitude" if "latitude" in da.coords else "lat"
        lon_name = "longitude" if "longitude" in da.coords else "lon"
        val = da.sel(
            {lat_name: lat, lon_name: lon},
            method=method,
        )
        return float(val.values)
    except Exception:
        # Fallback : interp xarray
        try:
            return float(da.interp(lat=lat, lon=lon).values)
        except Exception:
            return np.nan


# ---------------------------------------------------------------------------
# Comparaison principale
# ---------------------------------------------------------------------------

class DownscalingComparison:
    """
    Calcule et stocke les métriques de comparaison pour toutes les nuits critiques.

    Parameters
    ----------
    cfg:
        Configuration (runs/april2021/config.yml).
    stat_output:
        Chemin NetCDF sortie descente d'échelle statistique.
    dl_output:
        Chemin NetCDF sortie DL (optionnel).
    pmap_dir:
        Répertoire racine des runs PMAP.
    obs_synop_csv:
        Fichier CSV observations SYNOP.
    """

    def __init__(
        self,
        cfg: dict,
        stat_output: str | None = None,
        dl_output: str | None = None,
        pmap_dir: str | None = None,
        obs_synop_csv: str | None = None,
    ):
        self.cfg = cfg
        self.stations = cfg.get("verification", {}).get("stations", [])
        self.cold_nights = cfg.get("cold_nights", [])
        self.frost_thresh = cfg.get("detection", {}).get("tmin_threshold_c", -2.0)

        self.ds_stat = xr.open_dataset(stat_output, engine="netcdf4") if stat_output else None
        self.ds_dl   = xr.open_dataset(dl_output, engine="netcdf4") if dl_output else None
        self.pmap_dir = Path(pmap_dir) if pmap_dir else None
        self.obs_synop = load_synop(obs_synop_csv) if obs_synop_csv else {}

        self.scores: list[dict] = []

    def run(self):
        """Lance la comparaison sur toutes les nuits critiques."""
        for night in self.cold_nights:
            if not night.get("run_pmap", True):
                continue
            self._compare_night(night)

    def _compare_night(self, night: dict):
        """Compare toutes les méthodes pour une nuit donnée."""
        date = night["date"]
        log.info(f"Comparaison nuit {date}…")

        # Tmin nocturne ERA5 brut (regrillé sur la grille fine si possible)
        era5_sl = self.cfg["data"]["era5"]["single_level"].get(date)
        tmin_era5 = load_era5_tmin(era5_sl, date) if era5_sl else None

        # Tmin stat downscaling
        tmin_stat = None
        if self.ds_stat is not None and "t2m" in self.ds_stat:
            t2m = self.ds_stat["t2m"] - _K0
            night_t = t2m.sel(time=t2m.time.dt.date.astype(str) == date)
            if len(night_t.time) > 0:
                tmin_stat = night_t.min("time").rename("tmin_c")

        # Tmin DL downscaling
        tmin_dl = None
        if self.ds_dl is not None and "t2m" in self.ds_dl:
            t2m = self.ds_dl["t2m"] - _K0
            night_t = t2m.sel(time=t2m.time.dt.date.astype(str) == date)
            if len(night_t.time) > 0:
                tmin_dl = night_t.min("time").rename("tmin_c")

        # Tmin PMAP (ERA5 et CERRA)
        tmin_pmap_era5 = tmin_pmap_cerra = None
        if self.pmap_dir:
            tmin_pmap_era5  = load_pmap_tmin(self.pmap_dir, "era5",  date)
            tmin_pmap_cerra = load_pmap_tmin(self.pmap_dir, "cerra", date)

        # Scores aux stations SYNOP
        for station in self.stations:
            wmo = str(station["wmo"])
            lat, lon = station["lat"], station["lon"]
            name = station["name"]

            # Tmin observée (SYNOP) pour cette nuit
            obs_tmin = self._get_synop_tmin(wmo, date)
            if np.isnan(obs_tmin):
                continue

            row = {
                "date":    date,
                "station": name,
                "wmo":     wmo,
                "obs_tmin_c": round(obs_tmin, 2),
            }

            # Extraction de chaque méthode au point station
            for label, da in [
                ("era5",       tmin_era5),
                ("stat",       tmin_stat),
                ("dl",         tmin_dl),
                ("pmap_era5",  tmin_pmap_era5),
                ("pmap_cerra", tmin_pmap_cerra),
            ]:
                if da is None:
                    row[f"{label}_tmin_c"]  = np.nan
                    row[f"{label}_error_k"] = np.nan
                    continue
                val = extract_at_station(da, lat, lon)
                row[f"{label}_tmin_c"]  = round(val, 2)
                row[f"{label}_error_k"] = round(val - obs_tmin, 2)

            # Skill score stat vs ERA5
            if not np.isnan(row.get("stat_tmin_c", np.nan)) and \
               not np.isnan(row.get("era5_tmin_c", np.nan)):
                row["skill_stat_vs_era5"] = round(
                    skill_score_vs_reference(
                        np.array([row["stat_tmin_c"]]),
                        np.array([obs_tmin]),
                        np.array([row["era5_tmin_c"]]),
                    ), 3
                )

            self.scores.append(row)
            log.debug(f"  {name}: obs={obs_tmin:.1f}°C  "
                      f"stat={row.get('stat_tmin_c', '?')}  "
                      f"pmap_era5={row.get('pmap_era5_tmin_c', '?')}")

    def _get_synop_tmin(self, wmo: str, date: str) -> float:
        """Calcule la Tmin nocturne depuis les observations SYNOP."""
        obs = self.obs_synop.get(wmo, [])
        night_temps = [
            r["t2m_c"] for r in obs
            if r.get("datetime_utc", "")[:10] in (date, _next_date(date))
            and np.isfinite(r.get("t2m_c", np.nan))
        ]
        return float(np.min(night_temps)) if night_temps else np.nan

    def save_csv(self, out_path: str):
        """Sauvegarde les scores en CSV."""
        if not self.scores:
            log.warning("Aucun score calculé.")
            return
        keys = list(self.scores[0].keys())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.scores)
        log.info(f"Scores → {out_path}")

    def print_summary(self):
        """Affiche un résumé texte des métriques agrégées."""
        if not self.scores:
            print("Aucun score disponible.")
            return

        methods = ["era5", "stat", "dl", "pmap_era5", "pmap_cerra"]
        print("\n" + "=" * 70)
        print("RÉSUMÉ — Vérification descente d'échelle Drôme-Ardèche Avril 2021")
        print("=" * 70)

        for method in methods:
            errors = [
                r.get(f"{method}_error_k")
                for r in self.scores
                if r.get(f"{method}_error_k") is not None and
                   np.isfinite(r.get(f"{method}_error_k", np.nan))
            ]
            if not errors:
                continue
            errors = np.array(errors)
            print(f"\n  {method.upper():12s}  "
                  f"RMSE={np.sqrt((errors**2).mean()):.2f} K  "
                  f"Biais={errors.mean():+.2f} K  "
                  f"MAE={np.abs(errors).mean():.2f} K  "
                  f"(N={len(errors)} stations×nuits)")
        print("=" * 70)


def _next_date(date_str: str) -> str:
    from datetime import datetime, timedelta
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Comparaison méthodes de descente d'échelle")
    p.add_argument("--config",     default="runs/april2021/config.yml")
    p.add_argument("--stat-out",   default=None, help="NetCDF sortie stat downscaling")
    p.add_argument("--dl-out",     default=None, help="NetCDF sortie DL downscaling (optionnel)")
    p.add_argument("--pmap-dir",   default=None, help="Répertoire runs PMAP (output/pmap/)")
    p.add_argument("--obs-synop",  default=None, help="CSV observations SYNOP")
    p.add_argument("--out-dir",    default="output/verification/")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg_path = Path(__file__).resolve().parents[2] / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    comp = DownscalingComparison(
        cfg=cfg,
        stat_output=args.stat_out,
        dl_output=args.dl_out,
        pmap_dir=args.pmap_dir,
        obs_synop_csv=args.obs_synop,
    )

    comp.run()
    comp.print_summary()
    comp.save_csv(f"{args.out_dir}/scores_april2021.csv")


if __name__ == "__main__":
    main()
