#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "xarray>=2023.1",
#   "netCDF4>=1.6",
#   "pyyaml>=6.0",
#   "downscaling",
# ]
# ///
"""
Détection automatique des nuits froides dans les sorties de descente d'échelle.

Critères de déclenchement d'un run PMAP :
  1. Tmin nocturne (18h–06h UTC) < seuil (défaut −2 °C)
     sur au moins pixel_fraction_min des pixels du domaine
  2. GDD cumulés depuis le 1er janvier ≥ gdd_threshold
     (débourrement estimé : vignes à risque)

Sorties
-------
  cold_nights.json : liste des nuits critiques avec métadonnées (dates, sévérité,
                     Tmin spatiale, fraction de pixels déclencheurs)

Usage
-----
    python runs/scripts/detect_cold_nights.py \\
        --stat-out  output/stat/stat_downscaled_april2021.nc \\
        --config    runs/april2021/config.yml \\
        --out       runs/april2021/cold_nights.json \\
        --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.indices import growing_degree_days

log = logging.getLogger(__name__)

_K0 = 273.15


# ---------------------------------------------------------------------------
# Détection
# ---------------------------------------------------------------------------

def detect_cold_nights(
    ds: xr.Dataset,
    tmin_threshold_c: float = -2.0,
    pixel_fraction_min: float = 0.10,
    gdd_base_c: float = 5.0,
    gdd_threshold: float = 50.0,
    start_h_utc: int = 18,
    end_h_utc: int = 6,
) -> list[dict]:
    """
    Identifie les nuits où le gel post-débourrement dépasse le seuil.

    Parameters
    ----------
    ds:
        Dataset avec au minimum 't2m' horaire (K) sur la période d'intérêt.
        Optionnel : 'tmin' journalière pour le calcul GDD.
    tmin_threshold_c:
        Seuil température minimale (°C) pour déclencher un run PMAP.
    pixel_fraction_min:
        Fraction minimale de pixels sous le seuil pour valider la nuit.
    gdd_base_c:
        Température de base GDD (°C).
    gdd_threshold:
        GDD cumulés marquant le débourrement (Eichhorn-Lorenz stade B).
    start_h_utc, end_h_utc:
        Fenêtre nocturne UTC. end_h_utc est sur le jour suivant si < start_h_utc.

    Returns
    -------
    Liste de dicts, un par nuit critique, avec :
        date, label, start_utc, end_utc, tmin_p10, tmin_p50, tmin_p90,
        pixel_fraction_below_threshold, severity, run_pmap
    """
    if "t2m" not in ds:
        raise KeyError("Variable 't2m' manquante dans le dataset.")

    t2m = ds["t2m"]
    t2m_c = t2m - _K0

    # Calcul GDD depuis le 1er janvier
    year = int(t2m.time.dt.year[0].values)
    t2m_daily = t2m.resample(time="1D").min()  # Tmin journalière pour GDD
    tmin_daily = t2m_daily - _K0
    gdd_daily = (tmin_daily - gdd_base_c).clip(min=0.0)
    gdd_cum = gdd_daily.cumsum("time")

    # Date de débourrement : premier jour où GDD cumulé ≥ seuil
    # Calculé pixel par pixel → on prend la médiane spatiale
    deb_mask = gdd_cum >= gdd_threshold
    if not deb_mask.any():
        log.warning(f"GDD cumulé < {gdd_threshold} sur toute la période : "
                    "aucune nuit post-débourrement détectée.")
        deb_doy = 1
    else:
        deb_doy = int(deb_mask.argmax("time").median().values)
        deb_date = t2m_daily.time.values[deb_doy]
        log.info(f"Débourrement estimé : jour {deb_doy} ({np.datetime_as_string(deb_date, unit='D')})")

    # Identifier les nuits à partir du débourrement
    dates_in_ds = np.unique(t2m.time.dt.date.values)
    cold_nights = []

    for d in dates_in_ds[deb_doy:]:
        date_str = str(d)
        d_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        # Fenêtre nocturne : 18h J → 06h J+1
        t_start = d_dt.replace(hour=start_h_utc)
        t_end = (d_dt + timedelta(days=1)).replace(hour=end_h_utc)

        t_start_np = np.datetime64(t_start.replace(tzinfo=None))
        t_end_np = np.datetime64(t_end.replace(tzinfo=None))

        night_mask = (t2m.time >= t_start_np) & (t2m.time <= t_end_np)
        if not night_mask.any():
            continue

        t_night = t2m_c.where(night_mask, drop=True)
        tmin_night = t_night.min("time")

        below = (tmin_night < tmin_threshold_c)
        n_total = tmin_night.size
        n_below = int(below.sum().values)
        frac = n_below / n_total if n_total > 0 else 0.0

        if frac < pixel_fraction_min:
            log.debug(f"{date_str}: fraction {frac:.1%} < {pixel_fraction_min:.1%} → ignoré")
            continue

        tmin_vals = tmin_night.values.ravel()
        tmin_vals = tmin_vals[np.isfinite(tmin_vals)]

        p10 = float(np.percentile(tmin_vals, 10))
        p50 = float(np.percentile(tmin_vals, 50))
        p90 = float(np.percentile(tmin_vals, 90))

        # Sévérité
        if p10 < -5.0:
            severity = "catastrophic"
        elif p10 < -3.0:
            severity = "major"
        elif p10 < -1.5:
            severity = "significant"
        else:
            severity = "moderate"

        night_info = {
            "date":        date_str,
            "label":       f"nuit_{date_str.replace('-', '')}",
            "start_utc":   t_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_utc":     t_end.strftime("%Y-%m-%d %H:%M:%S"),
            "tmin_p10_c":  round(p10, 2),
            "tmin_p50_c":  round(p50, 2),
            "tmin_p90_c":  round(p90, 2),
            "pixel_fraction_below_threshold": round(frac, 4),
            "severity":    severity,
            "run_pmap":    True,
        }
        cold_nights.append(night_info)
        log.info(
            f"Nuit critique : {date_str}  Tmin P10={p10:.1f}°C  "
            f"fraction={frac:.0%}  sévérité={severity}"
        )

    return cold_nights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Détection des nuits froides post-débourrement")
    p.add_argument("--stat-out",    required=True, help="NetCDF sortie descente d'échelle stat")
    p.add_argument("--config",      default="runs/april2021/config.yml")
    p.add_argument("--threshold",   type=float, default=None,
                   help="Seuil Tmin (°C). Défaut depuis config.")
    p.add_argument("--gdd-thresh",  type=float, default=None,
                   help="GDD débourrement. Défaut depuis config.")
    p.add_argument("--out",         required=True, help="Fichier JSON de sortie")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    det_cfg = cfg.get("detection", {})
    tmin_thresh = args.threshold or det_cfg.get("tmin_threshold_c", -2.0)
    gdd_thresh  = args.gdd_thresh or det_cfg.get("gdd_threshold", 50.0)
    gdd_base    = det_cfg.get("gdd_base_c", 5.0)
    frac_min    = det_cfg.get("pixel_fraction_min", 0.10)
    start_h     = det_cfg.get("nocturnal_window", {}).get("start_h", 18)
    end_h       = det_cfg.get("nocturnal_window", {}).get("end_h", 6)

    log.info(f"Chargement {args.stat_out}…")
    ds = xr.open_dataset(args.stat_out, engine="netcdf4")

    cold_nights = detect_cold_nights(
        ds,
        tmin_threshold_c=tmin_thresh,
        pixel_fraction_min=frac_min,
        gdd_base_c=gdd_base,
        gdd_threshold=gdd_thresh,
        start_h_utc=start_h,
        end_h_utc=end_h,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "detection_params": {
                    "tmin_threshold_c": tmin_thresh,
                    "gdd_threshold": gdd_thresh,
                    "gdd_base_c": gdd_base,
                    "pixel_fraction_min": frac_min,
                    "source": args.stat_out,
                },
                "cold_nights": cold_nights,
            },
            f,
            indent=2,
        )
    log.info(f"{len(cold_nights)} nuit(s) critique(s) → {out_path}")
    for n in cold_nights:
        print(f"  [{n['severity']:12s}] {n['date']}  Tmin P10={n['tmin_p10_c']:+.1f}°C  "
              f"frac={n['pixel_fraction_below_threshold']:.0%}")


if __name__ == "__main__":
    main()
