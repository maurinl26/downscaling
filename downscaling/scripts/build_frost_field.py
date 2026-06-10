#!/usr/bin/env python
"""
Builder batch : CERRA 5,5 km → FrostField (contrat OTA de la carte), par millésime.

Remplace la donnée synthétique de `app/lib/frost.ts` par le **vrai Tmin CERRA**. Pour
chaque cellule de la maille CERRA dans la bbox : Tmin saisonnier (pire gel fév-mai) →
sévérité via seuils phénologiques floraison (T10/T90) → payout. Émet le JSON du contrat
`FrostField` (year, bbox, step, source, resolution, cells[]).

Étape 1 du pipeline prod (swap OTA Track A) : badge honnête `source="cerra-5.5km"`. La
calibration parcelle (Track B) viendra par le même contrat, en changeant `source`.

Usage :
  uv run --no-sync python -m downscaling.scripts.build_frost_field \
    --cerra-glob '/Users/loicmaurin/kDrive/karpos_datasets/data/raw/cerra/2m_temperature/*.nc' \
    --region baronnies --bbox 4.8 44.1 5.65 44.7 --year 2025 \
    --out reports_product/frost_fields/frost_baronnies_2025.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import xarray as xr

from downscaling.scripts.frost_eval_core import to_c

# Seuils floraison (abricot, stade le plus exposé) — T10 = activation, T90 = dommages majeurs.
T10_FLOWERING = -2.0
T90_FLOWERING = -4.0
# Fenêtre de sensibilité au gel (floraison → nouaison). On ne compte QUE le gel de cette
# période : le froid profond de février, hors floraison, n'endommage pas le verger.
# (v1 calendaire ; raffinement = stade phéno piloté par GDD, cf. phenology.py / IND-04.)
FLOWER_START = "03-15"
FLOWER_END = "04-30"


def load_cerra(cerra_glob: str | None, cerra_zarr: str | None) -> xr.DataArray:
    """CERRA t2m (time, lat, lon). Source nc mensuels (2022-2026) OU zarr (2015-2021)."""
    if cerra_zarr:
        d = xr.open_zarr(cerra_zarr)["t2m"]
        tdim = "valid_time" if "valid_time" in d.dims else "time"
        return d.rename({tdim: "time"}) if tdim != "time" else d
    parts = []
    for p in sorted(glob.glob(cerra_glob)):
        d = xr.open_dataset(p)["t2m"]
        tdim = "valid_time" if "valid_time" in d.dims else "time"
        parts.append(d.rename({tdim: "time"}))
    da = xr.concat(parts, dim="time").sortby("time")
    return da.drop_vars("expver") if "expver" in da.coords else da


def severity_from_tmin(tmin, t10=T10_FLOWERING, t90=T90_FLOWERING):
    """0 au-dessus de T10, 1 à/sous T90, linéaire entre (sévérité de gel dommageable)."""
    return np.clip((t10 - tmin) / (t10 - t90), 0.0, 1.0)


def payout_from_severity(sev):
    """Même formule que le champ démo (continuité du contrat) : paie au-delà de 50 %."""
    return np.where(sev > 0.5, np.round((sev - 0.5) * 2 * 5000), 0).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description="CERRA → FrostField (contrat carte)")
    ap.add_argument("--cerra-glob", help="nc mensuels 2022-2026 (glob)")
    ap.add_argument("--cerra-zarr", help="zarr CERRA 2015-2021 (store)")
    ap.add_argument("--region", default="baronnies")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t10", type=float, default=T10_FLOWERING)
    ap.add_argument("--t90", type=float, default=T90_FLOWERING)
    ap.add_argument("--flower-start", default=FLOWER_START, help="MM-DD début fenêtre floraison")
    ap.add_argument("--flower-end", default=FLOWER_END, help="MM-DD fin fenêtre floraison")
    args = ap.parse_args()

    if not (args.cerra_glob or args.cerra_zarr):
        raise SystemExit("Fournir --cerra-glob (nc) ou --cerra-zarr (zarr).")
    minlon, minlat, maxlon, maxlat = args.bbox
    da = load_cerra(args.cerra_glob, args.cerra_zarr).load()

    # fenêtre de floraison du millésime (gel dommageable seulement)
    sub = da.sel(time=slice(f"{args.year}-{args.flower_start}", f"{args.year}-{args.flower_end}"))
    if sub.time.size == 0:
        raise SystemExit(f"Aucune donnée CERRA pour la floraison {args.year} "
                         f"({args.flower_start}→{args.flower_end}).")

    # Tmin saisonnier (pire gel) par cellule, °C
    tmin_field = to_c(sub.min("time").values)            # (lat, lon)
    latv, lonv = da.latitude.values, da.longitude.values
    step = float(np.round(np.abs(np.diff(lonv)).mean(), 4))

    cells = []
    for i, la in enumerate(latv):
        if not (minlat <= la <= maxlat):
            continue
        for j, lo in enumerate(lonv):
            if not (minlon <= lo <= maxlon):
                continue
            tmin = float(tmin_field[i, j])
            sev = float(severity_from_tmin(tmin, args.t10, args.t90))
            cells.append({
                "lon": round(float(lo), 4), "lat": round(float(la), 4),
                "severity": round(sev, 4), "tmin": round(tmin, 2),
                "payout": int(payout_from_severity(np.array([sev]))[0]),
            })

    field = {
        "year": args.year,
        "bbox": [minlon, minlat, maxlon, maxlat],
        "step": step,
        "source": "cerra-5.5km",
        "resolution": "5500m",
        "cells": cells,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(field, ensure_ascii=False))

    sev = np.array([c["severity"] for c in cells])
    pay = np.array([c["payout"] for c in cells])
    tmn = np.array([c["tmin"] for c in cells])
    print(f"✓ {out}")
    print(f"  région {args.region} {args.year} : {len(cells)} cellules ({step}° ≈ {step*111:.1f} km)")
    print(f"  Tmin saison : min {tmn.min():.1f}°C / médian {np.median(tmn):.1f}°C")
    print(f"  sévérité médiane {np.median(sev):.2f} | cellules déclenchées {(pay>0).sum()} | payout total {pay.sum()} €")


if __name__ == "__main__":
    main()
