#!/usr/bin/env python3
"""Download ERA5 single-level synoptic variables for regime classification.

C5.1 (issue maurinl26/karpos-downscaling#TBD). Pulls the variables needed by
`flag_regimes.py` to classify each frost-flo night into a synoptic regime
(radiative, advective N/NE, cyclonic, anticyclonic doux, mixed).

Why ERA5 not CERRA :
- Regimes are synoptic-scale (> 100 km) → ERA5 0.25° suffit.
- Cheap : ~30-50 MB par année (vs CERRA atm enrichi ≫ 1 GB).
- N'interfère pas avec le pipeline CERRA Lot B existant.

Variables :
- 2m_temperature                  → T2m
- 2m_dewpoint_temperature          → Td2m (clear-sky proxy)
- mean_sea_level_pressure          → MSLP (cyclonic vs anticyclonic)
- 10m_u_component_of_wind / v      → wind speed + direction
- total_cloud_cover                → tcc (radiative vs couvert)

Bbox synoptique élargie autour Drôme : 42-47°N, 0-8°E.
Période : fév-mai (frost-flo abricot) sur les années cibles.
Grid : 0.25° natif ERA5.
Times : 00, 03, 06, 09, 12, 15, 18, 21 UTC (8 / jour).

Usage
-----

    uv run python -m downscaling.scripts.download_era5_synoptic \
      --years 2022 2023 2024 2025 2026 \
      --months 02 03 04 05 \
      --out /tmp/karpos_synoptic

Output : un fichier `era5_synoptic_<year>.nc` par année (~30-50 MB).
Skips existing files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("download_era5_synoptic")

DATASET = "reanalysis-era5-single-levels"
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover",
]
TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
GRID = [0.25, 0.25]
# Bbox synoptique élargie (Lyon → côte d'Azur, Pyrénées → Jura) : capture régimes Z500.
AREA = [47.0, 0.0, 42.0, 8.0]  # [N, W, S, E]


def _download(client, year: int, months: list[str], out: Path) -> Path:
    target = out / f"era5_synoptic_{year}.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        log.info("%d: skip (exists) %s", year, target)
        return target
    log.info("%d: requesting %d months → %s", year, len(months), target)
    client.retrieve(
        DATASET,
        {
            "variable": VARIABLES,
            "product_type": "reanalysis",
            "year": str(year),
            "month": months,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": TIMES,
            "area": AREA,
            "data_format": "netcdf",
            "grid": GRID,
        },
        str(target),
    )
    return target


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download ERA5 synoptic variables for regime classification"
    )
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--months", type=str, nargs="+", default=["02", "03", "04", "05"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    import cdsapi

    client = cdsapi.Client()
    for year in args.years:
        try:
            _download(client, year, args.months, args.out)
        except Exception as exc:
            log.error("%d: download failed (%s)", year, exc)
            continue
    log.info("done. Outputs in %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
