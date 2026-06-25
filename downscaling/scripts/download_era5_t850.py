#!/usr/bin/env python3
"""Download ERA5 pressure-level T850 for inversion proxy.

C5.4 — Sous-objectif B (régime enrichi). Permet de calculer la "force
d'inversion" = T850 - T2m, qui distingue les nuits couvertes avec cold pool
sous inversion (R4a, gel humide candidat) des nuits couvertes sans inversion
(R4b, doux).

Variables : temperature à 850 hPa seulement.
Bbox : 42-47°N × 0-8°E (cohérent avec synoptic).
Période : fév-mai par année cible.

Usage
-----

    uv run python -m downscaling.scripts.download_era5_t850 \
      --years 2022 2023 2024 2025 2026 \
      --months 02 03 04 05 \
      --out /tmp/karpos_synoptic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("download_era5_t850")

DATASET = "reanalysis-era5-pressure-levels"
TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
GRID = [0.25, 0.25]
AREA = [47.0, 0.0, 42.0, 8.0]


def _download(client, year: int, months: list[str], out: Path) -> Path:
    target = out / f"era5_t850_{year}.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        log.info("%d: skip (exists) %s", year, target)
        return target
    log.info("%d: requesting %d months → %s", year, len(months), target)
    client.retrieve(
        DATASET,
        {
            "variable": ["temperature"],
            "pressure_level": ["850"],
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
    p = argparse.ArgumentParser(description="Download ERA5 T850 for inversion proxy")
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
