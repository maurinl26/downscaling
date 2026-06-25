#!/usr/bin/env python3
"""Download CERRA + CERRA-Land for the recalibration pipeline — frost-window only.

Env-driven (cf. parametric_insurance/scripts/runpod_launch.py
--with-cerra-download). Restricted to t2m + skin_temperature only, frost-flo
window (Feb-April) per year, chunked month-by-month to stay under CDS cost
limits. Uses ``"grid": [0.05, 0.05]`` to bypass MARS LCC-crop unsupported
error (croppedRepresentation not implemented for the native CERRA grid).

Required env vars
-----------------

    CDSAPI_URL, CDSAPI_KEY
    CERRA_START, CERRA_END                          # ISO dates (year resolution)
    CERRA_BBOX_LAT_MIN/MAX, CERRA_BBOX_LON_MIN/MAX  # WGS84 bbox
    CERRA_OUT_ATM, CERRA_OUT_LAND                   # NetCDF output dirs

Output
------

NetCDF files, one per (dataset × year × month). Naming:
    cerra_atm_<year>_<MM>.nc
    cerra_land_<year>_<MM>.nc

Failure mode
------------

Hard-exits non-zero if any required env var is missing or empty. Each CDS
request that fails leaves the file absent (re-running the script will retry).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REQUIRED_ENV = (
    "CDSAPI_URL",
    "CDSAPI_KEY",
    "CERRA_START",
    "CERRA_END",
    "CERRA_BBOX_LAT_MIN",
    "CERRA_BBOX_LAT_MAX",
    "CERRA_BBOX_LON_MIN",
    "CERRA_BBOX_LON_MAX",
    "CERRA_OUT_ATM",
    "CERRA_OUT_LAND",
)

# CERRA atmospheric — t2m only (frost calibration use case).
ATM_DATASET = "reanalysis-cerra-single-levels"
ATM_VARIABLES = ["2m_temperature"]
ATM_TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

# CERRA-Land — skin_temperature only, 4 init × 3 leadtimes = 12 timesteps/day.
LAND_DATASET = "reanalysis-cerra-land"
LAND_VARIABLES = ["skin_temperature"]
LAND_INIT_TIMES = ["00:00", "06:00", "12:00", "18:00"]
LAND_LEADTIME_HOURS = ["1", "2", "3"]

# Frost-flo window for apricot Baronnies (BBCH 53→69) — Feb-April.
FROST_MONTHS = ["02", "03", "04"]

# 0.05° ≈ 5 km regular grid (matches CERRA native resolution after regridding).
# Required to bypass MARS "croppedRepresentation() not implemented" on the
# native LCC grid when "area" is set.
GRID = [0.05, 0.05]


def _require_env() -> dict[str, str]:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print(
            f"FATAL: missing CERRA download env vars: {missing}. "
            "Inspect runpod_launch.py --with-cerra-download wiring.",
            file=sys.stderr,
        )
        sys.exit(1)
    return {k: os.environ[k] for k in REQUIRED_ENV}


def _write_cdsapirc(url: str, key: str) -> None:
    path = Path.home() / ".cdsapirc"
    path.write_text(f"url: {url}\nkey: {key}\n", encoding="utf-8")
    path.chmod(0o600)


def _area(env: dict[str, str]) -> list[float]:
    return [
        float(env["CERRA_BBOX_LAT_MAX"]),
        float(env["CERRA_BBOX_LON_MIN"]),
        float(env["CERRA_BBOX_LAT_MIN"]),
        float(env["CERRA_BBOX_LON_MAX"]),
    ]


def _years(env: dict[str, str]) -> list[int]:
    from datetime import date as _date

    start = _date.fromisoformat(env["CERRA_START"]).year
    end = _date.fromisoformat(env["CERRA_END"]).year
    return list(range(start, end + 1))


def _download_atm(client, env: dict[str, str], year: int, month: str, out: Path) -> Path:
    target = out / f"cerra_atm_{year}_{month}.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[atm] skip (exists): {target}")
        return target
    print(f"[atm] requesting {year}-{month} → {target}")
    client.retrieve(
        ATM_DATASET,
        {
            "variable": ATM_VARIABLES,
            "level_type": "surface_or_atmosphere",
            "data_type": "reanalysis",
            "product_type": "analysis",
            "year": str(year),
            "month": [month],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ATM_TIMES,
            "area": _area(env),
            "data_format": "netcdf",
            "grid": GRID,
        },
        str(target),
    )
    return target


def _download_land(client, env: dict[str, str], year: int, month: str, out: Path) -> Path:
    target = out / f"cerra_land_{year}_{month}.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[land] skip (exists): {target}")
        return target
    print(f"[land] requesting {year}-{month} → {target}")
    client.retrieve(
        LAND_DATASET,
        {
            "variable": LAND_VARIABLES,
            "level_type": "surface",
            "product_type": "forecast",
            "year": str(year),
            "month": [month],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": LAND_INIT_TIMES,
            "leadtime_hour": LAND_LEADTIME_HOURS,
            "area": _area(env),
            "data_format": "netcdf",
            "grid": GRID,
        },
        str(target),
    )
    return target


def main() -> int:
    env = _require_env()
    _write_cdsapirc(env["CDSAPI_URL"], env["CDSAPI_KEY"])

    import cdsapi

    client = cdsapi.Client()

    out_atm = Path(env["CERRA_OUT_ATM"])
    out_land = Path(env["CERRA_OUT_LAND"])
    years = _years(env)
    print(
        f"CERRA frost-window download: years={years} months={FROST_MONTHS} "
        f"bbox={_area(env)} grid={GRID}"
    )
    print(f"  atm  → {out_atm}")
    print(f"  land → {out_land}")

    # NB: l'orographie CERRA (time-invariant) est gérée hors de ce script :
    # downloadée une fois via backtest/scripts/download_cerra.py --variable orography
    # depuis parametric_insurance, puis upload S3 Scaleway ; le pod la lit via
    # recalibrate_statistical --cerra-orog s3://karpos-backtest-data/...

    for y in years:
        for m in FROST_MONTHS:
            _download_atm(client, env, y, m, out_atm)
            _download_land(client, env, y, m, out_land)

    print("CERRA frost-window download done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
