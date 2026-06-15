#!/usr/bin/env python3
"""Download CERRA + CERRA-Land for the recalibration pipeline (Option C of
parametric_insurance issue #14).

Reads its configuration entirely from environment variables, designed to be
invoked by the RunPod orchestrator (`scripts/recalibration_pipeline.sh`), which
is itself launched by `parametric_insurance/scripts/runpod_launch.py
--with-cerra-download`.

Required env vars
-----------------

    CDSAPI_URL              # forwarded from the host shell
    CDSAPI_KEY              # forwarded from the host shell
    CERRA_START             # ISO date, e.g. "2022-01-01"
    CERRA_END               # ISO date, e.g. "2026-05-31" (inclusive year boundaries)
    CERRA_BBOX_LAT_MIN      # float
    CERRA_BBOX_LAT_MAX      # float
    CERRA_BBOX_LON_MIN      # float
    CERRA_BBOX_LON_MAX      # float
    CERRA_OUT_ATM           # output dir for atmospheric NetCDF (e.g. /workspace/data/cerra/)
    CERRA_OUT_LAND          # output dir for CERRA-Land NetCDF (e.g. /workspace/data/cerra_land/)

Output
------

NetCDF files, one per (variable × year), schema matching the existing
`parametric_insurance/backtest/scripts/convert_nc_to_zarr.py` normalization.

CERRA atmospheric variables:  2m_temperature, 2m_dewpoint_temperature,
                              10m_wind_speed, total_precipitation
CERRA-Land variables:         skin_temperature, total_precipitation

Failure mode
------------

Hard-exits non-zero if any required env var is missing or empty — avoids
burning GPU time on a doomed run. Lets the caller (entrypoint.sh) abort
before training.
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

# CERRA atmospheric — schéma figé (cf. backtest/config_drome_ardeche.yaml)
ATM_DATASET = "reanalysis-cerra-single-levels"
ATM_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_wind_speed",
    "total_precipitation",
]
ATM_TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

# CERRA-Land — surface, forecast leadtime 1-3h depuis 4 init/jour (12 timesteps/jour)
LAND_DATASET = "reanalysis-cerra-land"
LAND_VARIABLES = ["skin_temperature", "total_precipitation"]
LAND_INIT_TIMES = ["00:00", "06:00", "12:00", "18:00"]
LAND_LEADTIME_HOURS = ["1", "2", "3"]


def _require_env() -> dict[str, str]:
    """Return validated env dict or hard-exit non-zero."""
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
    """Write the cdsapi credentials file from env (overrides any pre-existing one)."""
    path = Path.home() / ".cdsapirc"
    path.write_text(f"url: {url}\nkey: {key}\n", encoding="utf-8")
    path.chmod(0o600)


def _area(env: dict[str, str]) -> list[float]:
    """CDS API uses [North, West, South, East]."""
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


def _download_atm(client, env: dict[str, str], year: int, out: Path) -> Path:
    target = out / f"cerra_atm_{year}.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[atm] skip (exists): {target}")
        return target
    print(f"[atm] requesting {year} → {target}")
    client.retrieve(
        ATM_DATASET,
        {
            "variable": ATM_VARIABLES,
            "level_type": "surface_or_atmosphere",
            "data_type": "reanalysis",
            "product_type": "analysis",
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ATM_TIMES,
            "area": _area(env),
            "format": "netcdf",
        },
        str(target),
    )
    return target


def _download_land(client, env: dict[str, str], year: int, out: Path) -> Path:
    target = out / f"cerra_land_{year}.nc"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[land] skip (exists): {target}")
        return target
    print(f"[land] requesting {year} → {target}")
    client.retrieve(
        LAND_DATASET,
        {
            "variable": LAND_VARIABLES,
            "level_type": "surface",
            "product_type": "forecast",
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": LAND_INIT_TIMES,
            "leadtime_hour": LAND_LEADTIME_HOURS,
            "area": _area(env),
            "format": "netcdf",
        },
        str(target),
    )
    return target


def main() -> int:
    env = _require_env()
    _write_cdsapirc(env["CDSAPI_URL"], env["CDSAPI_KEY"])

    # Import after env check so the failure happens before any heavy import.
    import cdsapi

    client = cdsapi.Client()

    out_atm = Path(env["CERRA_OUT_ATM"])
    out_land = Path(env["CERRA_OUT_LAND"])
    years = _years(env)
    print(f"CERRA download: years={years} bbox={_area(env)}")
    print(f"  atm  → {out_atm}")
    print(f"  land → {out_land}")

    for y in years:
        _download_atm(client, env, y, out_atm)
        _download_land(client, env, y, out_land)

    print("CERRA download done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
