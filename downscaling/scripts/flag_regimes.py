#!/usr/bin/env python3
"""Classify each frost-flo night into a synoptic regime (rule-based on ERA5).

C5.2 (issue maurinl26/karpos-downscaling#TBD). Sortie : un CSV par année avec un label
de régime par nuit + les features synoptiques. Le label est ensuite consommé
par `analyze_karpos_slr.py --regimes-csv` pour stratifier POD/FAR.

Cinq régimes (rule-based, médiane spatiale sur bbox Drôme, fenêtre nuit 18-09 UTC) :

- **R1 Radiatif**         : wind10m < 2.5 m/s · tcc < 0.30 · MSLP > 1015 hPa
                            → gel rayonnement, cas Baronnies typique
- **R2 Advectif N/NE**    : wind10m > 4 m/s · dir ∈ [290°, 50°] · MSLP > 1010 hPa
                            → gel d'advection, masses Sibérie/Norvège
- **R3 Cyclonic**         : wind10m > 5 m/s · tcc > 0.60 · MSLP < 1015 hPa
                            → perturbation, pas de gel attendu
- **R4 Anticyclonic doux** : MSLP > 1020 hPa · wind10m < 3 m/s · tcc > 0.30
                            → blocage, pas radiatif, peu de gel
- **R0 Mixed/Transition** : le reste

Sortie CSV par année :

    date,regime,wind_med,wind_dir_med,tcc_med,mslp_med,dewpoint_dep_med

Usage
-----

    uv run python -m downscaling.scripts.flag_regimes \
      --era5-dir /tmp/karpos_synoptic \
      --years 2022 2023 2024 2025 2026 \
      --bbox-lat 44.0 45.5 --bbox-lon 4.0 5.5 \
      --out /tmp/karpos_synoptic/regimes
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger("flag_regimes")


def _circular_median_dir(directions_deg: np.ndarray) -> float:
    """Median of wind directions (degrees), accounts for circular wrap."""
    finite = directions_deg[np.isfinite(directions_deg)]
    if finite.size == 0:
        return np.nan
    # Vector mean
    rad = np.deg2rad(finite)
    u_mean = np.nanmean(np.sin(rad))
    v_mean = np.nanmean(np.cos(rad))
    dir_mean = np.rad2deg(np.arctan2(u_mean, v_mean)) % 360.0
    return float(dir_mean)


def _normalize_era5(ds: xr.Dataset) -> xr.Dataset:
    """Normalize ERA5 NetCDF : valid_time→time, lat/lon naming, time decoding."""
    if (
        "valid_time" in ds.dims
        and "time" not in ds.dims
        or "valid_time" in ds.coords
        and "time" not in ds.coords
    ):
        ds = ds.rename({"valid_time": "time"})
    if ds["time"].dtype.kind != "M":
        ref = pd.Timestamp("1900-01-01")
        ds = ds.assign_coords(time=ref + pd.to_timedelta(ds["time"].values, unit="h"))
    # lat/lon naming
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds


def _night_window(ds: xr.Dataset, d: date) -> xr.Dataset:
    """Select ERA5 timesteps for the night of date `d` : 18h UTC (d-1) → 09h UTC (d)."""
    start = pd.Timestamp(d) - pd.Timedelta("6h")  # 18h day before
    end = pd.Timestamp(d) + pd.Timedelta("9h")  # 09h current day
    return ds.sel(time=slice(start, end))


# Candidats de noms de variables NetCDF CERRA (le nom court diffère du nom CDS).
_CERRA_VARS = {
    "wind_speed": ("si10", "ws10", "10si", "wind_speed", "10m_wind_speed"),
    "wind_dir": ("wdir10", "wd10", "10wdir", "wind_direction", "10m_wind_direction"),
    "mslp": ("msl", "prmsl", "mean_sea_level_pressure", "msror"),
    "tcc": ("tcc", "total_cloud_cover", "tciwv"),
    "rh": ("r2", "r", "2r", "relative_humidity", "hur"),
    "t2m": ("t2m", "2t", "t", "temperature", "air_temperature"),
}


def _resolve_var(ds: xr.Dataset, key: str) -> xr.DataArray:
    """Trouve une variable CERRA par sa liste de noms candidats, sinon erreur claire."""
    for cand in _CERRA_VARS[key]:
        if cand in ds.data_vars or cand in ds:
            return ds[cand]
    raise ValueError(
        f"CERRA : aucune variable '{key}' trouvée (candidats {_CERRA_VARS[key]}). "
        f"Variables présentes : {list(ds.data_vars)}"
    )


def _td_from_rh(t2m_c: np.ndarray, rh_frac: np.ndarray) -> np.ndarray:
    """Point de rosée (°C) depuis T2m (°C) et RH (fraction 0-1), Magnus inverse.

    e_s(T) = 6.112·exp(17.62·T/(243.12+T)) [hPa] ; e = RH·e_s(T) ;
    Td = 243.12·ln(e/6.112) / (17.62 − ln(e/6.112)). Valide -45..+60 °C.
    """
    es_t = 6.112 * np.exp(17.62 * t2m_c / (243.12 + t2m_c))
    e = np.clip(rh_frac, 1e-6, 1.0) * es_t
    gamma = np.log(e / 6.112)
    return 243.12 * gamma / (17.62 - gamma)


def _night_features_cerra(ds_night: xr.Dataset, bbox: dict[str, float]) -> dict[str, float]:
    """Features synoptiques CERRA pour une nuit (médiane bbox + fenêtre).

    CERRA fournit vitesse ET direction de vent directement (pas u/v) et RH (pas
    dewpoint) → Td calculé via Magnus. Unités auto-détectées (K→°C, %→fraction,
    Pa→hPa). Le t2m doit être mergé dans ``ds_night`` (fichier CERRA atm séparé).
    """
    # CERRA grid : lat peut être ascendante ou descendante — on tente les deux.
    sub = ds_night.sel(
        lat=slice(bbox["lat_max"], bbox["lat_min"]),
        lon=slice(bbox["lon_min"], bbox["lon_max"]),
    )
    if sub.sizes.get("lat", 0) == 0:
        sub = ds_night.sel(
            lat=slice(bbox["lat_min"], bbox["lat_max"]),
            lon=slice(bbox["lon_min"], bbox["lon_max"]),
        )
    if sub.sizes.get("time", 0) == 0 or sub.sizes.get("lat", 0) == 0:
        return {}

    wind = _resolve_var(sub, "wind_speed").values
    wdir = _resolve_var(sub, "wind_dir").values
    tcc = _resolve_var(sub, "tcc").values.astype(float)
    if np.nanmedian(tcc) > 1.5:  # % ou oktas → fraction
        tcc = tcc / (100.0 if np.nanmedian(tcc) > 8.5 else 8.0)
    msl = _resolve_var(sub, "mslp").values.astype(float)
    if np.nanmedian(msl) > 2000.0:  # Pa → hPa
        msl = msl / 100.0
    rh = _resolve_var(sub, "rh").values.astype(float)
    rh_frac = rh / 100.0 if np.nanmedian(rh) > 1.5 else rh
    t2m = _resolve_var(sub, "t2m").values.astype(float)
    t2m_c = t2m - 273.15 if np.nanmedian(t2m) > 100.0 else t2m
    td_c = _td_from_rh(t2m_c, rh_frac)
    dewpoint_dep = t2m_c - td_c

    return {
        "wind_med": float(np.nanmedian(wind)),
        "wind_dir_med": _circular_median_dir(wdir),
        "tcc_med": float(np.nanmedian(tcc)),
        "mslp_med": float(np.nanmedian(msl)),
        "dewpoint_dep_med": float(np.nanmedian(dewpoint_dep)),
        "t2m_med": float(np.nanmedian(t2m_c)),
        "rh_med": float(np.nanmedian(rh_frac)),
        "dewpoint_dep_min": float(np.nanmin(dewpoint_dep)),
        "rh_min": float(np.nanmin(rh_frac)),
    }


def _night_features(
    ds_night: xr.Dataset,
    bbox: dict[str, float],
    ds_t850_night: xr.Dataset | None = None,
) -> dict[str, float]:
    """Compute synoptic features for one night (median over bbox + time window).

    Si ds_t850_night fourni, ajoute `inversion_strength_med` = median(T850 - T2m).
    Inversion positive (T850 > T2m) = air froid près du sol sous air plus chaud
    en altitude → favorable au cold pooling sous nuage.
    """
    sub = ds_night.sel(
        lat=slice(bbox["lat_max"], bbox["lat_min"]),  # ERA5 lat descending
        lon=slice(bbox["lon_min"], bbox["lon_max"]),
    )
    if sub.sizes.get("time", 0) == 0 or sub.sizes.get("lat", 0) == 0:
        return {}

    u10 = sub["u10"].values
    v10 = sub["v10"].values
    wind = np.sqrt(u10**2 + v10**2)
    wind_dir = np.rad2deg(np.arctan2(u10, v10)) % 360.0  # 0=N, 90=E

    tcc = sub["tcc"].values  # 0-1
    msl = sub["msl"].values / 100.0  # Pa → hPa
    t2m = sub["t2m"].values
    d2m = sub["d2m"].values
    dewpoint_dep = t2m - d2m  # K (proxy clear-sky)

    # Humidité relative depuis t2m et d2m (formule Magnus, valide -45..+60 °C).
    # e_s(T) = 6.112 * exp(17.62 * T_c / (243.12 + T_c)) en hPa
    t2m_c = t2m - 273.15
    d2m_c = d2m - 273.15
    es_t = 6.112 * np.exp(17.62 * t2m_c / (243.12 + t2m_c))
    es_td = 6.112 * np.exp(17.62 * d2m_c / (243.12 + d2m_c))
    rh = np.clip(es_td / np.where(es_t > 0, es_t, np.nan), 0.0, 1.0)

    features = {
        "wind_med": float(np.nanmedian(wind)),
        "wind_dir_med": _circular_median_dir(wind_dir),
        "tcc_med": float(np.nanmedian(tcc)),
        "mslp_med": float(np.nanmedian(msl)),
        "dewpoint_dep_med": float(np.nanmedian(dewpoint_dep)),
        "t2m_med": float(np.nanmedian(t2m)),
        # Hygrométrie enrichie (issue #5 — variables FiLM hygro)
        "d2m_med": float(np.nanmedian(d2m)),  # Td (K) — borne inf gel
        "rh_med": float(np.nanmedian(rh)),  # humidité relative (0-1)
        "dewpoint_dep_min": float(np.nanmin(dewpoint_dep)),  # plus sec moment de la nuit
        "rh_min": float(np.nanmin(rh)),  # plus sec moment de la nuit
    }

    # Proxy inversion : T850 - T2m, médiane sur la nuit
    if ds_t850_night is not None:
        sub850 = ds_t850_night.sel(
            lat=slice(bbox["lat_max"], bbox["lat_min"]),
            lon=slice(bbox["lon_min"], bbox["lon_max"]),
        )
        if sub850.sizes.get("time", 0) > 0:
            t850 = sub850["t"].values if "t" in sub850 else sub850["temperature"].values
            # Aligne temporellement par broadcast (mêmes timesteps généralement)
            inv = np.nanmedian(t850) - features["t2m_med"]
            features["inversion_strength_med"] = float(inv)

    return features


def _classify(f: dict[str, float]) -> str:
    """Apply rule-based regime classification to feature dict.

    Taxonomie V2 use-case Karpos (gel arboriculture). Cadran 2×2 vent × ciel,
    avec subdivision du quadrant couvert+calme via la **force d'inversion**
    (T850-T2m). C'est là que se cachent les **gels humides sous nuage bas**
    qui font le plus de dégâts en arbo (audit FN 2022-2024).

    - **R1 Radiatif**          : vent ≤ 3.0 m/s · tcc ≤ 0.50 → gel rayonnement
                                  probable (cas Baronnies typique)
    - **R2 Advectif venté**    : vent > 3.0 m/s · tcc ≤ 0.50 → mélange forcé,
                                  gel possible par advection
    - **R3 Couvert venté**     : vent > 3.0 m/s · tcc > 0.50 → perturbé,
                                  gel rare
    - **R4a Cold pool anticyclonique** : vent ≤ 3.0 m/s · tcc > 0.50 ·
                                        MSLP ≥ 1020 hPa → anticyclone fort
                                        + nuage bas, **gel humide candidat**
                                        (cold pool en vallée sous Strato/Sc,
                                        audit FN 2022-2024 montre les gels
                                        ratés R4 ont tous MSLP ≥ 1017 hPa)
    - **R4b Couvert doux**            : vent ≤ 3.0 m/s · tcc > 0.50 ·
                                        MSLP < 1020 hPa → couvert sans
                                        anticyclone, gel rare
    - **R0** : feature manquant (catch-all NaN)

    Note : T850-T2m (téléchargé en parallèle) reste enregistré comme feature
    dans le CSV pour conditioning futur KarposSR (FiLM token), mais pas utilisé
    pour la classification (couche limite < 1500 m mal résolue par T850).
    """
    if not f or any(np.isnan(v) for v in f.values() if not (isinstance(v, float) and np.isnan(v))):
        # Tolère NaN sur inversion_strength_med (optionnel) — vérifié plus bas.
        pass

    required = ["wind_med", "tcc_med", "mslp_med"]
    if not f or any(np.isnan(f.get(k, np.nan)) for k in required):
        return "R0"

    wind_calm = f["wind_med"] <= 3.0
    sky_clear = f["tcc_med"] <= 0.50

    if wind_calm and sky_clear:
        return "R1"
    if not wind_calm and sky_clear:
        return "R2"
    if not wind_calm and not sky_clear:
        return "R3"

    # Quadrant couvert + calme : split par MSLP (anticyclone fort = cold pool
    # candidat). Critère MSLP ≥ 1020 hPa validé par audit FN 2022-2024 :
    # les 4 gels ratés R4 ont MSLP ∈ [1017, 1031] hPa.
    if f["mslp_med"] >= 1020.0:
        return "R4a"  # cold pool anticyclonique (gel humide candidat)
    return "R4b"  # couvert doux sans anticyclone fort


def _dates_for_year(ds: xr.Dataset, year: int, months: tuple[int, ...]) -> list[date]:
    """Unique morning dates in the requested year/months."""
    times = pd.DatetimeIndex(ds["time"].values)
    mask = (times.year == year) & np.isin(times.month, list(months))
    dates = pd.Index(times[mask].normalize()).unique().sort_values()
    return [d.date() for d in dates]


def main() -> int:
    p = argparse.ArgumentParser(description="Classify frost-flo nights into synoptic regimes")
    p.add_argument(
        "--source",
        choices=["era5", "cerra"],
        default="era5",
        help="Réanalyse source du régime. CERRA pour conditionner un downscaling CERRA "
        "(cohérence entrée/conditionnement). Défaut era5 (rétro-compat).",
    )
    p.add_argument(
        "--era5-dir",
        type=Path,
        default=None,
        help="[source=era5] Directory containing era5_synoptic_<year>.nc files",
    )
    p.add_argument(
        "--cerra-synoptic",
        type=Path,
        default=None,
        help="[source=cerra] NetCDF synoptique CERRA (si10, wdir10, msl, tcc, r2)",
    )
    p.add_argument(
        "--cerra-t2m",
        type=Path,
        default=None,
        help="[source=cerra] NetCDF CERRA atm t2m (pour calculer Td depuis RH)",
    )
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="Months of interest (default: 02 03 04 05 = flo abricot)",
    )
    p.add_argument(
        "--bbox-lat",
        type=float,
        nargs=2,
        default=[44.0, 45.5],
        help="Latitude bounds (min max), default Drôme = 44.0 45.5",
    )
    p.add_argument(
        "--bbox-lon",
        type=float,
        nargs=2,
        default=[4.0, 5.5],
        help="Longitude bounds (min max), default Drôme = 4.0 5.5",
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory (one CSV per year)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.source == "era5" and args.era5_dir is None:
        p.error("--era5-dir requis avec --source era5")
    if args.source == "cerra" and (args.cerra_synoptic is None or args.cerra_t2m is None):
        p.error("--cerra-synoptic ET --cerra-t2m requis avec --source cerra")

    args.out.mkdir(parents=True, exist_ok=True)

    bbox = {
        "lat_min": min(args.bbox_lat),
        "lat_max": max(args.bbox_lat),
        "lon_min": min(args.bbox_lon),
        "lon_max": max(args.bbox_lon),
    }
    log.info("Bbox: %s | source=%s", bbox, args.source)

    # CERRA : un seul NetCDF synoptique + t2m (fichier atm séparé), mergés une fois.
    cerra_ds: xr.Dataset | None = None
    if args.source == "cerra":
        ds_syn = _normalize_era5(xr.open_dataset(args.cerra_synoptic))
        ds_t2m_raw = _normalize_era5(xr.open_dataset(args.cerra_t2m))
        t2m_da = _resolve_var(ds_t2m_raw, "t2m").rename("t2m")
        cerra_ds = xr.merge([ds_syn, t2m_da], join="inner", compat="override")
        log.info("CERRA synoptique+t2m mergés : vars=%s", list(cerra_ds.data_vars))

    summary_per_year: dict[int, dict[str, int]] = {}
    all_rows: list[dict] = []

    for year in args.years:
        if args.source == "cerra":
            ds = cerra_ds
            ds_t850 = None
            log.info("--- year %d (cerra) ---", year)
        else:
            path = args.era5_dir / f"era5_synoptic_{year}.nc"
            if not path.exists():
                log.warning("%d: %s manquant, skip", year, path)
                continue
            log.info("--- year %d ---", year)
            ds = _normalize_era5(xr.open_dataset(path))

            # T850 optionnel pour proxy inversion
            path_t850 = args.era5_dir / f"era5_t850_{year}.nc"
            ds_t850 = None
            if path_t850.exists():
                ds_t850 = _normalize_era5(xr.open_dataset(path_t850))
                log.info("  T850 chargé : %s", path_t850.name)
            else:
                log.info("  T850 absent → R4 reste indivisé (fallback)")

        dates = _dates_for_year(ds, year, tuple(args.months))
        log.info("  %d nuits à classifier", len(dates))

        rows: list[dict] = []
        regime_counts: dict[str, int] = {
            "R0": 0,
            "R1": 0,
            "R2": 0,
            "R3": 0,
            "R4": 0,  # fallback si T850 absent
            "R4a": 0,  # cold pool sous inversion + nuage bas (gel humide)
            "R4b": 0,  # couvert doux sans inversion
        }
        for d in dates:
            ds_night = _night_window(ds, d)
            if args.source == "cerra":
                feats = _night_features_cerra(ds_night, bbox)
            else:
                ds_t850_night = _night_window(ds_t850, d) if ds_t850 is not None else None
                feats = _night_features(ds_night, bbox, ds_t850_night=ds_t850_night)
            regime = _classify(feats)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            row = {"date": d.isoformat(), "regime": regime, **feats}
            rows.append(row)
            all_rows.append({"year": year, **row})

        df = pd.DataFrame(rows)
        out_csv = args.out / f"regimes_{year}.csv"
        df.to_csv(out_csv, index=False)
        log.info("  → %s", out_csv)
        log.info("  régimes : %s", regime_counts)
        summary_per_year[year] = regime_counts

    # Synthèse globale
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        out_all = args.out / "regimes_all.csv"
        df_all.to_csv(out_all, index=False)
        log.info("Synthèse globale : %s (%d nuits)", out_all, len(df_all))

        log.info("Bilan régimes par année :")
        for year, counts in summary_per_year.items():
            total = sum(counts.values())
            line = f"  {year}: total={total:3d}"
            for r in ("R1", "R2", "R3", "R4a", "R4b", "R4", "R0"):
                n = counts.get(r, 0)
                if n == 0:
                    continue
                pct = (100.0 * n / total) if total else 0.0
                line += f" · {r}={n:3d} ({pct:4.1f}%)"
            log.info(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
