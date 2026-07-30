"""Chargement des observations Sencrop (réseau de capteurs agricoles de parcelle).

Sencrop expose des stations météo in situ (température, humidité, vent…). Pour la
calibration locale (étage C — cf. ``docs/architecture.md``), on ne retient que
**température + position + altitude**, dans le même conteneur que Netatmo
(:data:`StationObs`), afin de partager QC, agrégation Tmin nocturne et fine-tuning
sparse. Sencrop et Netatmo jouent donc le **même rôle** (référence terrain).

Schéma cible — pivot intermédiaire avant ``dataframe_to_station_obs`` :
``station_id, lat, lon, elevation_m, timestamp, t_celsius``.

Deux formats d'entrée supportés
-------------------------------

1. **Bulk Spark partitionné** (canonical, depuis 2026-06-15) — passer un *root*
   de type ``Path`` ou URI ``s3://…`` :

   .. code-block:: text

       <root>/
       ├── 2021.csv/   ← Spark output DIRECTORY, not a file
       │   └── part-*.csv
       ├── 2022.csv/   ← idem
       ├── …
       └── stations_integrated.csv   ← catalogue de référence (REFERENCE)

   Schéma du time-series : ``station_id, timestamp, temperature,
   temperature_source, humidity, humidity_source``.

   Deux points load-bearing, vérifiés 2026-06-15 :

   - ``station_id`` du bulk == ``bucket_id`` du catalogue, **PAS** ``device_id``.
     Plaisians-1 (device 38133) → bucket 22971 ; Plaisians-2 (device 51334) →
     bucket 24529.
   - On filtre ``temperature_source == 'station'`` (élimine l'imputation grid
     de Sencrop, sinon le biais est contaminé).

2. **Fichier CSV/Parquet single-date** (back-compat) — passer un chemin vers
   un seul fichier au schéma ad-hoc ``SENCROP_COLUMNS`` mappable via
   ``columns=``.

Pour les nouveaux entraînements (Lot B, KarposSR) la forme bulk est obligatoire ;
la forme single-file est conservée pour les fixtures de tests existantes.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

import fsspec
import pandas as pd

# Réexport pour l'agrégation Tmin (partagée avec Netatmo).
from .netatmo_qc import tmin_nocturnal  # noqa: F401
from .stations import StationObs, dataframe_to_station_obs

# ---------------------------------------------------------------------------
# Constantes — bulk Spark schema (vérifié 2026-06-15)
# ---------------------------------------------------------------------------
AVAILABLE_YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025, 2026)
STATIONS_FILE = "stations_integrated.csv"
PART_GLOB = "part-*.csv"

# Colonnes time-series attendues dans le bulk
TIMESERIES_COLUMNS = (
    "station_id",  # join key → stations_integrated.csv:bucket_id
    "timestamp",  # ISO8601 UTC
    "temperature",  # °C
    "temperature_source",  # {'station', 'grid'}
    "humidity",
    "humidity_source",
)

# Colonnes catalogue stations
STATION_COLUMNS = (
    "device_id",
    "serial",
    "bucket_id",  # join key
    "city",
    "latitude",
    "longitude",
    "altitude_m",
    "activation_date",
    "is_public",
)

# ---------------------------------------------------------------------------
# Back-compat — mapping pour les fixtures single-file
# ---------------------------------------------------------------------------
SENCROP_COLUMNS = {
    "station_id": "device_id",
    "lat": "latitude",
    "lon": "longitude",
    "elevation_m": "altitude",
    "timestamp": "measure_date",
    "t_celsius": "temperature",
}


# ---------------------------------------------------------------------------
# Helpers — URI / glob (mirrors parametric_insurance/backtest/src/datasources/sencrop.py)
# ---------------------------------------------------------------------------
def _is_remote(root: str | Path) -> bool:
    """True si ``root`` est une URI non-locale (``s3://``, ``gs://``…)."""
    if isinstance(root, Path):
        return False
    parsed = urlparse(str(root))
    return bool(parsed.scheme) and parsed.scheme not in ("", "file")


def _join(root: str | Path, *parts: str) -> str:
    """Join URI/path qui préserve les sémantiques distantes."""
    if _is_remote(root):
        head = str(root).rstrip("/")
        return "/".join([head, *parts])
    return str(Path(root).joinpath(*parts))


def _glob(pattern: str) -> list[str]:
    """Glob local + S3 via fsspec ; préserve le schéma."""
    fs, _ = fsspec.url_to_fs(pattern)
    matches = sorted(fs.glob(pattern))
    if _is_remote(pattern):
        protocol = pattern.split("://", 1)[0]
        return [m if "://" in m else f"{protocol}://{m}" for m in matches]
    return matches


def _is_bulk_root(path: str | Path) -> bool:
    """True si ``path`` ressemble à un root bulk (URI distante OU dossier contenant
    ``stations_integrated.csv``)."""
    if _is_remote(path):
        return True
    p = Path(path)
    if p.is_dir():
        return (p / STATIONS_FILE).exists()
    return False


# ---------------------------------------------------------------------------
# Loaders bulk
# ---------------------------------------------------------------------------
def load_stations_catalog(
    root: str | Path,
    bbox: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Lit ``stations_integrated.csv`` et filtre la bbox éventuelle."""
    df = pd.read_csv(_join(root, STATIONS_FILE))
    missing = set(STATION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"stations_integrated.csv missing expected columns: {sorted(missing)}")
    if bbox is not None:
        df = df[
            (df["latitude"] >= bbox["lat_min"])
            & (df["latitude"] <= bbox["lat_max"])
            & (df["longitude"] >= bbox["lon_min"])
            & (df["longitude"] <= bbox["lon_max"])
        ].reset_index(drop=True)
    return df


def _year_partition(root: str | Path, year: int) -> str:
    """Retourne l'unique part-* CSV dans ``<year>.csv/``. Raise si manquant."""
    year_dir = _join(root, f"{year}.csv")
    pattern = _join(year_dir, PART_GLOB)
    parts = _glob(pattern)
    if not parts:
        raise FileNotFoundError(f"No Spark partition under {year_dir} (expected {PART_GLOB})")
    if len(parts) > 1:
        names = [p.rsplit("/", 1)[-1] for p in parts]
        raise RuntimeError(f"Expected one partition under {year_dir}, found {len(parts)}: {names}")
    return parts[0]


def load_timeseries(
    years: Iterable[int],
    root: str | Path,
    *,
    station_only: bool = True,
    bucket_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Charge les partitions Spark des années demandées.

    Args:
        years: années à lire (sous-ensemble de :data:`AVAILABLE_YEARS`).
        root: racine du bulk (locale ou ``s3://…``).
        station_only: applique le filtre ``temperature_source == 'station'``
            (défaut). **Toujours True pour calibration / bias.**
        bucket_ids: whitelist optionnelle de ``bucket_id``.
    """
    years = list(years)
    bad = [y for y in years if y not in AVAILABLE_YEARS]
    if bad:
        raise ValueError(f"Years {bad} not in available bulk partitions {AVAILABLE_YEARS}")

    frames: list[pd.DataFrame] = []
    for y in years:
        part = _year_partition(root, y)
        df = pd.read_csv(part)
        missing = set(TIMESERIES_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{part} missing expected columns: {sorted(missing)}")
        if station_only:
            df = df[df["temperature_source"] == "station"]
        if bucket_ids is not None:
            df = df[df["station_id"].isin(list(bucket_ids))]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=list(TIMESERIES_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


# Cache process-local des chargements année-entière pour le bulk Sencrop.
# Évite N téléchargements S3 redondants quand `load_sencrop` est appelé en boucle
# par BulkSencropDataset (KarposSR entraînement). Clé : (root, year, bbox_tuple).
_BULK_YEAR_CACHE: dict = {}


def _bbox_key(bbox: dict[str, float] | None) -> tuple | None:
    if bbox is None:
        return None
    return tuple(sorted(bbox.items()))


def _build_normalized_df_from_bulk(
    root: str | Path,
    date: str,
    bbox: dict[str, float] | None,
) -> pd.DataFrame:
    """Produit un DataFrame ``(station_id, lat, lon, elevation_m, timestamp, t_celsius)``
    couvrant la nuit du ``date`` (20h → 08h+1 selon ``dataframe_to_station_obs``).
    Lit la partition de l'année de ``date``, joint au catalogue stations.

    Cache process-local par (root, year, bbox) : un appel en boucle sur plusieurs
    nuits de la même année ne refait pas le téléchargement S3 N fois.
    """
    year = pd.Timestamp(date).year
    cache_key = (str(root), year, _bbox_key(bbox))
    if cache_key in _BULK_YEAR_CACHE:
        return _BULK_YEAR_CACHE[cache_key]
    stations = load_stations_catalog(root, bbox=bbox)
    bucket_ids = stations["bucket_id"].tolist()
    ts = load_timeseries(years=[year], root=root, station_only=True, bucket_ids=bucket_ids)
    joined = ts.merge(
        stations[["bucket_id", "latitude", "longitude", "altitude_m"]],
        left_on="station_id",
        right_on="bucket_id",
        how="left",
    )
    # `dataframe_to_station_obs` parse `timestamp` lui-même (tz-naive)
    joined["timestamp"] = pd.to_datetime(joined["timestamp"], utc=True).dt.tz_convert(None)
    out = joined.rename(
        columns={
            "latitude": "lat",
            "longitude": "lon",
            "altitude_m": "elevation_m",
            "temperature": "t_celsius",
        }
    )[["station_id", "lat", "lon", "elevation_m", "timestamp", "t_celsius"]]
    _BULK_YEAR_CACHE[cache_key] = out
    return out


# ---------------------------------------------------------------------------
# Loader single-file (back-compat)
# ---------------------------------------------------------------------------
def _build_normalized_df_from_file(
    path: Path,
    columns: dict[str, str] | None,
) -> pd.DataFrame:
    """Lit un CSV/Parquet single-file et renomme via ``SENCROP_COLUMNS``."""
    mapping = {**SENCROP_COLUMNS, **(columns or {})}
    df = pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)
    return df.rename(columns={src: dst for dst, src in mapping.items()})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_sencrop(
    path: str | Path,
    date: str,
    bbox: dict[str, float] | None = None,
    columns: dict[str, str] | None = None,
) -> StationObs:
    """Charge les observations Sencrop d'une nuit → :class:`StationObs`.

    Args:
        path: soit le **root** du bulk (chemin local ou URI ``s3://…``), soit un
            **fichier** CSV/Parquet single-date (back-compat).
        date: nuit ciblée ``YYYY-MM-DD`` (fenêtre 20h → 08h+1).
        bbox: filtre spatial optionnel (``lat_min/lat_max/lon_min/lon_max``).
        columns: surcharge du mapping colonnes — *uniquement pour le mode
            single-file*, ignoré en mode bulk où le schéma est figé.

    Notes:
        - En mode bulk, ``station_id == bucket_id`` du catalogue (pas
          ``device_id``), filtre ``temperature_source == 'station'`` appliqué.
        - En mode single-file, les anciennes fixtures continuent de marcher.
    """
    if _is_bulk_root(path):
        df = _build_normalized_df_from_bulk(path, date=date, bbox=bbox)
    else:
        df = _build_normalized_df_from_file(Path(path), columns=columns)

    return dataframe_to_station_obs(df, date, bbox)
