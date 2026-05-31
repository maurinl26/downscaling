"""Chargement des observations Sencrop (réseau de capteurs agricoles de parcelle).

Sencrop expose des stations météo in situ (température, humidité, vent…). Pour la
calibration locale (étage C — cf. ``docs/architecture.md``), on ne retient que
**température + position + altitude**, dans le même conteneur que Netatmo
(:data:`StationObs`), afin de partager QC, agrégation Tmin nocturne et fine-tuning
sparse. Sencrop et Netatmo jouent donc le **même rôle** (référence terrain).

Le schéma d'export Sencrop réel peut varier ; ``SENCROP_COLUMNS`` mappe les noms
de colonnes Sencrop → schéma normalisé, surchargeable via ``columns=``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Réexport pour l'agrégation Tmin (partagée avec Netatmo).
from .netatmo_qc import tmin_nocturnal  # noqa: F401
from .stations import StationObs, dataframe_to_station_obs

# Export Sencrop → schéma normalisé (station_id, lat, lon, elevation_m,
# timestamp, t_celsius). À ajuster au format réel de l'API/export Sencrop.
SENCROP_COLUMNS = {
    "station_id": "device_id",
    "lat": "latitude",
    "lon": "longitude",
    "elevation_m": "altitude",
    "timestamp": "measure_date",
    "t_celsius": "temperature",
}


def load_sencrop(
    path: str | Path,
    date: str,
    bbox: dict[str, float] | None = None,
    columns: dict[str, str] | None = None,
) -> StationObs:
    """Charge les observations Sencrop d'une nuit (CSV ou Parquet) → StationObs.

    Args:
        path: fichier d'export Sencrop (``.csv`` ou ``.parquet``).
        date: nuit ciblée ``YYYY-MM-DD`` (fenêtre 20h → 08h+1).
        bbox: filtre spatial optionnel (``lat_min/lat_max/lon_min/lon_max``).
        columns: surcharge du mapping colonnes (défaut :data:`SENCROP_COLUMNS`).
    """
    mapping = {**SENCROP_COLUMNS, **(columns or {})}
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)

    # Renomme colonnes Sencrop → schéma normalisé.
    df = df.rename(columns={src: dst for dst, src in mapping.items()})
    return dataframe_to_station_obs(df, date, bbox)
