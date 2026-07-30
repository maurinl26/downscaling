"""RADOME / SYNOP open data loader for daily Tmin (data.gouv.fr / Météo-France).

Source : ``Q_<dept>_previous-1950-2024_RR-T-Vent.csv.gz`` files from
`<https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes/>`_.
Open data depuis la réforme 2024, license Open Licence v2.0.

Schéma CSV (séparateur ``;``) :

    NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJ;RR;QRR;TN;QTN;HTN;QHTN;TX;QTX;HTX;...

On extrait juste ``NUM_POSTE, NOM_USUEL, LAT, LON, ALTI, AAAAMMJJ, TN, HTN``
pour la supervision sparse KarposSR (Tmin nocturne par station par nuit).

Densité 2022-2024 par dept (mesurée 2026-06-22) :
    26 Drôme       : 36 stations (51-1400 m)
    21 Côte-d'Or   : 26 stations (182-593 m)
    51 Marne       : 24 stations (77-225 m)
    67 Bas-Rhin    : 21 stations (120-1065 m)
    68 Haut-Rhin   : 24 stations (177-1184 m)
    84 Vaucluse    : 18 stations (34-1445 m)

À comparer avec Sencrop (densité ~5× supérieure mais altitude tronquée à
~700-800 m max → RADOME comble structurellement le haut).

URL pattern :
    https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/
    Q_<NN>_previous-1950-2024_RR-T-Vent.csv.gz

Locale awk note : si vous parsez ces CSV en bash awk, forcer ``LC_ALL=C``
sinon les décimaux français cassent la conversion numérique (44.20 → 44).
Le loader Python ci-dessous utilise pandas et ne souffre pas de ce bug.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .stations import StationObs, dataframe_to_station_obs

log = logging.getLogger("radome")

RADOME_URL_TEMPLATE = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/"
    "BASE/QUOT/Q_{dept:02d}_previous-1950-2024_RR-T-Vent.csv.gz"
)

# Colonnes utiles pour le pipeline gel
_RADOME_COLS = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJ", "TN", "HTN"]


def _dept_file_path(root: Path, dept: int) -> Path:
    """Chemin attendu du fichier dept dans le root local."""
    return Path(root) / f"Q_{dept:02d}_previous-1950-2024_RR-T-Vent.csv.gz"


@lru_cache(maxsize=16)
def _load_dept_frame(dept_file: str) -> pd.DataFrame:
    """Lit et cache (en mémoire) un fichier dept RADOME.

    Cache LRU 16 entrées (= jusqu'à 16 départements). Chaque DataFrame fait
    ~50-200 Mo en mémoire (60-80k lignes/an × 30-50 ans). Pour audit régional
    sur 5 ans on charge typiquement 2-3 départements simultanément, OK.
    """
    df = pd.read_csv(
        dept_file,
        sep=";",
        usecols=_RADOME_COLS,
        dtype={"NUM_POSTE": str, "NOM_USUEL": str, "AAAAMMJJ": str},
        low_memory=False,
    )
    # Conversion numérique (locale-safe via pandas)
    for c in ("LAT", "LON", "ALTI", "TN"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log.debug("RADOME loaded %s : %d rows", dept_file, len(df))
    return df


def load_radome(
    root: str | Path,
    date: str,
    depts: list[int] | None = None,
    bbox: dict[str, float] | None = None,
) -> StationObs:
    """Charge les observations RADOME d'une nuit → :class:`StationObs`.

    Args:
        root: dossier local contenant les fichiers ``Q_<dept>_previous-1950-2024_RR-T-Vent.csv.gz``.
            Télécharger avec ``downscaling.scripts.fetch_radome`` ou
            ``curl <RADOME_URL_TEMPLATE>``.
        date: nuit ciblée ``YYYY-MM-DD``. RADOME daily donne **un seul TN par
            nuit par station** (mesure du Tmin nocturne) ; on l'expose comme
            timestamp à l'heure du Tmin (colonne HTN, fallback 06:00 UTC).
        depts: liste des codes département à scanner (par ex. ``[26, 84, 05]``
            pour les Baronnies). Si ``None``, scanne tous les fichiers présents
            sous ``root`` (peut être lent).
        bbox: filtre spatial optionnel (``lat_min/lat_max/lon_min/lon_max``).

    Notes:
        - Les fichiers RADOME du portail data.gouv s'arrêtent à 2024 inclus
          (label ``previous-1950-2024``). Pour 2025-2026 il faut un autre
          dataset (live ou archives mises à jour).
        - Filtre stations sans TN (NaN) automatique.
    """
    root = Path(root)
    if depts is None:
        # Auto-discover from filenames
        depts = []
        for f in root.glob("Q_*_previous-*_RR-T-Vent.csv.gz"):
            try:
                dept = int(f.name.split("_")[1])
                depts.append(dept)
            except ValueError:
                continue
        if not depts:
            raise FileNotFoundError(f"No RADOME dept files in {root} (pattern Q_<NN>_...)")

    # Format date target : RADOME stores AAAAMMJJ as 'YYYYMMDD' string
    dt = pd.Timestamp(date)
    date_key = dt.strftime("%Y%m%d")

    parts = []
    for dept in depts:
        f = _dept_file_path(root, dept)
        if not f.exists():
            log.warning("RADOME dept %d file not found at %s, skipping", dept, f)
            continue
        df = _load_dept_frame(str(f))
        sub = df[df["AAAAMMJJ"] == date_key]
        if sub.empty:
            continue
        parts.append(sub)

    if not parts:
        raise ValueError(f"No RADOME station data for {date} across depts {depts}")

    df = pd.concat(parts, ignore_index=True)
    # Drop stations sans Tmin
    df = df.dropna(subset=["TN"])
    if bbox:
        df = df[
            (df["LAT"] >= bbox["lat_min"])
            & (df["LAT"] <= bbox["lat_max"])
            & (df["LON"] >= bbox["lon_min"])
            & (df["LON"] <= bbox["lon_max"])
        ]
    if df.empty:
        raise ValueError(f"No RADOME station in bbox for {date}")

    # Construire un DataFrame normalisé compatible dataframe_to_station_obs.
    # Reconstruire timestamp via HTN (heure HHMM string). Fallback 06:00 UTC.
    def _build_ts(row):
        htn = row.get("HTN")
        try:
            if pd.notna(htn) and str(htn).strip() not in ("", "nan"):
                # HTN peut être "0530" (4 digits) ou "530" (3 digits) ou float
                s = str(int(float(htn))).zfill(4)
                h, m = int(s[:2]), int(s[2:])
                return pd.Timestamp(date) + pd.Timedelta(hours=h, minutes=m)
        except (ValueError, TypeError):
            pass
        # Fallback : 06h UTC, milieu typique du Tmin nocturne en hiver
        return pd.Timestamp(date) + pd.Timedelta("6h")

    df_norm = pd.DataFrame(
        {
            "station_id": "RADOME-" + df["NUM_POSTE"].astype(str),
            "lat": df["LAT"].values,
            "lon": df["LON"].values,
            "elevation_m": df["ALTI"].values,
            "timestamp": df.apply(_build_ts, axis=1),
            "t_celsius": df["TN"].values.astype(np.float32),
        }
    )

    # Réutilise la même conversion que Sencrop (pivote stations × timestamps).
    # Comme on a 1 timestamp par station, le pivot retourne 1 colonne.
    # La fenêtre nuit du dataframe_to_station_obs est 20h(date) → 08h(date+1),
    # on s'assure que nos timestamps tombent dedans (HTN typiquement 03-08h ⇒
    # date+1 matin ⇒ on shifte les timestamps à date+0,5j si HTN < 12h).
    df_norm["timestamp"] = df_norm["timestamp"].apply(
        lambda t: t + pd.Timedelta("1D") if t.hour < 12 else t
    )
    # Si la rotation a poussé l'obs après date+1 08h, on rebascule à 06h date+1
    cutoff = pd.Timestamp(date) + pd.Timedelta("1D") + pd.Timedelta("8h")
    df_norm.loc[df_norm["timestamp"] >= cutoff, "timestamp"] = (
        pd.Timestamp(date) + pd.Timedelta("1D") + pd.Timedelta("6h")
    )

    return dataframe_to_station_obs(df_norm, date, bbox=None)


def combine_station_obs(*obs_list: StationObs) -> StationObs:
    """Concatène plusieurs StationObs (Sencrop + RADOME, par exemple).

    Préserve l'ordre des stations (Sencrop d'abord par convention puisqu'il
    sert de référence catalogue). Dédoublonne par ``station_id``.
    """
    obs_list = [o for o in obs_list if o is not None]
    if not obs_list:
        raise ValueError("No StationObs to combine")
    if len(obs_list) == 1:
        return obs_list[0]

    all_ids = []
    all_lat = []
    all_lon = []
    all_elev = []
    rows = []
    times = obs_list[0].times  # utilise la time grid du premier (Sencrop est canonique)
    for obs in obs_list:
        for i, sid in enumerate(obs.station_id):
            if sid in all_ids:
                continue
            all_ids.append(sid)
            all_lat.append(obs.lat[i])
            all_lon.append(obs.lon[i])
            all_elev.append(obs.elevation_m[i])
            # Réaligne la time grid (interp / NaN-fill si différentes)
            if list(obs.times) == list(times):
                row = obs.t_raw[i]
            else:
                # Build a NaN row with `times` length, fill where obs.times matches
                row = np.full(len(times), np.nan, dtype=np.float32)
                src_times = pd.DatetimeIndex(obs.times)
                tgt_times = pd.DatetimeIndex(times)
                # Approximate match within ±30 min
                for j, t in enumerate(tgt_times):
                    deltas = np.abs((src_times - t).total_seconds())
                    if deltas.size and deltas.min() <= 1800:
                        row[j] = obs.t_raw[i, deltas.argmin()]
            rows.append(row)

    return StationObs(
        station_id=np.array(all_ids),
        lat=np.array(all_lat),
        lon=np.array(all_lon),
        elevation_m=np.array(all_elev),
        t_raw=np.array(rows, dtype=np.float32),
        times=times,
    )
