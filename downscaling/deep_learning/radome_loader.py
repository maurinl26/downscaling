"""Loader RADOME/climato quotidien → cibles de supervision sparse pour le Lot C.

Les obs Météo-France DPClim (arborescence ``station=<id>/*.csv``, séparateur
``;``, **décimales virgule FR**) donnent une Tmin **quotidienne** : ``TN`` (min
de l'air, fallback ``TNSOL`` = min au sol). On les ajoute comme stations de
supervision supplémentaires à côté de Sencrop, en particulier les postes
d'**altitude** que Sencrop (plafond ~720 m) n'échantillonne pas — pour donner
au DL des exemples de creusement de cuvette (régime R4a).

Alignement nuit — APPROXIMATION documentée
------------------------------------------
La TN quotidienne MF datée ``D`` est utilisée comme cible pour la nuit clé
``D`` (format ``YYYY-MM-DD``) du dataset. La TN MF est le minimum journalier ;
la convention nuit Sencrop est 20h(D) → 08h(D+1). Un décalage ~±1 jour entre
le min MF daté D et le min de la nuit Sencrop D est donc possible. Acceptable
pour de la supervision d'altitude (on cherche des exemples de cuvette, pas un
appariement horaire strict). À raffiner si l'appariement s'avère limitant.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

import fsspec
import pandas as pd

log = logging.getLogger("radome_loader")

RadomeTargets = dict[str, list[tuple[float, float, float, float]]]


def _storage_options(url: str) -> dict:
    """Storage options fsspec/s3fs cohérentes avec le reste (endpoint Scaleway +
    skip_instance_cache pour éviter le deadlock s3fs multi-lectures)."""
    if "://" not in url or url.startswith(("file://", "/")):
        return {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_S3_ENDPOINT")
    opts: dict = {"skip_instance_cache": True}
    if endpoint:
        opts["client_kwargs"] = {"endpoint_url": endpoint}
    return opts


def load_radome_targets(obs_root: str, catalogue_path: str) -> RadomeTargets:
    """Lit catalogue + CSV obs → ``{date 'YYYY-MM-DD': [(lat, lon, alt_m, tmin_C), …]}``.

    Args:
        obs_root: racine ``.../quotidienne/<year>`` contenant ``station=<id>/*.csv``
            (chemin local ou URI ``s3://``).
        catalogue_path: CSV avec colonnes ``station_id, lat, lon, alt_m``.
    """
    def _norm_id(x) -> str:
        """Normalise un id station (drop zéros initiaux : path '05126001' ↔ POSTE 5126001)."""
        s = str(x).strip()
        try:
            return str(int(s))
        except ValueError:
            return s

    def _num_col(df: pd.DataFrame, col: str):
        """Colonne numérique tolérant les décimales virgule FR quotées ('−3,4' → -3.4)."""
        if col not in df.columns:
            return None
        return pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

    cat = pd.read_csv(catalogue_path, storage_options=_storage_options(catalogue_path) or None)
    meta = {
        _norm_id(r.station_id): (float(r.lat), float(r.lon), float(r.alt_m))
        for r in cat.itertuples()
    }

    fs, _ = fsspec.url_to_fs(obs_root, **_storage_options(obs_root))
    proto = (obs_root.split("://", 1)[0] + "://") if "://" in obs_root else ""
    files = fs.glob(obs_root.rstrip("/") + "/station=*/*.csv")

    targets: RadomeTargets = defaultdict(list)
    used_stations: set[str] = set()
    for f in files:
        sid_raw = f.split("station=", 1)[1].split("/", 1)[0] if "station=" in f else ""
        sid = _norm_id(sid_raw)
        if sid not in meta:
            continue
        lat, lon, alt = meta[sid]
        uri = f if "://" in f else (proto + f if proto else f)
        try:
            # CSV sauvé par le batch : séparateur virgule, décimales FR quotées ('6,4').
            df = pd.read_csv(uri, storage_options=_storage_options(uri) or None)
        except Exception as exc:  # noqa: BLE001 — station illisible : on skip, pas fatal
            log.warning("RADOME station %s illisible (%s) — skip", sid, exc)
            continue
        if "DATE" not in df.columns:
            continue
        tn = _num_col(df, "TN")
        tnsol = _num_col(df, "TNSOL")
        if tn is None:
            continue
        if tnsol is not None:
            tn = tn.fillna(tnsol)
        for date_raw, tval in zip(df["DATE"].astype(str), tn):
            if pd.isna(tval) or len(date_raw) != 8:
                continue
            key = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            targets[key].append((lat, lon, alt, float(tval)))
            used_stations.add(sid)

    n_high = sum(1 for sid in used_stations if meta[sid][2] > 800)
    log.info(
        "RADOME: %d stations mergées (%d >800 m), %d nuits couvertes",
        len(used_stations),
        n_high,
        len(targets),
    )
    return dict(targets)
