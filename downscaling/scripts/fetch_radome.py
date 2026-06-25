#!/usr/bin/env python3
"""Download RADOME / SYNOP daily Tmin files from data.gouv.fr (Météo-France).

Pulls the per-department ``Q_<NN>_previous-1950-2024_RR-T-Vent.csv.gz`` files
needed by :mod:`downscaling.prtihvi_wxc.radome` (loader créé dans le commit
parent — cf. PR #61). Optionnellement, sync vers un bucket S3 pour cohérence
avec la stratégie [[project-data-strategy-runpod-s3]].

Source : https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes
Licence : Open Licence v2.0.

Données disponibles : depuis l'ouverture des stations (1900-1950 pour
principales, 1990+ pour secondaires) jusqu'à 2024 inclus. Pour 2025-2026
il faut un dataset live complémentaire (TODO).

Usage
-----

Download Drôme + Vaucluse + Hautes-Alpes pour la zone Baronnies::

    uv run python -m downscaling.scripts.fetch_radome \\
        --depts 26 84 5 \\
        --out /Users/loicmaurin/kDrive/karpos_datasets/data/raw/radome

Download cibles MoU Sencrop (Bourgogne + Champagne + Alsace)::

    uv run python -m downscaling.scripts.fetch_radome \\
        --depts 21 51 67 68 \\
        --out radome_data \\
        --s3-prefix s3://karpos-backtest-data/radome

Avec le flag ``--s3-prefix``, les fichiers téléchargés sont aussi pushés
vers S3 via fsspec. Credentials Scaleway via ``AWS_ACCESS_KEY_ID`` /
``AWS_SECRET_ACCESS_KEY`` env (cf. ``downscaling/utils/io.py``).

Sizes typiques : 5-25 Mo par département × période (Drôme = 16 Mo,
Marne = 10 Mo, Côte d'Or = 13 Mo, Bas-Rhin = 17 Mo).

Skips existing files unless ``--force``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

log = logging.getLogger("fetch_radome")

# URL pattern data.gouv.fr (object.files.data.gouv.fr → Scaleway S3 hostname)
URL_TEMPLATE_RR_T_VENT = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/"
    "BASE/QUOT/Q_{dept:02d}_previous-1950-2024_RR-T-Vent.csv.gz"
)
URL_TEMPLATE_OTHER = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/"
    "BASE/QUOT/Q_{dept:02d}_previous-1950-2024_autres-parametres.csv.gz"
)


def _file_name(dept: int, *, kind: str = "RR-T-Vent") -> str:
    return f"Q_{dept:02d}_previous-1950-2024_{kind}.csv.gz"


def _download(url: str, dest: Path, force: bool = False) -> tuple[bool, int]:
    """Download url to dest. Returns (downloaded, size_bytes)."""
    if dest.exists() and not force:
        size = dest.stat().st_size
        log.info("  %s : already present (%.1f MB), skip", dest.name, size / 1024 / 1024)
        return False, size
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log.info("  %s : downloading from %s", dest.name, url)
    try:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        urlretrieve(url, tmp)
        tmp.rename(dest)
    except Exception as exc:
        log.error("  %s : download failed: %s", dest.name, exc)
        if dest.with_suffix(dest.suffix + ".tmp").exists():
            dest.with_suffix(dest.suffix + ".tmp").unlink()
        raise
    size = dest.stat().st_size
    elapsed = time.time() - start
    log.info("  %s : downloaded %.1f MB in %.1f s", dest.name, size / 1024 / 1024, elapsed)
    return True, size


def _push_s3(local_path: Path, s3_url: str) -> None:
    """Push a local file to S3 via fsspec."""
    try:
        import fsspec
    except ImportError as exc:
        raise RuntimeError(
            "fsspec required for --s3-prefix, install with `uv pip install fsspec s3fs`"
        ) from exc
    from downscaling.utils.io import _s3_endpoint, _normalize_s3_url

    url = _normalize_s3_url(s3_url)
    fs = fsspec.filesystem("s3", client_kwargs={"endpoint_url": _s3_endpoint()})
    target = f"{url.rstrip('/')}/{local_path.name}"
    log.info("  → S3 %s", target)
    with (
        open(local_path, "rb") as f_in,
        fsspec.open(target, mode="wb", client_kwargs={"endpoint_url": _s3_endpoint()}) as f_out,
    ):
        # Stream copy 8 MB chunks
        while True:
            chunk = f_in.read(8 * 1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch RADOME daily Tmin per dept from data.gouv.fr")
    p.add_argument(
        "--depts",
        type=int,
        nargs="+",
        required=True,
        help="Department codes (e.g. 26 84 5 for Baronnies). "
        "Use the numeric code, not zero-padded.",
    )
    p.add_argument("--out", type=Path, required=True, help="Local output directory")
    p.add_argument(
        "--include-other-params",
        action="store_true",
        help="Also download Q_<NN>_..._autres-parametres.csv.gz (humidity, pressure, snow...)",
    )
    p.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help="Optional S3 prefix to mirror downloaded files "
        "(e.g. s3://karpos-backtest-data/radome). "
        "Requires AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY env.",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if local file exists")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    total_size = 0

    for dept in args.depts:
        log.info("=== Dept %02d ===", dept)
        kinds = ["RR-T-Vent"]
        if args.include_other_params:
            kinds.append("autres-parametres")
        for kind in kinds:
            url_template = URL_TEMPLATE_RR_T_VENT if kind == "RR-T-Vent" else URL_TEMPLATE_OTHER
            url = url_template.format(dept=dept)
            dest = args.out / _file_name(dept, kind=kind)
            try:
                downloaded, size = _download(url, dest, force=args.force)
            except Exception:
                log.warning("  Dept %02d %s : failed, continuing with next", dept, kind)
                continue
            if downloaded:
                total_downloaded += 1
            total_size += size
            if args.s3_prefix:
                try:
                    _push_s3(dest, args.s3_prefix)
                except Exception as exc:
                    log.warning("  S3 push failed: %s", exc)

    log.info(
        "=== DONE : %d files downloaded, total %.1f MB ===",
        total_downloaded,
        total_size / 1024 / 1024,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
