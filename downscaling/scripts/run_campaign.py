"""
Orchestrateur campagne backtesting gel arboricole — Drôme-Ardèche 2000–2024.

Pipeline par saison (oct N → mai N+1) :
  1. Descente d'échelle statistique (ERA5Land ou CERRALand → 1 km)
  2. Détection automatique des nuits de gel post-débourrement
  3. Pour chaque nuit critique :
       a. Génération des LBC horaires (era5_to_pmap_lbc.py)
       b. Génération FORCING.nc + run SURFEX offline
       c. Génération config PMAP depuis template
       d. Lancement run PMAP-LES (pmap-les config.yml)
  4. Calcul indices gel sur outputs PMAP + agrégation saisonnière

Usage
-----
    # Depuis la racine du repo downscaling/ :

    # Campagne complète ERA5Land
    run-campaign --config runs/campaign/config.yml --source era5land --step all

    # Campagne complète les deux sources
    run-campaign --config runs/campaign/config.yml --source both --step all

    # Seulement détecter les nuits (sans lancer PMAP)
    run-campaign --config runs/campaign/config.yml --source era5land --step detect

    # Relancer PMAP pour une saison spécifique
    run-campaign --config runs/campaign/config.yml --source era5land --season 2021 --step run-pmap

    # Via uv (sans installation préalable)
    uv run --with downscaling downscaling/scripts/run_campaign.py --source era5land --step all

Installation
------------
    uv sync --extra pmap     # installe downscaling + pmap depuis git
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# downscaling/scripts/run_campaign.py → parents[1]=downscaling/, parents[2]=atmospheric_models/
REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNSCALING_ROOT = REPO_ROOT / "downscaling"
PMAP_REPO = REPO_ROOT / "PMAP-LES-shared"
PMAP_SCRIPTS = PMAP_REPO / "scripts"


# ---------------------------------------------------------------------------
# Itération sur les saisons
# ---------------------------------------------------------------------------

def iter_seasons(start_year: int, end_year: int, start_month: int, end_month: int):
    """
    Génère (year_start, month_start, year_end, month_end) pour chaque saison.

    Pour start_month=10, end_month=5 :
      (2000, 10, 2001, 5), (2001, 10, 2002, 5), …
    """
    for year in range(start_year, end_year):
        if start_month > end_month:
            yield (year, start_month, year + 1, end_month)
        else:
            yield (year, start_month, year, end_month)


def season_months(year_start: int, month_start: int, year_end: int, month_end: int):
    """Génère toutes les paires (année, mois) d'une saison."""
    y, m = year_start, month_start
    while (y, m) <= (year_end, month_end):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


# ---------------------------------------------------------------------------
# Données source
# ---------------------------------------------------------------------------

def source_file(cfg: dict, source: str, year: int, month: int) -> Path | None:
    """Retourne le chemin du fichier mensuel pour une source donnée."""
    src_cfg = cfg["campaign"]["sources"][source]
    data_root = DOWNSCALING_ROOT / src_cfg["data_root"]
    filename = src_cfg["file_pattern"].format(year=year, month=month)
    path = data_root / filename
    return path if path.exists() else None


def collect_season_files(cfg: dict, source: str, year_start: int, month_start: int,
                         year_end: int, month_end: int) -> list[Path]:
    """Retourne la liste ordonnée des fichiers mensuels disponibles pour la saison."""
    files = []
    for y, m in season_months(year_start, month_start, year_end, month_end):
        p = source_file(cfg, source, y, m)
        if p is not None:
            files.append(p)
        else:
            log.warning(f"Fichier manquant : {source} {y}-{m:02d}")
    return files


# ---------------------------------------------------------------------------
# Étape 1 : Descente d'échelle statistique
# ---------------------------------------------------------------------------

def step_stat_downscaling(cfg: dict, source: str, year_start: int, month_start: int,
                           year_end: int, month_end: int) -> Path | None:
    """Lance la descente d'échelle statistique pour une saison."""
    season_label = f"{year_start}{month_start:02d}_{year_end}{month_end:02d}"
    out_dir = DOWNSCALING_ROOT / cfg["data"]["stat_output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"stat_{source}_{season_label}.nc"

    if out_file.exists():
        log.info(f"Sortie stat déjà disponible : {out_file}")
        return out_file

    files = collect_season_files(cfg, source, year_start, month_start, year_end, month_end)
    if not files:
        log.warning(f"Aucun fichier {source} pour la saison {season_label} → ignorée")
        return None

    script = "run_era5land_downscaling.py" if "land" in source else "run_statistical_downscaling.py"
    cmd = [
        sys.executable,
        f"downscaling/scripts/{script}",
        "--config",         "config/drome_ardeche.yml",
        "--input-files",    *[str(f) for f in files],
        "--dem",            cfg["data"]["dem"]["raw"],
        "--out",            str(out_file),
        "--compute-indices",
    ]
    _run(cmd, cwd=DOWNSCALING_ROOT, label=f"stat-downscaling-{season_label}")
    return out_file if out_file.exists() else None


# ---------------------------------------------------------------------------
# Étape 2 : Détection des nuits de gel
# ---------------------------------------------------------------------------

def step_detect(cfg: dict, stat_output: Path, season_label: str) -> list[dict]:
    """Détecte les nuits critiques depuis la sortie stat."""
    out_dir = DOWNSCALING_ROOT / "runs" / "campaign" / "cold_nights"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"cold_nights_{season_label}.json"

    cmd = [
        sys.executable,
        "runs/scripts/detect_cold_nights.py",
        "--stat-out", str(stat_output),
        "--config",   "runs/campaign/config.yml",
        "--out",      str(out_json),
    ]
    _run(cmd, cwd=DOWNSCALING_ROOT, label=f"detect-{season_label}")

    if not out_json.exists():
        return []
    with open(out_json) as f:
        data = json.load(f)
    nights = data.get("cold_nights", data) if isinstance(data, dict) else data
    log.info(f"  {len(nights)} nuit(s) critique(s) pour {season_label}")
    return nights


# ---------------------------------------------------------------------------
# Étape 3a : Préparation LBC
# ---------------------------------------------------------------------------

def step_prepare_lbc(cfg: dict, night: dict, source: str) -> Path | None:
    """Génère les LBC horaires (13 snapshots 18h–06h) pour une nuit critique."""
    d = night["date"]
    lbc_dir = DOWNSCALING_ROOT / cfg["pmap"]["lbc_root"] / source / d
    if lbc_dir.exists() and any(lbc_dir.iterdir()):
        log.info(f"LBC déjà disponibles : {lbc_dir}")
        return lbc_dir

    lbc_dir.mkdir(parents=True, exist_ok=True)
    domain = cfg["domain"]
    pmap_cfg = cfg["pmap"]

    night_date = date.fromisoformat(d)
    pl_file = source_file(cfg, source, night_date.year, night_date.month)
    sl_file = pl_file  # ERA5Land/CERRALand : un seul fichier mensuel

    if pl_file is None:
        log.warning(f"Fichier {source} manquant pour {d} → LBC ignorées")
        return None

    cmd = [
        sys.executable,
        str(PMAP_SCRIPTS / "era5_to_pmap_lbc.py"),
        "--era5-pl",         str(pl_file),
        "--era5-sl",         str(sl_file),
        "--outdir",          str(lbc_dir),
        "--lat-min",         str(domain["lat_min"]),
        "--lat-max",         str(domain["lat_max"]),
        "--lon-min",         str(domain["lon_min"]),
        "--lon-max",         str(domain["lon_max"]),
        "--nx",              str(pmap_cfg["nx"]),
        "--ny",              str(pmap_cfg["ny"]),
        "--nz",              str(pmap_cfg["nz"]),
        "--xmax",            str(pmap_cfg["xmax"]),
        "--ymax",            str(pmap_cfg["ymax"]),
        "--zmax",            str(pmap_cfg["zmax"]),
        "--dz-near-surface", str(pmap_cfg["dz_near_surface"]),
        "--lat-center",      str(domain["lat_center"]),
        "--start-timestep",  str(cfg["detection"]["nocturnal_window"]["start_h"]),
        "--nstep",           str(cfg["lbc_prep"]["nstep"]),
    ]
    _run(cmd, cwd=REPO_ROOT, label=f"lbc-{source}-{d}")
    return lbc_dir


# ---------------------------------------------------------------------------
# Étape 3b : Préparation SURFEX
# ---------------------------------------------------------------------------

def step_prepare_surfex(cfg: dict, night: dict, source: str) -> Path | None:
    """Génère FORCING.nc et lance SURFEX offline pour une nuit critique."""
    d = night["date"]
    night_date = date.fromisoformat(d)
    domain = cfg["domain"]
    pmap_cfg = cfg["pmap"]

    forcing_dir = DOWNSCALING_ROOT / cfg["data"]["surfex"]["forcing_dir"] / source
    forcing_dir.mkdir(parents=True, exist_ok=True)
    forcing_out = forcing_dir / f"FORCING_{d.replace('-', '')}.nc"

    sl_file = source_file(cfg, source, night_date.year, night_date.month)
    if sl_file is None:
        log.warning(f"Fichier {source} SL manquant pour {d} → SURFEX ignoré")
        return None

    if not forcing_out.exists():
        cmd = [
            sys.executable,
            str(PMAP_SCRIPTS / "era5_to_surfex_forcing.py"),
            "--era5-sl",    str(sl_file),
            "--outfile",    str(forcing_out),
            "--lat-min",    str(domain["lat_min"]),
            "--lat-max",    str(domain["lat_max"]),
            "--lon-min",    str(domain["lon_min"]),
            "--lon-max",    str(domain["lon_max"]),
            "--nx",         str(pmap_cfg["nx"]),
            "--ny",         str(pmap_cfg["ny"]),
            "--lat-center", str(domain["lat_center"]),
            "--lon-center", str(domain["lon_center"]),
        ]
        _run(cmd, cwd=REPO_ROOT, label=f"surfex-forcing-{d}")

    surfex_exe = shutil.which("offline")
    surfex_out_dir = DOWNSCALING_ROOT / cfg["data"]["surfex"]["output_dir"] / source
    surfex_out_dir.mkdir(parents=True, exist_ok=True)
    surfex_output = surfex_out_dir / f"OUTPUT_DIAG_ISBA_{d.replace('-', '')}.nc"

    if surfex_output.exists():
        log.info(f"Output SURFEX déjà disponible : {surfex_output}")
        return surfex_output

    if surfex_exe is None:
        log.warning(f"SURFEX 'offline' introuvable → run SURFEX ignoré pour {d}")
        log.warning("  Compiler open-SURFEX et ajouter l'exe au PATH.")
        return None

    nam_file = (REPO_ROOT / "open-SURFEX-V9-1-0" / "MY_RUN" / "NAMELIST" /
                "drome_ardeche" / "offline_drome_ardeche.nam")
    if not nam_file.exists():
        log.warning(f"Namelist SURFEX introuvable : {nam_file}")
        return None

    _run([surfex_exe, str(nam_file)], cwd=REPO_ROOT, label=f"surfex-offline-{d}")
    return surfex_output if surfex_output.exists() else None


# ---------------------------------------------------------------------------
# Étape 3c : Génération config PMAP depuis template
# ---------------------------------------------------------------------------

def generate_pmap_config(cfg: dict, night: dict, source: str,
                          lbc_dir: Path, surfex_output: Path) -> Path:
    """Instancie le template YAML PMAP pour une nuit donnée."""
    d = night["date"]
    night_date = date.fromisoformat(d)
    next_day = night_date + timedelta(days=1)

    init_file = source_file(cfg, source, night_date.year, night_date.month)
    output_dir = DOWNSCALING_ROOT / cfg["pmap"]["output_root"] / source / d
    output_dir.mkdir(parents=True, exist_ok=True)

    template_name = f"pmap_{source}_template.yml"
    template_path = DOWNSCALING_ROOT / cfg["pmap"]["template_dir"] / template_name
    template_text = template_path.read_text()

    substitutions = {
        "night_date":       d,
        "start_datetime":   f"{d} 18:00:00",
        "end_datetime":     f"{next_day.isoformat()} 06:00:00",
        "era5_file":        str(init_file) if init_file else "MISSING",
        "lbc_directory":    str(lbc_dir),
        "surfex_output":    str(surfex_output) if surfex_output else "MISSING",
        "output_directory": str(output_dir) + "/",
    }
    for key, val in substitutions.items():
        template_text = template_text.replace("{{ " + key + " }}", val)
        template_text = template_text.replace("{{" + key + "}}", val)

    config_dir = DOWNSCALING_ROOT / "runs" / "campaign" / "pmap_configs" / source
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"pmap_{source}_{d.replace('-', '')}.yml"
    config_path.write_text(template_text)
    log.info(f"Config PMAP générée : {config_path}")
    return config_path


# ---------------------------------------------------------------------------
# Étape 3d : Lancement PMAP-LES
# ---------------------------------------------------------------------------

def step_run_pmap(cfg: dict, night: dict, source: str, pmap_config: Path):
    """Lance pmap-les pour une nuit critique."""
    pmap_exe = shutil.which(cfg["pmap"]["executable"])
    if pmap_exe is None:
        log.error(
            f"'{cfg['pmap']['executable']}' introuvable.\n"
            "  → uv sync --extra pmap"
        )
        return
    d = night["date"]
    log.info(f"Lancement PMAP [{source.upper()}] nuit {d}…")
    _run([pmap_exe, str(pmap_config)], cwd=DOWNSCALING_ROOT, label=f"pmap-{source}-{d}")


# ---------------------------------------------------------------------------
# Pipeline par nuit critique
# ---------------------------------------------------------------------------

def run_night(cfg: dict, night: dict, source: str, steps: list[str]):
    """Enchaîne LBC → SURFEX → PMAP pour une nuit détectée."""
    d = night["date"]
    log.info(f"━━━ Nuit {d} [{source}] sévérité={night.get('severity', '?')} ━━━")

    lbc_dir = None
    surfex_output = None

    if "prepare-lbc" in steps or "all" in steps:
        lbc_dir = step_prepare_lbc(cfg, night, source)

    if "prepare-surfex" in steps or "all" in steps:
        surfex_output = step_prepare_surfex(cfg, night, source)

    if "run-pmap" in steps or "all" in steps:
        if lbc_dir is None or surfex_output is None:
            log.warning(f"LBC ou SURFEX manquant pour {d} → PMAP ignoré")
            return
        pmap_config = generate_pmap_config(cfg, night, source, lbc_dir, surfex_output)
        step_run_pmap(cfg, night, source, pmap_config)


# ---------------------------------------------------------------------------
# Pipeline par saison
# ---------------------------------------------------------------------------

def run_season(cfg: dict, source: str, year_start: int, month_start: int,
               year_end: int, month_end: int, steps: list[str]) -> list[dict]:
    """Exécute le pipeline complet pour une saison (oct N → mai N+1)."""
    season_label = f"{year_start}{month_start:02d}_{year_end}{month_end:02d}"
    log.info(f"══════ Saison {season_label} [{source}] ══════")

    stat_output = None
    cold_nights: list[dict] = []

    if "stat-downscaling" in steps or "all" in steps:
        stat_output = step_stat_downscaling(
            cfg, source, year_start, month_start, year_end, month_end
        )

    if "detect" in steps or "all" in steps:
        if stat_output is None or not stat_output.exists():
            log.warning(f"Sortie stat absente pour {season_label} → détection ignorée")
            return []
        cold_nights = step_detect(cfg, stat_output, season_label)
    else:
        out_json = (DOWNSCALING_ROOT / "runs" / "campaign" / "cold_nights" /
                    f"cold_nights_{season_label}.json")
        if out_json.exists():
            with open(out_json) as f:
                data = json.load(f)
            cold_nights = data.get("cold_nights", data) if isinstance(data, dict) else data

    for night in cold_nights:
        run_night(cfg, night, source, steps)

    return cold_nights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path, label: str):
    log.info(f"[{label}] $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"[{label}] Échec (code {result.returncode})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Campagne backtesting gel arboricole — ERA5Land/CERRALand 2000–2024"
    )
    p.add_argument("--config",  default="runs/campaign/config.yml")
    p.add_argument("--source",  choices=["era5land", "cerraland", "both"], default="era5land")
    p.add_argument("--season",  type=int, default=None,
                   help="Limiter à une saison (ex. 2021 pour oct 2021–mai 2022)")
    p.add_argument("--step",
                   choices=["stat-downscaling", "detect", "prepare-lbc",
                             "prepare-surfex", "run-pmap", "all"],
                   default="all")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config_path = DOWNSCALING_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    campaign = cfg["campaign"]
    start_year  = args.season or campaign["start_year"]
    end_year    = (args.season + 1) if args.season else campaign["end_year"]
    start_month = campaign["start_month"]
    end_month   = campaign["end_month"]

    sources = ["era5land", "cerraland"] if args.source == "both" else [args.source]
    steps = [args.step]

    for source in sources:
        if not campaign["sources"].get(source, {}).get("enabled", False):
            log.info(f"Source {source} désactivée dans la config → ignorée")
            continue

        total_nights = 0
        for season in iter_seasons(start_year, end_year, start_month, end_month):
            yr_s, mo_s, yr_e, mo_e = season
            nights = run_season(cfg, source, yr_s, mo_s, yr_e, mo_e, steps)
            total_nights += len(nights)

        log.info(f"[{source}] Total : {total_nights} nuit(s) critique(s) traitée(s)")

    log.info("Campagne terminée.")


if __name__ == "__main__":
    main()
