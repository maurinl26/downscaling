#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "xarray>=2023.1",
#   "netCDF4>=1.6",
#   "pyyaml>=6.0",
#   "downscaling",
# ]
# ///
"""
Orchestrateur du workflow de vérification — Gel Drôme-Ardèche Avril 2021.

Framework
---------
  [stat/DL downscaling] → [détection nuits froides] → [prép. LBC] → [runs PMAP]

Étapes disponibles
------------------
  stat-downscaling  : descente d'échelle statistique sur tout avril 2021
  detect            : détection des nuits froides depuis la sortie stat
  prepare-lbc       : génération des fichiers LBC ERA5/CERRA pour chaque nuit
  prepare-surfex    : génération FORCING.nc + run SURFEX offline pour chaque nuit
  run-pmap          : lancement des runs PMAP pour les nuits critiques
  all               : enchaîne toutes les étapes dans l'ordre

Usage
-----
    # Production complète
    python runs/scripts/orchestrate.py \\
        --config runs/april2021/config.yml \\
        --forcing era5 \\
        --step all

    # Uniquement préparer les LBC ERA5 pour les nuits déjà détectées
    python runs/scripts/orchestrate.py \\
        --config      runs/april2021/config.yml \\
        --cold-nights runs/april2021/cold_nights.json \\
        --forcing     era5 \\
        --step        prepare-lbc

    # Lancer les runs PMAP CERRA pour les 3 nuits
    python runs/scripts/orchestrate.py \\
        --config      runs/april2021/config.yml \\
        --cold-nights runs/april2021/cold_nights.json \\
        --forcing     cerra \\
        --step        run-pmap
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# Racine du dépôt atmospheric_models
REPO_ROOT = Path(__file__).resolve().parents[3]
PMAP_SCRIPTS = REPO_ROOT / "PMAP-LES-shared" / "scripts"


# ---------------------------------------------------------------------------
# Étapes individuelles
# ---------------------------------------------------------------------------

def step_stat_downscaling(cfg: dict, args: argparse.Namespace):
    """Lance la descente d'échelle statistique sur tout avril 2021."""
    data_cfg = cfg["data"]
    domain = cfg["domain"]

    cmd = [
        sys.executable,
        "scripts/run_statistical_downscaling.py",
        "--config",   "config/drome_ardeche.yml",
        "--era5-sl",  data_cfg["era5"]["single_level"]["april2021"],
        "--dem",      data_cfg["dem"]["raw"],
        "--date",     "april2021",
        "--out",      data_cfg["downscaling"]["stat_output"],
        "--compute-indices",
    ]
    if args.mod_ref:
        cmd += ["--mod-ref", args.mod_ref]
    if args.obs_ref:
        cmd += ["--obs-ref", args.obs_ref]

    _run(cmd, cwd=REPO_ROOT / "downscaling", label="stat-downscaling")


def step_detect(cfg: dict, args: argparse.Namespace) -> list[dict]:
    """Détecte les nuits froides depuis la sortie stat."""
    stat_out = cfg["data"]["downscaling"]["stat_output"]
    cold_nights_path = Path(args.cold_nights)

    cmd = [
        sys.executable,
        "runs/scripts/detect_cold_nights.py",
        "--stat-out", stat_out,
        "--config",   args.config,
        "--out",      str(cold_nights_path),
    ]
    if args.verbose:
        cmd.append("-v")
    _run(cmd, cwd=REPO_ROOT / "downscaling", label="detect-cold-nights")

    with open(cold_nights_path) as f:
        return json.load(f)["cold_nights"]


def step_prepare_lbc(cfg: dict, cold_nights: list[dict], forcing: str):
    """
    Génère les fichiers LBC pour chaque nuit critique.

    Pour chaque nuit :
      - Prépare 13 fichiers input_N.nc (18h → 06h+1) depuis ERA5 ou CERRA
      - Appelle era5_to_pmap_lbc.py (accepte aussi CERRA après standardisation CDS)
    """
    data_cfg = cfg["data"]
    lbc_cfg  = cfg["lbc_prep"]
    domain   = cfg["domain"]
    pmap_cfg = cfg["pmap"]

    for night in cold_nights:
        if not night.get("run_pmap", True):
            continue

        date = night["date"]
        label = night["label"]
        log.info(f"Préparation LBC {forcing.upper()} pour {label}…")

        if forcing == "era5":
            pl_file = data_cfg["era5"]["pressure_level"].get(date)
            sl_file = data_cfg["era5"]["single_level"].get(date)
            lbc_dir = data_cfg["lbc"]["era5"].get(date)
        else:
            pl_file = data_cfg["cerra"]["pressure_level"].get(date)
            sl_file = data_cfg["cerra"]["single_level"].get(date)
            lbc_dir = data_cfg["lbc"]["cerra"].get(date)

        if not pl_file or not sl_file:
            log.warning(f"  Fichiers {forcing.upper()} manquants pour {date} → ignoré")
            continue

        lbc_dir = REPO_ROOT / lbc_dir
        lbc_dir.mkdir(parents=True, exist_ok=True)

        # Heure de début nocturne (18h dans le fichier journalier)
        start_h = cfg["detection"]["nocturnal_window"]["start_h"]

        cmd = [
            sys.executable,
            str(PMAP_SCRIPTS / "era5_to_pmap_lbc.py"),
            "--era5-pl",          str(REPO_ROOT / pl_file),
            "--era5-sl",          str(REPO_ROOT / sl_file),
            "--outdir",           str(lbc_dir),
            "--lat-min",          str(domain["lat_min"]),
            "--lat-max",          str(domain["lat_max"]),
            "--lon-min",          str(domain["lon_min"]),
            "--lon-max",          str(domain["lon_max"]),
            "--nx",               str(pmap_cfg["nx"]),
            "--ny",               str(pmap_cfg["ny"]),
            "--nz",               str(pmap_cfg["nz"]),
            "--xmax",             str(pmap_cfg["xmax"]),
            "--ymax",             str(pmap_cfg["ymax"]),
            "--zmax",             str(pmap_cfg["zmax"]),
            "--dz-near-surface",  str(pmap_cfg["dz_near_surface"]),
            "--lat-center",       str(domain["lat_center"]),
            "--start-timestep",   str(start_h),   # index 18h dans le fichier
            "--nstep",            str(lbc_cfg["nstep"]),
        ]
        _run(cmd, cwd=REPO_ROOT, label=f"lbc-{forcing}-{date}")


def step_prepare_surfex(cfg: dict, cold_nights: list[dict]):
    """
    Génère FORCING.nc + lance SURFEX offline pour chaque nuit critique.

    Le FORCING.nc couvre 18h–06h UTC (12h nocturne).
    SURFEX doit avoir été compilé et PGD/PREP doivent exister.
    """
    data_cfg = cfg["data"]
    domain   = cfg["domain"]
    pmap_cfg = cfg["pmap"]

    surfex_exe = shutil.which("offline")
    if surfex_exe is None:
        log.warning("Exécutable SURFEX 'offline' introuvable dans le PATH.")
        log.warning("  → Compiler SURFEX : cd open-SURFEX-V9-1-0/src && make -j8 ARCH=MCgfortran OPTLEVEL=O2 VER_MPI=NOMPI OFFLINE")
        log.warning("  → Ou ajouter open-SURFEX-V9-1-0/exe/ au PATH")

    for night in cold_nights:
        if not night.get("run_pmap", True):
            continue
        date = night["date"]

        # 1. Générer FORCING.nc
        era5_sl = data_cfg["era5"]["single_level"].get(date)
        forcing_out = data_cfg["surfex"]["forcing"].get(date)

        if era5_sl and forcing_out:
            log.info(f"Génération FORCING.nc pour {date}…")
            Path(REPO_ROOT / forcing_out).parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(PMAP_SCRIPTS / "era5_to_surfex_forcing.py"),
                "--era5-sl",     str(REPO_ROOT / era5_sl),
                "--outfile",     str(REPO_ROOT / forcing_out),
                "--lat-min",     str(domain["lat_min"]),
                "--lat-max",     str(domain["lat_max"]),
                "--lon-min",     str(domain["lon_min"]),
                "--lon-max",     str(domain["lon_max"]),
                "--nx",          str(pmap_cfg["nx"]),
                "--ny",          str(pmap_cfg["ny"]),
                "--lat-center",  str(domain["lat_center"]),
                "--lon-center",  str(domain["lon_center"]),
            ]
            _run(cmd, cwd=REPO_ROOT, label=f"surfex-forcing-{date}")
        else:
            log.warning(f"ERA5 SL ou chemin FORCING manquant pour {date}")

        # 2. Run SURFEX offline
        surfex_out = data_cfg["surfex"]["output"].get(date)
        nam_file = REPO_ROOT / "open-SURFEX-V9-1-0" / "MY_RUN" / "NAMELIST" / \
                   "drome_ardeche" / "offline_drome_ardeche.nam"

        if surfex_exe and nam_file.exists() and surfex_out:
            log.info(f"Run SURFEX offline pour {date}…")
            Path(REPO_ROOT / surfex_out).parent.mkdir(parents=True, exist_ok=True)
            _run([surfex_exe, str(nam_file)], cwd=REPO_ROOT, label=f"surfex-offline-{date}")
        elif not surfex_exe:
            log.warning(f"  SURFEX non disponible → run SURFEX {date} ignoré")
        else:
            log.warning(f"  Namelist SURFEX ou chemin output manquant pour {date}")


def step_run_pmap(cfg: dict, cold_nights: list[dict], forcing: str):
    """Lance les runs PMAP-LES pour les nuits critiques."""
    pmap_exe = shutil.which(cfg["pmap"]["executable"])
    if pmap_exe is None:
        log.error(
            f"Exécutable '{cfg['pmap']['executable']}' introuvable.\n"
            "  → pip install pmap[dev] ou ajouter le venv au PATH"
        )
        return

    config_dir = REPO_ROOT / "downscaling" / cfg["pmap"]["config_dir"]

    for night in cold_nights:
        if not night.get("run_pmap", True):
            continue

        date = night["date"]
        date_compact = date.replace("-", "")[2:]   # "210427"
        config_file = config_dir / f"{forcing}_night_{date_compact}.yml"

        if not config_file.exists():
            log.warning(f"Config PMAP manquante : {config_file}")
            continue

        log.info(f"Lancement PMAP [{forcing.upper()}] nuit {date}…")
        _run([pmap_exe, str(config_file)], cwd=REPO_ROOT, label=f"pmap-{forcing}-{date}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path, label: str):
    """Exécute une commande subprocess avec logging."""
    log.info(f"[{label}] $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"[{label}] Échec (code {result.returncode})")


def _load_cold_nights(path: str | Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("cold_nights", [])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Orchestrateur vérification gel Drôme-Ardèche 2021")
    p.add_argument("--config",      default="runs/april2021/config.yml")
    p.add_argument("--cold-nights", default="runs/april2021/cold_nights.json",
                   help="Fichier JSON nuits critiques (produit par detect_cold_nights.py)")
    p.add_argument("--forcing",     choices=["era5", "cerra"], default="era5")
    p.add_argument("--step",
                   choices=["stat-downscaling", "detect", "prepare-lbc",
                             "prepare-surfex", "run-pmap", "all"],
                   default="all")
    p.add_argument("--mod-ref",  default=None, help="Forçage historique pour calibration QDM")
    p.add_argument("--obs-ref",  default=None, help="Référence pour calibration QDM")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config_path = REPO_ROOT / "downscaling" / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cold_nights = []
    cold_nights_path = REPO_ROOT / "downscaling" / args.cold_nights

    if args.step in ("detect", "all"):
        cold_nights = step_detect(cfg, args)
    elif cold_nights_path.exists():
        cold_nights = _load_cold_nights(cold_nights_path)
        log.info(f"{len(cold_nights)} nuit(s) critique(s) chargée(s) depuis {cold_nights_path}")
    else:
        # Utiliser les nuits définies dans config.yml
        cold_nights = [n for n in cfg.get("cold_nights", []) if n.get("run_pmap")]
        log.info(f"Utilisation des {len(cold_nights)} nuits depuis {args.config}")

    steps = {
        "stat-downscaling": lambda: step_stat_downscaling(cfg, args),
        "detect":           lambda: None,      # déjà fait ci-dessus
        "prepare-lbc":      lambda: step_prepare_lbc(cfg, cold_nights, args.forcing),
        "prepare-surfex":   lambda: step_prepare_surfex(cfg, cold_nights),
        "run-pmap":         lambda: step_run_pmap(cfg, cold_nights, args.forcing),
    }

    if args.step == "all":
        order = ["stat-downscaling", "detect", "prepare-lbc", "prepare-surfex", "run-pmap"]
    else:
        order = [args.step]

    for s in order:
        if s == "detect":
            continue  # déjà exécuté
        log.info(f"━━━ Étape : {s} ━━━")
        steps[s]()

    log.info("Orchestration terminée.")


if __name__ == "__main__":
    main()
