#!/usr/bin/env python3
"""
run_prithvi_frost_reanalysis.py
================================
Réanalyse des nuits de gel via Prithvi WxC (NASA/IBM) + conditioning DEM.

Exemples d'utilisation :

  # Inférence complète avec modèle IBM Granite (pas de checkpoint requis)
  python scripts/run_prithvi_frost_reanalysis.py \\
      --config config/prithvi_wxc_drome.yml \\
      --start 2000-10-01 --end 2024-05-31 \\
      --out output/frost_reanalysis_prithvi.zarr

  # Avec checkpoint fine-tuné localement
  python scripts/run_prithvi_frost_reanalysis.py \\
      --config config/prithvi_wxc_drome.yml \\
      --checkpoint checkpoints/prithvi_adapter_drome.pt \\
      --out output/frost_reanalysis_prithvi_finetuned.zarr

  # Rolling complet sur une seule nuit (debug)
  python scripts/run_prithvi_frost_reanalysis.py \\
      --config config/prithvi_wxc_drome.yml \\
      --single-night 2021-04-07 \\
      --out output/frost_20210407.zarr

RunPod / GPU :
  Remplacer --device cpu par --device cuda dans la config YAML.
  L4 (24GB) : batch_size=4, ~15 min/année en mode frost_only.
  H100 (80GB): batch_size=16, ~4 min/année.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_learning.prithvi_wxc import FrostReanalysisRunner, FrostNightDataset, load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_prithvi_frost")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Réanalyse nuits de gel — Prithvi WxC (NASA/IBM)"
    )
    p.add_argument(
        "--config", required=True,
        help="Chemin vers le fichier YAML de configuration",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Chemin vers un checkpoint adapter fine-tuné (optionnel)",
    )
    p.add_argument(
        "--start", default=None,
        help="Date de début (YYYY-MM-DD). Surcharge la config.",
    )
    p.add_argument(
        "--end", default=None,
        help="Date de fin (YYYY-MM-DD). Surcharge la config.",
    )
    p.add_argument(
        "--out", default=None,
        help="Chemin de sortie Zarr. Surcharge la config.",
    )
    p.add_argument(
        "--single-night", default=None,
        help="Lancer uniquement sur une nuit (YYYY-MM-DD). Mode debug.",
    )
    p.add_argument(
        "--no-granite", action="store_true",
        help="Ne pas essayer IBM Granite (adapter initialisé aléatoirement).",
    )
    p.add_argument(
        "--device", default=None,
        help="Device PyTorch : 'cuda', 'cuda:0', 'cpu'. Surcharge la config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Charger la config ------------------------------------------------
    config = load_config(args.config)

    # Surcharges CLI
    if args.start:
        config.setdefault("period", {})["start"] = args.start
    if args.end:
        config.setdefault("period", {})["end"] = args.end
    if args.out:
        config.setdefault("output", {})["zarr_path"] = args.out
    if args.device:
        config["device"] = args.device

    start = config.get("period", {}).get("start", "2000-10-01")
    end   = config.get("period", {}).get("end",   "2024-05-31")
    zarr_out = config.get("output", {}).get("zarr_path", "output/frost_prithvi.zarr")

    if args.single_night:
        # Mode debug : une seule nuit
        start = args.single_night
        end   = args.single_night

    log.info(f"Période : {start} → {end}")
    log.info(f"Sortie   : {zarr_out}")

    # --- Dataset ----------------------------------------------------------
    data_cfg = config.get("data", {})
    dataset = FrostNightDataset(
        era5_path=data_cfg.get("era5_path", "data/era5/era5_drome_ardeche.zarr"),
        dem_path=data_cfg.get("dem_path",  "data/dem/copdem_drome_ardeche_200m.tif"),
        start_date=start,
        end_date=end,
        source=data_cfg.get("source", "era5"),
        climatology_path=data_cfg.get("climatology_path"),
        frost_only=config.get("period", {}).get("frost_only", True),
    )

    if len(dataset) == 0:
        log.error("Dataset vide — vérifier les chemins et la période.")
        sys.exit(1)

    log.info(
        f"Dataset prêt : {len(dataset)} paires | "
        f"LR shape {dataset.lr_shape} | HR shape {dataset.hr_shape}"
    )

    # --- Modèle -----------------------------------------------------------
    runner = FrostReanalysisRunner(config)
    model  = runner.load_model(
        checkpoint_path=args.checkpoint,
        use_granite=not args.no_granite,
    )

    # --- Inférence --------------------------------------------------------
    ds_out = runner.run(model, dataset, zarr_out)

    # --- Résumé -----------------------------------------------------------
    n_frost = int(ds_out["frost_flag"].sum().values)
    n_total = int(ds_out["frost_flag"].size)
    pct     = 100 * n_frost / n_total if n_total > 0 else 0.0

    log.info("=" * 60)
    log.info(f"Réanalyse terminée.")
    log.info(f"  Nuits de gel détectées : {n_frost}/{n_total} ({pct:.1f}%)")
    log.info(f"  Zarr écrit : {zarr_out}")
    log.info(
        "  Requête DuckDB : "
        "SELECT time, MIN(T2m_min_night) FROM read_parquet(...)"
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
