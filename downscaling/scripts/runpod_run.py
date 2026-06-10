#!/usr/bin/env python
"""
Lance un run downscaling sur RunPod (self-contained, sans image custom).

Crée un pod GPU avec une image PyTorch publique + le volume réseau, et exécute un
bootstrap : clone la branche (repo public) → `uv sync` → pull data S3 → run → push S3.
Évite l'image Scaleway privée (pas de creds registry) et l'auth GitHub (repo public).

Secrets via env : RUNPOD_API_KEY, SCW_ACCESS_KEY, SCW_SECRET_KEY, SCW_S3_ENDPOINT.

Usage :
  RUNPOD_API_KEY=... SCW_ACCESS_KEY=... SCW_SECRET_KEY=... \
  uv run python downscaling/scripts/runpod_run.py --gpu "A100" --name caltaw \
    --run "uv run python downscaling/scripts/run_calibration.py cluster=cloud ..."
  uv run python downscaling/scripts/runpod_run.py --status <POD_ID>
  uv run python downscaling/scripts/runpod_run.py --logs <POD_ID>
  uv run python downscaling/scripts/runpod_run.py --stop <POD_ID>
"""

from __future__ import annotations

import argparse
import os
import sys

# Image officielle PyTorch : exécute le CMD directement (pas d'entrypoint Jupyter qui
# ignore docker_args, contrairement aux images runpod/*). torch (PyPI) embarque sa CUDA.
IMAGE = "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
VOLUME_NAME = "downscaling-workspace"
MOUNT = "/workspace"
BRANCH = "exp/frost-weighted-loss"
REPO = "https://github.com/maurinl26/downscaling.git"
S3_PREFIX = "s3://karpos-backtest-data/downscaling"

# Données tirées de S3 avant le run (s3_subdir -> local_dir)
PULLS = [
    (f"{S3_PREFIX}/checkpoints/mv_run/", "checkpoints/mv_run/"),
    (f"{S3_PREFIX}/cerra_fine_mv_cal/", "data/cerra_fine_mv_cal/"),
    (f"{S3_PREFIX}/cerra_fine_mv_test/", "data/cerra_fine_mv_test/"),
    (f"{S3_PREFIX}/sencrop/", "data/sencrop/"),
    (f"{S3_PREFIX}/dem/", "data/training/"),
]


def _key():
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        sys.exit("RUNPOD_API_KEY manquant")
    return k


def _bootstrap(task: str) -> str:
    """docker_args minimal : clone la branche puis délègue au script bootstrap du repo
    (tout le quoting Hydra vit dans le script → pas d'enfer de quoting via l'API)."""
    steps = [
        "set -e",
        "export DEBIAN_FRONTEND=noninteractive",
        "(command -v git && command -v curl) >/dev/null 2>&1 || "
        "(apt-get update && apt-get install -y git curl)",
        "cd /workspace",
        f"(cd downscaling && git fetch && git checkout {BRANCH} && git pull) || "
        f"git clone -b {BRANCH} {REPO} downscaling",
        "cd downscaling",
        f"bash scripts/runpod_bootstrap.sh {task}",
    ]
    return "bash -lc " + repr(" && ".join(steps))


def create(args):
    import runpod
    runpod.api_key = _key()
    vols = _volume_id(runpod)
    boot = _bootstrap(args.task)
    for gpu in _gpu_ids(runpod, args.gpu):
        try:
            pod = runpod.create_pod(
                name=f"dl-{args.name}", image_name=IMAGE, gpu_type_id=gpu, gpu_count=1,
                container_disk_in_gb=40, network_volume_id=vols, volume_mount_path=MOUNT,
                env={"SCW_ACCESS_KEY": os.environ.get("SCW_ACCESS_KEY", ""),
                     "SCW_SECRET_KEY": os.environ.get("SCW_SECRET_KEY", ""),
                     "SCW_S3_ENDPOINT": os.environ.get("SCW_S3_ENDPOINT", "https://s3.fr-par.scw.cloud")},
                docker_args=boot,
            )
            print(f"Pod créé : {pod['id']}  (gpu {gpu})")
            print(f"Suivi : uv run python downscaling/scripts/runpod_run.py --logs {pod['id']}")
            return
        except Exception as e:
            print(f"  {gpu} indispo ({str(e)[:60]}), suivant…")
    sys.exit("Aucun GPU dispo.")


def _volume_id(runpod):
    from runpod.api.graphql import run_graphql_query
    r = run_graphql_query("{ myself { networkVolumes { id name } } }")
    for v in r["data"]["myself"]["networkVolumes"]:
        if v["name"] == VOLUME_NAME:
            return v["id"]
    sys.exit(f"Volume {VOLUME_NAME} introuvable")


def _gpu_ids(runpod, name):
    gpus = runpod.get_gpus()
    m = [g["id"] for g in gpus if name.lower() in g["id"].lower()]
    return sorted(m, key=lambda x: ("80" not in x, x))  # 80GB d'abord


def main():
    ap = argparse.ArgumentParser(description="Run downscaling sur RunPod")
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--name", default="run")
    ap.add_argument("--task", default="smoke", help="smoke | caltaw | unet-train")
    ap.add_argument("--status"); ap.add_argument("--logs"); ap.add_argument("--stop")
    args = ap.parse_args()
    import runpod
    runpod.api_key = _key()
    if args.status:
        print(runpod.get_pod(args.status)); return
    if args.stop:
        runpod.terminate_pod(args.stop); print("stoppé"); return
    if args.logs:
        print("Logs via le dashboard RunPod (pod → Logs) ou SSH. Pod:", args.logs); return
    create(args)


if __name__ == "__main__":
    main()
