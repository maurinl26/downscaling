#!/usr/bin/env python
"""
Sync objets entre le local et Scaleway Object Storage (S3-compatible).

Sert au flux d'entraînement pro (cf. docs/infra_pro.md) : pousser les tuiles
d'entraînement + DEM vers S3 une fois, puis le pod RunPod fait `pull` au démarrage
et `push` du checkpoint à la fin — supprime l'upload manuel sur le volume réseau.

S3-compatible générique : fonctionne avec n'importe quel endpoint (Scaleway,
AWS, MinIO) via variables d'environnement.

⚠ Prérequis Scaleway (à créer une fois, pas encore fait) : un projet + un bucket
Object Storage (région fr-par) + une paire de clés API. Voir docs/infra_pro.md §3.

Variables d'environnement :
    SCW_ACCESS_KEY      clé d'accès
    SCW_SECRET_KEY      clé secrète
    SCW_S3_ENDPOINT     défaut https://s3.fr-par.scw.cloud
    SCW_S3_REGION       défaut fr-par

Usage :
    uv run python downscaling/scripts/s3_sync.py push data/training/ s3://karpos-downscaling/training/drome_ardeche/2015-2021/
    uv run python downscaling/scripts/s3_sync.py pull s3://karpos-downscaling/training/drome_ardeche/2015-2021/ data/training/
    uv run python downscaling/scripts/s3_sync.py push checkpoints/drome_ardeche/best_model.ckpt s3://karpos-downscaling/artifacts/run123/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_ENDPOINT = "https://s3.fr-par.scw.cloud"
DEFAULT_REGION = "fr-par"


def _client():
    try:
        import boto3
    except ImportError:
        sys.exit("boto3 requis : uv sync --extra cloud  (ou pip install boto3)")
    access = os.environ.get("SCW_ACCESS_KEY")
    secret = os.environ.get("SCW_SECRET_KEY")
    if not (access and secret):
        sys.exit("SCW_ACCESS_KEY / SCW_SECRET_KEY manquants (créer une clé API Scaleway).")
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("SCW_S3_ENDPOINT", DEFAULT_ENDPOINT),
        region_name=os.environ.get("SCW_S3_REGION", DEFAULT_REGION),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
    )


def _split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        sys.exit(f"URI S3 invalide (attendu s3://bucket/prefix) : {uri}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    return bucket, key


def _iter_files(local: Path):
    if local.is_file():
        yield local, local.name
    else:
        for p in sorted(local.rglob("*")):
            if p.is_file():
                yield p, str(p.relative_to(local))


def push(local: str, uri: str) -> None:
    s3 = _client()
    bucket, prefix = _split_uri(uri)
    local_path = Path(local)
    n = 0
    for path, rel in _iter_files(local_path):
        if prefix.endswith("/"):
            key = f"{prefix}{rel}"  # dossier S3 explicite
        elif local_path.is_dir():
            key = f"{prefix}/{rel}"  # dir local → préfixe
        else:
            key = prefix  # fichier → clé exacte
        s3.upload_file(str(path), bucket, key)
        n += 1
        print(f"  ↑ {path}  →  s3://{bucket}/{key}")
    print(f"[push] {n} fichier(s) → s3://{bucket}/{prefix}")


def pull(uri: str, local: str) -> None:
    s3 = _client()
    bucket, prefix = _split_uri(uri)
    out = Path(local)
    out.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix) :].lstrip("/") if prefix else key
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            n += 1
            print(f"  ↓ s3://{bucket}/{key}  →  {dest}")
    print(f"[pull] {n} fichier(s) ← s3://{bucket}/{prefix}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync local ↔ Scaleway S3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="local → S3")
    p.add_argument("local")
    p.add_argument("uri")
    g = sub.add_parser("pull", help="S3 → local")
    g.add_argument("uri")
    g.add_argument("local")
    args = ap.parse_args()
    if args.cmd == "push":
        push(args.local, args.uri)
    else:
        pull(args.uri, args.local)


if __name__ == "__main__":
    main()
