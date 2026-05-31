"""fetch_pretrained.py — pré-télécharge les poids Prithvi WxC dans ``modelstore/``.

Récupère les artefacts HuggingFace (backbone Prithvi WxC, adapter IBM Granite
downscaling, et leurs annexes : config, scalers de normalisation, climatologie)
dans une arborescence locale ``modelstore/`` (cf. :data:`downscaling.paths.MODELSTORE`),
ignorée par git. Sépare ainsi le téléchargement (une fois) de l'inférence, et
permet de pré-charger un volume réseau RunPod via ``DOWNSCALING_MODELSTORE``.

Usage (console entry point) ::

    fetch-pretrained --list                # affiche le plan, ne télécharge rien
    fetch-pretrained                       # télécharge tout dans modelstore/
    fetch-pretrained --only granite        # un seul artefact
    DOWNSCALING_MODELSTORE=/workspace/models fetch-pretrained

Le manifeste ci-dessous liste les fichiers par dépôt. ``snapshot_download`` avec
``allow_patterns`` évite d'avoir à figer chaque nom de fichier exact (les annexes
de scalers/climatologie évoluent selon les releases).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from downscaling.paths import MODELSTORE

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Artifact:
    """Un artefact de poids à matérialiser sous ``modelstore/<local_dir>``.

    ``mode`` :
      - ``"official"`` : délègue à l'API Prithvi WxC (``load_prithvi_backbone``),
        qui télécharge config + scalers + poids dans la disposition attendue par
        ``PrithviWxC.configs.load_model`` (garantit le bon layout).
      - ``"snapshot"`` : ``huggingface_hub.snapshot_download`` filtré par patterns.
    """

    key: str
    repo_id: str
    local_dir: str
    mode: str = "snapshot"
    config_name: str = ""
    patterns: list[str] = field(default_factory=list)
    note: str = ""

    def dest(self, root: Path) -> Path:
        return root / self.local_dir


# Manifeste des poids requis.
MANIFEST: tuple[Artifact, ...] = (
    Artifact(
        key="backbone",
        repo_id="Prithvi-WxC/prithvi.wxc.2300m.v1",
        local_dir="prithvi-wxc",
        mode="official",
        config_name="large",
        note="Backbone foundation (2,3 B) + config + scalers (via API officielle)",
    ),
    Artifact(
        key="granite",
        repo_id="ibm-granite/granite-geospatial-wxc-downscaling",
        local_dir="granite-downscaling",
        patterns=["*.safetensors", "*.yaml", "*.json"],
        note="Modèle downscaling IBM Granite (architecture distincte — cf. issue #1)",
    ),
)


def plan(only: str | None = None, root: Path = MODELSTORE) -> list[Artifact]:
    """Retourne les artefacts à traiter (filtrés par ``--only``)."""
    if only is None:
        return list(MANIFEST)
    selected = [a for a in MANIFEST if a.key == only]
    if not selected:
        keys = ", ".join(a.key for a in MANIFEST)
        sys.exit(f"ERROR: artefact inconnu '{only}'. Choix : {keys}")
    return selected


def fetch_one(artifact: Artifact, root: Path) -> Path:
    """Télécharge un artefact dans ``root/<local_dir>`` (idempotent)."""
    dest = artifact.dest(root)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("⇣ %s → %s", artifact.repo_id, dest)

    if artifact.mode == "official":
        # API officielle Prithvi WxC : télécharge config + scalers + poids dans
        # la disposition attendue par load_model (cf. loader.load_prithvi_backbone).
        from downscaling.prtihvi_wxc.loader import load_prithvi_backbone

        load_prithvi_backbone(
            config_name=artifact.config_name, data_dir=dest,
            load_weights=True, device="cpu",
        )
        return dest

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=artifact.repo_id,
        local_dir=str(dest),
        allow_patterns=artifact.patterns or None,
    )
    return dest


def _print_plan(artifacts: list[Artifact], root: Path) -> None:
    print(f"modelstore : {root}\n")
    for a in artifacts:
        present = "✓ présent" if a.dest(root).exists() else "absent"
        detail = f"config={a.config_name}" if a.mode == "official" else f"patterns={a.patterns}"
        print(f"  [{a.key}] {a.repo_id}  (mode={a.mode})")
        print(f"      → {a.dest(root)}  ({present})")
        print(f"      {detail}")
        print(f"      {a.note}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-télécharge les poids Prithvi WxC")
    parser.add_argument("--only", metavar="KEY",
                        help=f"Un seul artefact ({', '.join(a.key for a in MANIFEST)})")
    parser.add_argument("--list", "--dry-run", action="store_true", dest="dry_run",
                        help="Affiche le plan sans télécharger")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    artifacts = plan(args.only)
    if args.dry_run:
        _print_plan(artifacts, MODELSTORE)
        return

    for artifact in artifacts:
        dest = fetch_one(artifact, MODELSTORE)
        log.info("✓ %s prêt (%s)", artifact.key, dest)
    log.info("Terminé. modelstore = %s", MODELSTORE)


if __name__ == "__main__":
    main()
