"""
loader.py — Chargement du **vrai** modèle Prithvi WxC (NASA/IBM) + tête de
downscaling conditionnée par le MNT.

Le backbone est construit et chargé via l'API officielle du package
``PrithviWxC`` (``PrithviWxC.configs.load_model``), qui télécharge config +
facteurs d'échelle (climatologie) + poids depuis HuggingFace et instancie le
modèle réel. **Aucune réimplémentation** de la construction : on délègue.

Architecture réelle (≠ ancien placeholder) :
  - Prithvi WxC est un **prévisionniste** masqué : ``forward(batch) → ŷ`` de
    forme ``[B, C, lat, lon]`` (même grille MERRA-2 que l'entrée), où ``batch``
    contient ``x, y, static, climate, input_time, lead_time``.
  - La **descente d'échelle** est une tête CNN conditionnée DEM appliquée *sur la
    prévision* (`DEMConditionedAdapter`), pas sur des features d'encodeur.

Référence :
  Schmude et al. (2024) "Prithvi WxC: Foundation Model for Weather and Climate"
  arXiv:2409.13598
  Yu et al. (2025) "Fine-Tuning Foundational Models for Downscaling..." NASA NTRS 20250006603
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from downscaling.paths import MODELSTORE

# ---------------------------------------------------------------------------
# Identifiants HuggingFace
# ---------------------------------------------------------------------------
PRITHVI_WXC_REPO = "Prithvi-WxC/prithvi.wxc.2300m.v1"
# Version fine-tunée downscaling IBM Granite (architecture propre, cf. issue #1).
GRANITE_DOWNSCALING_REPO = "ibm-granite/granite-geospatial-wxc-downscaling"

# Sous-dossier de modelstore/ où l'API officielle télécharge config/scalers/poids.
PRITHVI_DATA_DIR = MODELSTORE / "prithvi-wxc"

_MODELSTORE_DIRS = {
    PRITHVI_WXC_REPO: "prithvi-wxc",
    GRANITE_DOWNSCALING_REPO: "granite-downscaling",
}


def resolve_device(device: str = "auto") -> str:
    """Résout ``"auto"`` vers cuda / mps / cpu selon le matériel disponible."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_artifact(repo_id: str, filename: str) -> str:
    """Chemin local d'un poids : ``modelstore/`` d'abord, sinon cache HF."""
    subdir = _MODELSTORE_DIRS.get(repo_id, repo_id.replace("/", "_"))
    local = MODELSTORE / subdir / filename
    if local.exists():
        return str(local)
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_prithvi_backbone(
    config_name: str = "large",
    data_dir: str | Path | None = None,
    load_weights: bool = True,
    device: str = "auto",
) -> nn.Module:
    """Construit + charge le backbone Prithvi WxC réel (API officielle).

    Délègue à ``PrithviWxC.configs.load_model`` : télécharge config, facteurs
    d'échelle (``climatology/musigma_*.nc``, ``anomaly_variance_*.nc``) et poids
    dans ``data_dir`` (défaut : ``modelstore/prithvi-wxc``), puis instancie le
    ``PrithviWxC`` réel et charge le ``state_dict`` (``strict=True``).

    Args:
        config_name: "small" (jouet), "large" (2,3 B) ou "large_rollout".
        data_dir: cache des artefacts ; défaut ``modelstore/prithvi-wxc``.
        load_weights: télécharge et charge les poids pré-entraînés.
        device: "cuda" / "mps" / "cpu" / "auto".

    Le backbone est renvoyé **gelé** (``requires_grad=False``) et en ``eval()``.
    """
    try:
        from PrithviWxC.configs import load_model
    except ImportError as e:
        raise ImportError(
            "Package Prithvi WxC requis :\n"
            "  uv pip install 'git+https://github.com/NASA-IMPACT/Prithvi-WxC.git'\n"
            "(inclus dans l'extra `prithvi`)."
        ) from e

    data_dir = Path(data_dir or PRITHVI_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_model(config_name, data_dir, load_weights=load_weights)
    backbone = backbone.to(resolve_device(device)).eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone


# ---------------------------------------------------------------------------
# Tête de downscaling conditionnée DEM
# ---------------------------------------------------------------------------

class DEMConditionedAdapter(nn.Module):
    """
    Tête CNN appliquée sur la **prévision** du backbone, conditionnée par le MNT.

    Architecture :
        forecast (B, C, H_lr, W_lr)  ← sortie Prithvi WxC (C = in_channels)
        dem_hr   (B, 3, H_hr, W_hr)  ← élévation, pente, exposition
              ↓  [interpolate forecast → H_hr, W_hr] + [concat dem_hr]
              ↓  Conv2d → PixelShuffle ×scale_factor
        T2m haute résolution (B, out_channels, H_hr·scale, W_hr·scale)

    ``in_channels`` correspond au **nombre de canaux de la prévision** du
    backbone (cf. ``PrithviWxCDownscaler.from_pretrained``), plus de valeur
    codée en dur.
    """

    def __init__(
        self,
        in_channels: int,
        dem_channels: int = 3,
        hidden_channels: int = 128,
        out_channels: int = 1,    # T2m uniquement
        scale_factor: int = 6,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        total_in = in_channels + dem_channels

        self.adapter = nn.Sequential(
            nn.Conv2d(total_in, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                out_channels * scale_factor * scale_factor,
                kernel_size=3,
                padding=1,
            ),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, forecast: torch.Tensor, dem_hr: torch.Tensor) -> torch.Tensor:
        H_hr, W_hr = dem_hr.shape[-2:]
        x = nn.functional.interpolate(
            forecast, size=(H_hr, W_hr), mode="bilinear", align_corners=False
        )
        x = torch.cat([x, dem_hr], dim=1)
        x = self.adapter(x)
        return self.pixel_shuffle(x)


# ---------------------------------------------------------------------------
# Modèle complet : backbone Prithvi WxC réel + tête DEM
# ---------------------------------------------------------------------------

class PrithviWxCDownscaler(nn.Module):
    """
    Prithvi WxC (backbone gelé, prévisionniste) + tête de downscaling DEM.

    ``forward(batch, dem_hr)`` exécute la prévision réelle ``backbone(batch)``
    puis la descente d'échelle conditionnée MNT.

    Usage ::

        model = PrithviWxCDownscaler.from_pretrained(config_name="large", scale_factor=6)
        t2m_hr = model(batch, dem_hr)   # (B, 1, H_hr·scale, W_hr·scale) en K

    où ``batch`` est le dict attendu par Prithvi WxC
    (``x, y, static, climate, input_time, lead_time``).
    """

    def __init__(self, backbone: nn.Module, adapter: DEMConditionedAdapter):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    @classmethod
    def from_pretrained(
        cls,
        config_name: str = "large",
        scale_factor: int = 6,
        out_channels: int = 1,
        device: str = "auto",
        load_weights: bool = True,
        data_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> PrithviWxCDownscaler:
        """
        Charge le backbone réel (API officielle) et construit la tête DEM.

        Args:
            config_name: "large" (2,3 B), "small" (jouet) ou "large_rollout".
            scale_factor: facteur d'upscaling de la tête de downscaling.
            out_channels: nombre de variables produites (T2m → 1).
            device: "cuda"/"mps"/"cpu"/"auto".
            load_weights: charge les poids pré-entraînés du backbone.
            data_dir: cache des artefacts (défaut ``modelstore/prithvi-wxc``).
            checkpoint_path: checkpoint local de la tête DEM fine-tunée (optionnel).
        """
        device = resolve_device(device)
        backbone = load_prithvi_backbone(
            config_name=config_name, data_dir=data_dir,
            load_weights=load_weights, device=device,
        )
        # in_channels de la tête = nb de canaux de la prévision backbone.
        adapter = DEMConditionedAdapter(
            in_channels=backbone.in_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
        ).to(device)

        model = cls(backbone=backbone, adapter=adapter)

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=device)
            adapter_state = {
                k.replace("adapter.", ""): v
                for k, v in state.items()
                if k.startswith("adapter.")
            }
            model.adapter.load_state_dict(adapter_state)

        return model.to(device)

    def forward(self, batch: dict[str, torch.Tensor], dem_hr: torch.Tensor) -> torch.Tensor:
        """
        Prévision Prithvi WxC réelle puis descente d'échelle conditionnée DEM.

        Args:
            batch: dict d'entrée Prithvi WxC (``x, y, static, climate,
                   input_time, lead_time``).
            dem_hr: (B, 3, H_hr, W_hr) — élévation, pente, exposition.

        Returns:
            (B, out_channels, H_hr·scale, W_hr·scale) — T2m haute résolution.
        """
        # Backbone gelé → no_grad (mémoire) ; la tête DEM reste entraînable.
        with torch.no_grad():
            forecast = self.backbone(batch)   # (B, C, lat, lon)
        return self.adapter(forecast, dem_hr)
