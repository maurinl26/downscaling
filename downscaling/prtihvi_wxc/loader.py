"""
loader.py — Chargement de Prithvi WxC (NASA/IBM) depuis HuggingFace
pour inférence de downscaling avec conditioning orographique (DEM).

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

# ---------------------------------------------------------------------------
# Identifiants HuggingFace
# ---------------------------------------------------------------------------
PRITHVI_WXC_REPO = "Prithvi-WxC/prithvi.wxc.2300m.v1"
# Version fine-tunée downscaling IBM Granite (×12 resolution enhancement)
GRANITE_DOWNSCALING_REPO = "ibm-granite/granite-geospatial-wxc-downscaling"

# Variables MERRA-2 utilisées par Prithvi WxC (sous-ensemble pertinent gel)
# Ordre exact attendu par le modèle en entrée
MERRA2_SURFACE_VARS = [
    "T2M",   # Temperature 2m (K)  ← variable cible
    "U10M",  # Vent zonal 10m
    "V10M",  # Vent méridional 10m
    "PS",    # Pression de surface
    "QV2M",  # Humidité spécifique 2m (effet sur gel radiatif)
    "TQV",   # Eau précipitable totale
]

MERRA2_PRESSURE_VARS = [
    "T",     # Température sur niveaux de pression (850, 925 hPa)
    "H",     # Hauteur géopotentielle
]


# ---------------------------------------------------------------------------
# CNN Adapter + DEM conditioning (architecture du papier NASA)
# ---------------------------------------------------------------------------

class DEMConditionedAdapter(nn.Module):
    """
    Adaptateur CNN branché sur la sortie de l'encodeur Prithvi WxC.
    Intègre le DEM haute résolution comme feature auxiliaire (concaténation)
    avant le PixelShuffle pour upscaling ×N.

    Architecture (Yu et al. 2025, Fig. 3) :
        backbone_output (B, C, H_lr, W_lr)
        dem_hr          (B, 3, H_hr, W_hr)  ← élévation, pente, exposition
              ↓
        [interpolate backbone → H_hr, W_hr]
        [concat dem_hr]
              ↓  Conv2d adapter
        [PixelShuffle ×scale_factor]
              ↓
        T2m haute résolution (B, 1, H_hr, W_hr)
    """

    def __init__(
        self,
        in_channels: int = 512,   # dim sortie encodeur Prithvi WxC
        dem_channels: int = 3,    # élévation + pente + exposition
        hidden_channels: int = 128,
        out_channels: int = 1,    # T2m uniquement
        scale_factor: int = 6,    # ERA5 31km → ~5km ; CERRA 5.5km → ~1km
    ):
        super().__init__()
        self.scale_factor = scale_factor

        total_in = in_channels + dem_channels

        self.adapter = nn.Sequential(
            nn.Conv2d(total_in, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            # PixelShuffle nécessite out_channels * scale_factor^2 canaux
            nn.Conv2d(
                hidden_channels,
                out_channels * scale_factor * scale_factor,
                kernel_size=3,
                padding=1,
            ),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(
        self,
        backbone_out: torch.Tensor,  # (B, C, H_lr, W_lr)
        dem_hr: torch.Tensor,        # (B, 3, H_hr, W_hr)
    ) -> torch.Tensor:
        H_hr, W_hr = dem_hr.shape[-2:]

        # Upsample backbone features à la résolution cible
        x = nn.functional.interpolate(
            backbone_out, size=(H_hr, W_hr), mode="bilinear", align_corners=False
        )

        # Concaténer DEM haute résolution
        x = torch.cat([x, dem_hr], dim=1)

        # Adapter CNN + PixelShuffle
        x = self.adapter(x)
        x = self.pixel_shuffle(x)

        return x  # (B, 1, H_hr * scale, W_hr * scale) — mais scale déjà H_hr


# ---------------------------------------------------------------------------
# Modèle complet : Prithvi WxC backbone + DEM adapter
# ---------------------------------------------------------------------------

class PrithviWxCDownscaler(nn.Module):
    """
    Wrapper inférence : Prithvi WxC (backbone gelé) + DEMConditionedAdapter.

    Usage :
        model = PrithviWxCDownscaler.from_pretrained(scale_factor=6)
        t2m_hr = model(era5_lr, dem_hr)  # (B, 1, H_hr, W_hr) en K
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: DEMConditionedAdapter,
        backbone_out_channels: int = 512,
    ):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.backbone_out_channels = backbone_out_channels

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path | None = None,
        use_granite_downscaling: bool = True,
        scale_factor: int = 6,
        device: str = "cpu",
    ) -> "PrithviWxCDownscaler":
        """
        Charge le modèle depuis HuggingFace ou un checkpoint local.

        Args:
            checkpoint_path: Chemin vers un checkpoint local fine-tuné.
                             Si None, tente de charger IBM Granite downscaling.
            use_granite_downscaling: Utilise le modèle IBM Granite fine-tuné
                                     pour downscaling (recommandé sans fine-tuning).
            scale_factor: Facteur d'upscaling spatial.
            device: "cuda", "cpu", ou "cuda:0".
        """
        try:
            from PrithviWxC.model import PrithviWxC  # type: ignore
        except ImportError:
            raise ImportError(
                "Installer le package Prithvi WxC :\n"
                "  pip install 'git+https://github.com/NASA-IMPACT/Prithvi-WxC.git'"
            )

        print(f"[PrithviWxC] Chargement backbone depuis {PRITHVI_WXC_REPO} ...")

        backbone = PrithviWxC.from_pretrained(PRITHVI_WXC_REPO)
        backbone.eval()
        # Geler le backbone — seul l'adapter est entraîné/fine-tuné
        for param in backbone.parameters():
            param.requires_grad = False

        adapter = DEMConditionedAdapter(scale_factor=scale_factor)

        model = cls(backbone=backbone, adapter=adapter)

        if checkpoint_path is not None:
            print(f"[PrithviWxC] Chargement adapter depuis {checkpoint_path} ...")
            state = torch.load(checkpoint_path, map_location=device)
            # On ne charge que l'adapter (le backbone reste HF)
            adapter_state = {
                k.replace("adapter.", ""): v
                for k, v in state.items()
                if k.startswith("adapter.")
            }
            model.adapter.load_state_dict(adapter_state)
        elif use_granite_downscaling:
            print(
                f"[PrithviWxC] Tentative chargement IBM Granite downscaling "
                f"depuis {GRANITE_DOWNSCALING_REPO} ..."
            )
            try:
                _load_granite_adapter(model, device)
            except Exception as e:
                print(
                    f"[PrithviWxC] Granite non disponible ({e}). "
                    "Adapter initialisé aléatoirement — fine-tuning requis."
                )

        model = model.to(device)
        print(f"[PrithviWxC] Modèle prêt sur {device}.")
        return model

    def forward(
        self,
        era5_t0: torch.Tensor,   # (B, C, H_lr, W_lr) — timestamp t
        era5_t1: torch.Tensor,   # (B, C, H_lr, W_lr) — timestamp t+3h
        dem_hr: torch.Tensor,    # (B, 3, H_hr, W_hr) — élévation, pente, expo
    ) -> torch.Tensor:
        """
        Retourne T2m haute résolution (B, 1, H_hr, W_hr) en Kelvin.

        Le backbone Prithvi WxC attend deux timestamps comme décrit dans
        Yu et al. (2025) : l'interpolation temporelle à 3h permet de couvrir
        un cycle journalier complet par rolling (voir inference.py).
        """
        # Concaténer les deux timestamps sur la dimension channel
        x = torch.cat([era5_t0, era5_t1], dim=1)

        # Encoder avec Prithvi WxC (sortie : features basse résolution)
        with torch.no_grad():
            backbone_features = self.backbone.encode(x)  # (B, C_enc, H_lr, W_lr)

        # Upscaling conditionné DEM
        t2m_hr = self.adapter(backbone_features, dem_hr)

        return t2m_hr


# ---------------------------------------------------------------------------
# Helper interne
# ---------------------------------------------------------------------------

def _load_granite_adapter(model: PrithviWxCDownscaler, device: str) -> None:
    """
    Tente de charger les poids de l'adapter IBM Granite downscaling.
    L'architecture exacte peut différer — adaptation automatique par clé.
    """
    weights_path = hf_hub_download(
        repo_id=GRANITE_DOWNSCALING_REPO,
        filename="model.safetensors",
    )
    import safetensors.torch as sf  # type: ignore

    state = sf.load_file(weights_path, device=device)

    # Tentative de correspondance flexible des clés
    adapter_keys = {k: v for k, v in state.items() if "adapter" in k or "pixel_shuffle" in k}
    if adapter_keys:
        model.adapter.load_state_dict(adapter_keys, strict=False)
        print(f"[PrithviWxC] {len(adapter_keys)} clés adapter chargées depuis Granite.")
    else:
        print("[PrithviWxC] Aucune clé adapter trouvée dans Granite — skip.")
