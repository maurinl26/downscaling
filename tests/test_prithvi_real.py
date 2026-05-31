"""Tests d'intégration du **vrai** backbone Prithvi WxC (issue #1).

Construit un ``PrithviWxC`` réel sur une **config minuscule** (pas de poids, pas
de réseau) et valide le câblage de bout en bout de ``PrithviWxCDownscaler`` :
prévision backbone réelle ``forward(batch) → [B, C, H, W]`` puis tête de
downscaling conditionnée DEM. Prouve que l'intégration cible la vraie API
(``forward(batch: dict)``), et non l'ancien ``.encode()`` fictif.

Sauté si le package ``PrithviWxC`` n'est pas installé (extra ``prithvi``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PrithviWxC")

from PrithviWxC.model import PrithviWxC

from downscaling.prtihvi_wxc.loader import DEMConditionedAdapter, PrithviWxCDownscaler

# Config jouet : grille minuscule, contraintes respectées
# (n_lats % mask_unit == 0, mask_unit % patch == 0).
C, C_STATIC, T = 4, 2, 2
H_LR, W_LR = 6, 8


def _tiny_backbone() -> PrithviWxC:
    return PrithviWxC(
        in_channels=C, input_size_time=T, in_channels_static=C_STATIC,
        input_scalers_mu=torch.zeros(C), input_scalers_sigma=torch.ones(C),
        input_scalers_epsilon=0.0,
        static_input_scalers_mu=torch.zeros(C_STATIC),
        static_input_scalers_sigma=torch.ones(C_STATIC),
        static_input_scalers_epsilon=0.0,
        output_scalers=torch.ones(C),
        n_lats_px=H_LR, n_lons_px=W_LR, patch_size_px=(2, 2), mask_unit_size_px=(2, 2),
        mask_ratio_inputs=0.0, mask_ratio_targets=0.0,
        embed_dim=32, n_blocks_encoder=1, n_blocks_decoder=1,
        mlp_multiplier=2, n_heads=2, dropout=0.0, drop_path=0.0,
        parameter_dropout=0.0, residual="none", masking_mode="local",
        positional_encoding="absolute",
    ).eval()


def _batch() -> dict:
    return {
        "x": torch.randn(1, T, C, H_LR, W_LR),
        "y": torch.randn(1, C, H_LR, W_LR),
        "static": torch.randn(1, C_STATIC, H_LR, W_LR),
        "climate": torch.randn(1, C, H_LR, W_LR),
        "input_time": torch.zeros(1),
        "lead_time": torch.zeros(1),
    }


def test_real_backbone_forward_is_a_forecast():
    """Le vrai backbone consomme un batch dict et rend une prévision même grille."""
    backbone = _tiny_backbone()
    with torch.no_grad():
        out = backbone(_batch())
    assert tuple(out.shape) == (1, C, H_LR, W_LR)


def test_downscaler_end_to_end_shape():
    """forward(batch, dem_hr) : prévision réelle → tête DEM → champ haute résolution."""
    backbone = _tiny_backbone()
    scale = 2
    adapter = DEMConditionedAdapter(in_channels=backbone.in_channels, scale_factor=scale)
    model = PrithviWxCDownscaler(backbone=backbone, adapter=adapter)

    H_HR, W_HR = 12, 16
    dem_hr = torch.randn(1, 3, H_HR, W_HR)
    out = model(_batch(), dem_hr)
    # tête : interpole la prévision → (H_HR, W_HR), concat DEM, pixelshuffle ×scale.
    assert tuple(out.shape) == (1, 1, H_HR * scale, W_HR * scale)


def test_adapter_dim_derives_from_backbone():
    """La tête est dimensionnée sur les canaux de la prévision (pas de 512 codé)."""
    backbone = _tiny_backbone()
    adapter = DEMConditionedAdapter(in_channels=backbone.in_channels, scale_factor=2)
    first_conv = adapter.adapter[0]
    assert first_conv.in_channels == backbone.in_channels + 3  # + DEM (élévation/pente/expo)


def test_gradients_flow_to_head_only():
    """Backbone gelé (forward sous no_grad) ; seule la tête DEM reçoit des gradients."""
    backbone = _tiny_backbone()
    adapter = DEMConditionedAdapter(in_channels=backbone.in_channels, scale_factor=2)
    model = PrithviWxCDownscaler(backbone=backbone, adapter=adapter)

    out = model(_batch(), torch.randn(1, 3, 12, 16))
    out.mean().backward()

    assert all(p.grad is not None for p in model.adapter.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())
