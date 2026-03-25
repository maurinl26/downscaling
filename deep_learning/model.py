"""
U-Net conditionné par le MNT pour la descente d'échelle météorologique.

Architecture
------------

    Encoder (basse résolution)          DEM encoder
        ┌─────────────┐                ┌─────────────┐
        │  Conv bloc  │                │  Conv bloc  │
        │  ↓ stride 2 │                │  (fixe)     │
        └──────┬──────┘                └──────┬──────┘
               │  ← skip connection           │
    FiLM conditioning ←───────────────────────┘
        (γ·x + β)
               │
        ┌──────┴──────┐
        │  Decoder    │
        │  ↑ upsamp   │
        └─────────────┘
               │
        Output (H, W, C_met)

FiLM Layers
-----------
Feature-wise Linear Modulation (Perez et al. 2018) : à chaque niveau du U-Net,
les activations du décodeur météo sont modulées par des paramètres (γ, β) produits
par un petit réseau conditionné sur les caractéristiques du MNT :

    y = γ(DEM) · x_met + β(DEM)

Cela permet au réseau d'apprendre des corrections spécifiques à l'altitude
(lapse-rate appris), à la pente, à l'exposition (effet de Föhn, pluie
orographique, etc.).

Référence : Perez et al. (2018) FiLM: Visual Reasoning with a General
Conditioning Layer. AAAI-2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Briques de base
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv 3×3 → BN → ReLU (avec padding 'same')."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Bloc résiduel : 2× (Conv → BN → ReLU) + skip."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvBnRelu(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x).clone()) + x)


# ---------------------------------------------------------------------------
# FiLM conditioning
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

    Génère les paramètres de modulation (γ, β) depuis les features DEM,
    puis applique y = γ · x + β aux features météo.

    Parameters
    ----------
    dem_ch:
        Nombre de canaux DEM en entrée.
    met_ch:
        Nombre de canaux météo à moduler (= out_ch du bloc encoder correspondant).
    """

    def __init__(self, dem_ch: int, met_ch: int):
        super().__init__()
        hidden = max(met_ch, 32)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # pooling spatial → vecteur
            nn.Flatten(),
            nn.Linear(dem_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * met_ch),  # [γ, β] concaténés
        )
        # Init : γ=1, β=0 (identité)
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.constant_(self.fc[-1].bias[:met_ch], 1.0)   # γ
        nn.init.zeros_(self.fc[-1].bias[met_ch:])            # β

    def forward(self, x_met: torch.Tensor, x_dem: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_met.shape
        params = self.fc(x_dem)           # (B, 2C)
        gamma = params[:, :C].view(B, C, 1, 1)
        beta = params[:, C:].view(B, C, 1, 1)
        return gamma * x_met + beta


# ---------------------------------------------------------------------------
# Encodeur DEM (poids figés ou entraînables)
# ---------------------------------------------------------------------------

class DEMEncoder(nn.Module):
    """
    Encodeur CNN pour les attributs MNT.

    Produit des représentations DEM à plusieurs résolutions, utilisées
    pour conditionner chaque niveau du U-Net via FiLM.

    Parameters
    ----------
    in_ch:
        Nombre d'attributs DEM en entrée (élévation, pente, exposition, courbure).
    base_ch:
        Nombre de filtres de base.
    n_levels:
        Nombre de niveaux (doit correspondre au U-Net météo).
    """

    def __init__(self, in_ch: int = 4, base_ch: int = 32, n_levels: int = 4):
        super().__init__()
        self.n_levels = n_levels
        self.encoders = nn.ModuleList()

        ch_in = in_ch
        for i in range(n_levels):
            ch_out = base_ch * (2 ** i)
            block = nn.Sequential(
                ConvBnRelu(ch_in, ch_out),
                ResBlock(ch_out),
            )
            self.encoders.append(block)
            ch_in = ch_out

        self.downsamples = nn.ModuleList(
            [nn.MaxPool2d(2) for _ in range(n_levels - 1)]
        )

    def forward(self, dem: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns
        -------
        Liste de features DEM à chaque résolution [level_0, …, level_N-1]
        de la haute résolution vers la basse.
        """
        feats = []
        x = dem
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            feats.append(x)
            if i < self.n_levels - 1:
                x = self.downsamples[i](x)
        return feats  # [haut résolution → bas résolution]


# ---------------------------------------------------------------------------
# U-Net météo conditionné FiLM
# ---------------------------------------------------------------------------

class DownscalingUNet(nn.Module):
    """
    U-Net pour la descente d'échelle météorologique conditionné par le MNT.

    Paramètres
    ----------
    met_in_ch:
        Canaux météo en entrée (variables ERA5/CERRA).
    met_out_ch:
        Canaux météo en sortie (variables haute résolution). Défaut = met_in_ch.
    dem_in_ch:
        Canaux attributs MNT (élévation, pente, aspect, courbure).
    base_ch:
        Nombre de filtres de base (doublé à chaque niveau d'encodeur).
    n_levels:
        Profondeur du U-Net.
    use_film:
        Active le conditionnement FiLM par le MNT. Désactiver pour ablation.
    """

    def __init__(
        self,
        met_in_ch: int = 5,
        met_out_ch: int | None = None,
        dem_in_ch: int = 4,
        base_ch: int = 64,
        n_levels: int = 4,
        use_film: bool = True,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.use_film = use_film
        met_out_ch = met_out_ch or met_in_ch

        # ---- Encodeur DEM ------------------------------------------------
        self.dem_encoder = DEMEncoder(in_ch=dem_in_ch, base_ch=base_ch // 2, n_levels=n_levels)

        # ---- Encodeur météo (downsampling) --------------------------------
        self.met_encoders = nn.ModuleList()
        self.met_pools = nn.ModuleList()
        ch_in = met_in_ch
        self.enc_channels = []
        for i in range(n_levels):
            ch_out = base_ch * (2 ** i)
            self.met_encoders.append(nn.Sequential(
                ConvBnRelu(ch_in, ch_out),
                ResBlock(ch_out),
                ConvBnRelu(ch_out, ch_out),
            ))
            if i < n_levels - 1:
                self.met_pools.append(nn.MaxPool2d(2))
            self.enc_channels.append(ch_out)
            ch_in = ch_out

        # ---- FiLM layers -------------------------------------------------
        if use_film:
            dem_channels = [base_ch // 2 * (2 ** i) for i in range(n_levels)]
            self.film_layers = nn.ModuleList([
                FiLMLayer(dem_ch, met_ch)
                for dem_ch, met_ch in zip(dem_channels, self.enc_channels)
            ])

        # ---- Décodeur météo (upsampling + skip connections) ---------------
        self.met_decoders = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(n_levels - 1, 0, -1):
            ch_in_dec = self.enc_channels[i] + self.enc_channels[i - 1]
            ch_out_dec = self.enc_channels[i - 1]
            self.up_convs.append(
                nn.ConvTranspose2d(self.enc_channels[i], self.enc_channels[i], 2, stride=2)
            )
            self.met_decoders.append(nn.Sequential(
                ConvBnRelu(ch_in_dec, ch_out_dec),
                ResBlock(ch_out_dec),
            ))

        # ---- Tête de sortie -----------------------------------------------
        self.head = nn.Conv2d(base_ch, met_out_ch, 1)

    def forward(self, x_met: torch.Tensor, x_dem: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_met:
            Champs météo basse résolution (B, C_met, H, W).
        x_dem:
            Attributs MNT haute résolution (B, C_dem, H, W).

        Returns
        -------
        Champs météo haute résolution (B, C_met_out, H, W).
        """
        # Encode DEM à toutes les résolutions
        dem_feats = self.dem_encoder(x_dem)  # liste du plus fin au plus grossier

        # ---- Encodage météo -----------------------------------------------
        enc_feats = []
        x = x_met
        for i, (enc, dem_f) in enumerate(zip(self.met_encoders, dem_feats)):
            x = enc(x)
            if self.use_film:
                x = self.film_layers[i](x, dem_f)
            enc_feats.append(x)
            if i < self.n_levels - 1:
                x = self.met_pools[i](x)

        # ---- Décodage avec skip connections --------------------------------
        x = enc_feats[-1]
        for i, (up, dec) in enumerate(zip(self.up_convs, self.met_decoders)):
            x = up(x)
            skip = enc_feats[self.n_levels - 2 - i]
            # Ajustement de taille si nécessaire (padding)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Variante légère : SRCNN conditionné (pour test rapide)
# ---------------------------------------------------------------------------

class LightSRCNN(nn.Module):
    """
    Super-Resolution CNN simplifié conditionné par le MNT.

    Adapté pour les tests rapides ou les domaines de petite taille.
    Pas de U-Net : simple chaîne convolutive avec concaténation DEM.
    """

    def __init__(self, met_in_ch: int = 5, dem_in_ch: int = 4, met_out_ch: int | None = None):
        super().__init__()
        met_out_ch = met_out_ch or met_in_ch
        in_ch = met_in_ch + dem_in_ch

        self.net = nn.Sequential(
            ConvBnRelu(in_ch, 64),
            ResBlock(64),
            ResBlock(64),
            ConvBnRelu(64, 32),
            nn.Conv2d(32, met_out_ch, 1),
        )

    def forward(self, x_met: torch.Tensor, x_dem: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_met, x_dem], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    architecture: str = "unet",
    met_in_ch: int = 5,
    dem_in_ch: int = 4,
    base_ch: int = 64,
    n_levels: int = 4,
    use_film: bool = True,
) -> nn.Module:
    """
    Construit le modèle selon l'architecture choisie.

    Parameters
    ----------
    architecture:
        'unet' (défaut) ou 'srcnn' (modèle léger).
    """
    if architecture == "unet":
        model = DownscalingUNet(
            met_in_ch=met_in_ch,
            dem_in_ch=dem_in_ch,
            base_ch=base_ch,
            n_levels=n_levels,
            use_film=use_film,
        )
    elif architecture == "srcnn":
        model = LightSRCNN(met_in_ch=met_in_ch, dem_in_ch=dem_in_ch)
    else:
        raise ValueError(f"Architecture inconnue : {architecture}. Choisir 'unet' ou 'srcnn'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modèle {architecture} : {n_params:,} paramètres entraînables.")
    return model
