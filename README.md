# Downscaling ERA5 / CERRA → 1 km pour indices d'assurance paramétrique

Ce module propose deux approches de descente d'échelle de réanalyses atmosphériques
vers une résolution kilométrique, en tenant compte du modèle numérique de terrain (MNT).
L'objectif est de fournir des champs météorologiques à haute résolution pour le calcul
d'indices d'assurance paramétrique (gel, précipitations extrêmes, vent).

---

## Sources de données

| Source | Résolution | Période | Variables clés |
|--------|-----------|---------|----------------|
| ERA5 | ~31 km / 1h | 1940–présent | T2m, Tp, U10, V10, TP |
| CERRA | ~5.5 km / 1h | 1985–2021 | T2m, Tp, U10, V10, TP |
| EU-DEM / COP-DEM | 25–30 m | — | Élévation |
| SAFRAN (validation) | 8 km / 1h | 1958–présent | T, Precip, Vent |

---

## Structure du module

```
downscaling/
├── shared/
│   ├── loaders.py       # Chargement ERA5, CERRA, DEM
│   └── indices.py       # Indices paramétriques (gel, GDD, précip extrêmes…)
├── statistical/
│   ├── lapse_rate.py    # Correction lapse-rate (température × altitude)
│   ├── quantile_mapping.py  # QDM / EQM (biais climatologique)
│   └── pipeline.py      # Pipeline complet descente d'échelle statistique
├── deep_learning/
│   ├── dataset.py       # Dataset PyTorch (paires basse/haute résolution)
│   ├── model.py         # U-Net conditionné par le MNT (FiLM layers)
│   ├── train.py         # Boucle d'entraînement
│   └── inference.py     # Inférence sur nouvelles données
├── config/
│   └── drome_ardeche.yml
└── scripts/
    ├── run_statistical_downscaling.py
    └── run_dl_inference.py
```

---

## Approche 1 — Descente d'échelle statistique

### Température

1. **Correction lapse-rate** : `T_fin = T_grossier + γ × (z_fin – z_grossier)`
   - γ calibré mensuellement sur stations SYNOP (ou -6.5 K/km par défaut)
   - Régression linéaire multiple avec pente, exposition en options

2. **Quantile Delta Mapping (QDM)** (Cannon et al. 2015) :
   - Correction des biais climatologiques ERA5→CERRA ou ERA5→SAFRAN
   - Préserve le signal de changement climatique

### Précipitations

- **BCSD** (Bias Correction & Spatial Disaggregation) :
  1. QDM sur les quantiles mensuels
  2. Désagrégation spatiale par la méthode des analogues ou par le MNT

### Usage rapide

```bash
cd downscaling/
python scripts/run_statistical_downscaling.py \
    --era5-sl data/era5/era5_sl_20210427.nc \
    --dem     data/dem/copdem_drome_ardeche_100m.tif \
    --out     output/stat_downscaled_20210427.nc \
    --variable t2m --variable tp
```

---

## Approche 2 — Descente d'échelle par deep learning

La descente d'échelle par deep learning utilisant `prithvi_wxc` a été ajoutée au dépôt pour prendre en compte l'orographie.

### Architecture : U-Net conditionné par le MNT

```
Input (basse résolution)          DEM haute résolution
  ERA5 / CERRA champs ──────────┐    ┌── élévation, pente, exposition
                                 ▼    ▼
                        ┌──────────────────┐
                        │   Encoder CNN    │  ← FiLM conditioning (DEM)
                        │   Skip conns     │
                        │   Decoder CNN    │
                        └────────┬─────────┘
                                 ▼
                        Champs haute résolution (1 km)
                          T2m, Tmin, Tmax, TP, U10
```

**FiLM (Feature-wise Linear Modulation)** : le MNT module les activations
intermédiaires du U-Net via γ·x + β appris, ce qui permet au réseau d'apprendre
la dépendance altitude–température et les effets orographiques sur les précipitations.

### Données d'entraînement recommandées

- **Input** : ERA5 (31 km) agrégé ou CERRA (5.5 km) → rééchantillonné à basse résolution
- **Target** : CERRA (5.5 km) ou SAFRAN (8 km) ou analyse AROME (1.3 km)
- **Conditionnement** : EU-DEM / COP-DEM rééchantillonné à la résolution cible

### Entraînement

```bash
python scripts/run_dl_train.py \
    --config config/drome_ardeche.yml \
    --data-dir data/training/ \
    --epochs 100 --batch-size 8
```

### Inférence

```bash
python scripts/run_dl_inference.py \
    --config  config/drome_ardeche.yml \
    --checkpoint checkpoints/best_model.pt \
    --era5-sl data/era5/era5_sl_20210427.nc \
    --dem     data/dem/copdem_drome_ardeche_100m.tif \
    --out     output/dl_downscaled_20210427.nc
```

---

## Indices paramétriques disponibles

| Indice | Description | Trigger type |
|--------|-------------|-------------|
| `frost_days` | Nb jours Tmin < 0 °C | Assurance gel |
| `frost_hours` | Nb heures T2m < seuil | Gel vigne/arbo |
| `spring_frost` | Gel après débourrement (GDD > seuil) | Gel printanier |
| `gdd` | Growing Degree Days (somme thermique) | Agronomie |
| `extreme_precip_days` | Jours précip > seuil (mm/j) | Inondation |
| `dry_spell` | Nb jours consécutifs précip < 1 mm | Sécheresse |
| `wind_storm` | Nb heures rafales > seuil | Tempête |
| `heatwave` | Tmax > seuil sur N jours consécutifs | Canicule |

---

## Références

- Cannon A.J. et al. (2015) *Bias Correction of GCM Precipitation by Quantile
  Delta Mapping*. J. Climate 28, 6938–6959.
- Wang F. et al. (2021) *FILM: Visual Reasoning with a General Conditioning Layer*.
  AAAI 2018. [FiLM layers]
- Ronneberger O. et al. (2015) *U-Net*. MICCAI.
- Baño-Medina J. et al. (2020) *Configuration and intercomparison of deep learning
  neural models for statistical downscaling*. Geosci. Model Dev. 13, 2109–2124.
- Höhlein K. et al. (2020) *A comparative study of convolutional neural network
  models for wind field downscaling*. Met. Apps 27, e1961.
