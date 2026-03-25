# Runs de vérification — Gel de printemps Drôme-Ardèche, Avril 2021

## Framework opérationnel

```
                 ERA5 (31 km) / CERRA (5.5 km)
                           │
               ┌───────────┴───────────┐
               │                       │
     Descente statistique         Descente DL
     (lapse-rate + QDM)           (U-Net + FiLM)
               │                       │
               └───────────┬───────────┘
                           │
                  Détection nuits froides
                  Tmin < −2 °C après GDD > 50
                           │
                  Nuits critiques identifiées
                  (25-26, 26-27, 27-28, 28-29 avril)
                           │
                           ▼
              Descente dynamique PMAP-LES (1 km)
              ┌────────────┬────────────────────┐
              │            │                    │
          ERA5-forced  CERRA-forced        CERRA+SURFEX
          (baseline)   (meilleure IC)     (production)
              │            │                    │
              └────────────┴────────────────────┘
                           │
                   Comparaison & vérification
                   SYNOP stations: Valence, Montélimar,
                   Aubenas, Privas, Romans-sur-Isère
```

## Nuits critiques identifiées — Gel de printemps 2021

| Nuit          | Tmin ERA5 (vallées) | Tmin CERRA (vallées) | Statut |
|---------------|--------------------|--------------------|--------|
| 25→26 avril   | −0.5 °C           | −1.2 °C            | Gel faible |
| **26→27 avril** | **−3.1 °C**     | **−3.8 °C**        | **Gel significatif** |
| **27→28 avril** | **−4.8 °C**     | **−5.2 °C**        | **Gel majeur ★** |
| 28→29 avril   | −2.1 °C           | −2.6 °C            | Gel modéré |
| 29→30 avril   | −0.8 °C           | −1.1 °C            | Gel faible |

★ Nuit principale : dégâts sur 80 % des vignobles (Crozes-Hermitage, Côtes-du-Rhône).

## Structure des runs

```
runs/
├── april2021/
│   ├── config.yml                        # Master config : nuits critiques + chemins
│   └── pmap/
│       ├── era5_night_20210426.yml        # PMAP nuit 26→27 (ERA5)
│       ├── era5_night_20210427.yml        # PMAP nuit 27→28 (ERA5) ← PRINCIPALE
│       ├── era5_night_20210428.yml        # PMAP nuit 28→29 (ERA5)
│       ├── cerra_night_20210426.yml       # PMAP nuit 26→27 (CERRA)
│       ├── cerra_night_20210427.yml       # PMAP nuit 27→28 (CERRA)
│       └── cerra_night_20210428.yml       # PMAP nuit 28→29 (CERRA)
└── scripts/
    ├── detect_cold_nights.py             # Détection automatique depuis stat downscaling
    ├── orchestrate.py                    # Orchestrateur complet
    └── compare_downscaling.py            # Comparaison méthodes + scores SYNOP
```

## Workflow complet

### Étape 1 — Production opérationnelle (stat ou DL)

```bash
# Descente statistique sur tout avril 2021
uv run scripts/run_statistical_downscaling.py \
    --config  config/drome_ardeche.yml \
    --era5-sl data/era5/era5_sl_april2021.nc \
    --dem     data/dem/copdem_drome_100m.tif \
    --date    april2021 \
    --compute-indices \
    --out     output/stat/stat_downscaled_april2021.nc

# OU descente DL (si modèle entraîné)
uv run scripts/run_dl_inference.py \
    --config     config/drome_ardeche.yml \
    --checkpoint checkpoints/drome_ardeche/best_model.pt \
    --era5-sl    data/era5/era5_sl_april2021.nc \
    --dem-attrs  data/dem/dem_attributes.nc \
    --stats      checkpoints/drome_ardeche/normalization_stats.json \
    --out        output/dl/dl_downscaled_april2021.nc \
    --compute-indices
```

### Étape 2 — Détection des nuits critiques

```bash
uv run runs/scripts/detect_cold_nights.py \
    --stat-out   output/stat/stat_downscaled_april2021.nc \
    --threshold  -2.0 \
    --gdd-thresh 50.0 \
    --out        runs/april2021/cold_nights.json
```

### Étape 3 — Runs PMAP (vérification dynamique)

```bash
# Préparer les LBC pour chaque nuit critique
uv run runs/scripts/orchestrate.py \
    --config      runs/april2021/config.yml \
    --cold-nights runs/april2021/cold_nights.json \
    --forcing     era5           # 'era5' ou 'cerra'
    --step        prepare-lbc    # 'prepare-lbc', 'run-pmap', ou 'all'

# Lancer PMAP pour les nuits critiques
uv run runs/scripts/orchestrate.py \
    --config      runs/april2021/config.yml \
    --cold-nights runs/april2021/cold_nights.json \
    --forcing     era5 \
    --step        run-pmap
```

### Étape 4 — Comparaison et vérification

```bash
uv run runs/scripts/compare_downscaling.py \
    --stat-out    output/stat/stat_downscaled_april2021.nc \
    --pmap-dir    output/pmap/era5/ \
    --obs-synop   data/obs/synop_april2021.csv \
    --out-report  output/verification/report_april2021.html
```

## Stations SYNOP de référence (domaine Drôme-Ardèche)

| Station | WMO ID | Lat | Lon | Alt (m) | Variable clé |
|---------|--------|-----|-----|---------|-------------|
| Valence | 07481 | 44.916 | 4.967 | 156 | T2m, vent |
| Montélimar | 07486 | 44.558 | 4.739 | 73 | T2m (vallée) |
| Romans-sur-Isère | 07471 | 45.050 | 5.067 | 162 | T2m |
| Aubenas | 07558 | 44.622 | 4.387 | 303 | T2m, précip |
| Privas | 07552 | 44.735 | 4.600 | 317 | T2m |
| St-Martin-d'Ardèche | 07558 | 44.304 | 4.540 | 45 | T2m (basse vallée) |

## Métriques de vérification

- **Biais** (°C) : T2m à 04:00 UTC (heure la plus froide typique)
- **RMSE** sur T2m nocturne (18:00–06:00 UTC)
- **Skill score** (stat downscaling vs PMAP vs climatologie)
- **Hit rate** gel (-2 °C) : fraction de pixels avec Tmin < −2 °C bien capturée
- **False alarm ratio** gel : pixels faussement identifiés
