# Downscaling ERA5 / CERRA → 1 km pour indices d'assurance paramétrique

Descente d'échelle de réanalyses atmosphériques vers une résolution kilométrique,
conditionnée par le modèle numérique de terrain (MNT), pour produire des champs
météorologiques haute résolution et en dériver des **indices d'assurance
paramétrique** (gel, précipitations extrêmes, vent, canicule).

Trois familles de méthodes cohabitent derrière une **configuration unique** (Hydra)
et un **compute interchangeable** (Mac MPS local ↔ GPU cloud) :

1. **Statistique** — lapse-rate + quantile mapping (CPU, sans réseau de neurones).
2. **Deep learning** — U-Net FiLM conditionné par le MNT (Lightning).
3. **Foundation model** — Prithvi WxC (NASA/IBM), backbone gelé + adapter DEM.

---

## Sources de données

| Source | Résolution | Période | Variables clés |
|--------|-----------|---------|----------------|
| ERA5 | ~31 km / 1h | 1940–présent | T2m, Tp, U10, V10, TP |
| CERRA | ~5.5 km / 1h | 1985–2021 | T2m, Tp, U10, V10, TP |
| EU-DEM / COP-DEM | 25–30 m | — | Élévation |
| SAFRAN (validation) | 8 km / 1h | 1958–présent | T, Precip, Vent |

---

## Architecture

> 📐 **Sources, rôles et cascade MERRA-2 → CERRA → calibration capteurs** :
> voir [`docs/architecture.md`](docs/architecture.md) — pourquoi MERRA-2 et CERRA
> ne sont pas interchangeables (entrée vs cible vs calibration), et les deux
> chemins produits.

### Vue d'ensemble

```
.
├── downscaling/              # package installable (uv / setuptools)
│   ├── paths.py              # ancres absolues : CONFIG_DIR, MODELSTORE
│   ├── config.py             # load_config(overrides) — compose Hydra → dict
│   ├── shared/
│   │   ├── loaders.py        # chargement ERA5 / CERRA / DEM
│   │   └── indices.py        # indices paramétriques (cœur de valeur métier)
│   ├── statistical/
│   │   ├── lapse_rate.py     # correction T × altitude
│   │   ├── quantile_mapping.py  # QDM / EQM (biais climatologique)
│   │   └── pipeline.py       # pipeline statistique complet
│   ├── deep_learning/
│   │   ├── model.py          # U-Net conditionné MNT (FiLM)
│   │   ├── lightning_module.py  # LightningModule + DataModule
│   │   ├── train.py          # build_trainer + main (pl.Trainer)
│   │   └── inference.py      # inférence tuilée (fenêtre de Hann)
│   ├── prtihvi_wxc/          # Prithvi WxC (NASA/IBM) : backbone gelé + adapter DEM
│   │   ├── loader.py         # from_pretrained (modelstore/ puis HuggingFace)
│   │   ├── lightning_finetune.py  # fine-tune adapter (Lightning, sparse Netatmo)
│   │   └── inference.py      # FrostReanalysisRunner (rolling → Zarr)
│   └── scripts/              # entry points console (cf. table ci-dessous)
├── configs/                  # composition Hydra (groupes + experiment/)
├── modelstore/               # poids pré-entraînés (gitignored, cf. fetch-pretrained)
├── runs/                     # orchestration de campagnes multi-saisons
└── tests/                    # pytest — cœur métier + glue ML, sans GPU ni réseau
```

### Configuration : composition Hydra

Toute la paramétrisation passe par `configs/`, composée par
[Hydra](https://hydra.cc). Plus de YAML monolithique : on assemble des **groupes**
(`domain/`, `data/`, `statistical/`, `dl/`, `indices/`, `cluster/`) et une
**expérience** par région.

```
configs/
├── config.yaml              # racine — defaults: [experiment: drome_ardeche, cluster: local]
├── domain/   data/          # géométrie + chemins de fichiers, par région
├── statistical/ dl/ indices/   # hyperparamètres des méthodes
├── cluster/  {local,cloud}.yaml   # accelerator / precision / num_workers
└── experiment/<region>.yaml # `# @package _global_` : sélectionne domain+data+overrides
```

Ajouter une région = un petit `experiment/<region>.yaml` (+ `domain/` + `data/`),
**pas** une copie intégrale de la config. Les surcharges se passent en *dotlist* :

```bash
downscaling-run experiment=drome_ardeche cluster=cloud statistical.quantile_mapping.enabled=false
```

En code, `downscaling.config.load_config(overrides)` renvoie un dict « à plat »
(clés `domain`, `data`, `statistical`, `deep_learning`, `indices`, `cluster`,
`run`), de sorte que les pipelines restent agnostiques d'Hydra.

### Agnostique du compute (`cluster=`)

**La même config tourne en local (Mac) et sur GPU cloud** : seul le groupe
`cluster=` change. Le `pl.Trainer` (cf. `deep_learning/train.build_trainer`) lit
`accelerator`, `precision` et `num_workers` depuis ce groupe — rien d'autre à
toucher.

| `cluster=` | accelerator | precision | num_workers | Cible |
|-----------|-------------|-----------|-------------|-------|
| `local` (défaut) | `auto` | `32-true` | 0 | Mac Apple Silicon (MPS), CPU |
| `cloud` | `gpu` | `bf16-mixed` | 8 | RunPod A100 / L4 |

```bash
# Entraînement local (MPS), config par défaut
run-dl-train --data-dir data/training/ --epochs 5

# Même commande, compute cloud (GPU + bf16)
run-dl-train --data-dir data/training/ --epochs 150 --override cluster=cloud
```

Le pipeline statistique est CPU pur (numpy/scipy/sklearn) ; Prithvi WxC suit la
même logique `device="auto"` → cuda / mps / cpu.

### Entry points (console scripts)

Installés par `uv sync` / `pip install` (cf. `[project.scripts]`) :

| Commande | Rôle |
|----------|------|
| `downscaling-run` | Pipeline statistique piloté par Hydra |
| `run-dl-train` | Entraînement U-Net FiLM (Lightning) |
| `run-dl-inference` | Inférence U-Net sur nouvelles données |
| `run-campaign` | Campagne multi-saisons (stat + détection) |
| `fetch-pretrained` | Pré-télécharge les poids Prithvi WxC → `modelstore/` |
| `launch-dl-job` | Lance un pod GPU RunPod |
| `run-on-mac` | Inférence / QC locale Apple MPS |

---

## Approche 1 — Descente d'échelle statistique

### Température

1. **Correction lapse-rate** : `T_fin = T_grossier + γ × (z_fin – z_grossier)`
   - γ calibré mensuellement sur stations SYNOP (ou -6.5 K/km par défaut),
     avec un régime nocturne distinct (inversions de fond de vallée).
2. **Quantile Delta Mapping (QDM)** (Cannon et al. 2015) :
   - Correction des biais climatologiques ERA5→CERRA ou ERA5→SAFRAN,
     préservant le signal de changement climatique.

### Précipitations

- **BCSD** (Bias Correction & Spatial Disaggregation) : QDM sur quantiles
  mensuels puis désagrégation spatiale (analogues ou MNT).

### Usage rapide

```bash
# Config composée par Hydra (configs/) ; surcharges en dotlist.
# Liste de fichiers = saison concaténée temporellement.
downscaling-run \
    data.era5_sl=data/era5/era5_sl_20210427.nc \
    data.dem_raw=data/dem/copdem_drome_ardeche_100m.tif \
    run.out=output/stat_downscaled_20210427.nc \
    run.compute_indices=true
```

---

## Approche 2 — Descente d'échelle par deep learning (U-Net FiLM)

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
intermédiaires du U-Net via `γ·x + β` appris — le réseau capture la dépendance
altitude–température et les effets orographiques sur les précipitations.

L'entraînement est piloté par **Lightning** (`deep_learning/lightning_module.py`) :
loss composite (MSE + spectrale + gradient), scheduler warmup→cosine,
`ModelCheckpoint(monitor="val/rmse")`, early stopping, logger MLflow (ou CSV).

### Données d'entraînement recommandées

- **Input** : ERA5 (31 km) ou CERRA (5.5 km) rééchantillonné en basse résolution.
- **Target** : CERRA (5.5 km), SAFRAN (8 km) ou analyse AROME (1.3 km).
- **Conditionnement** : EU-DEM / COP-DEM à la résolution cible.

### Entraînement / inférence

```bash
run-dl-train \
    --data-dir data/training/ \
    --epochs 100 --batch-size 8 \
    --override dl.base_ch=64 cluster=cloud   # config via Hydra

run-dl-inference \
    --checkpoint checkpoints/best_model.pt \
    --era5-sl data/era5/era5_sl_20210427.nc \
    --dem-attrs data/dem/dem_attributes.nc \
    --out output/dl_downscaled_20210427.nc
```

---

## Approche 3 — Prithvi WxC (foundation model)

[Prithvi WxC](https://huggingface.co/Prithvi-WxC) (NASA/IBM, 2,3 B params) sert de
**backbone gelé** ; seul un **adapter CNN conditionné DEM** (~2 M params) est
fine-tuné, par supervision *sparse* aux stations Netatmo QC'd
(`prtihvi_wxc/lightning_finetune.py`, Lightning — adapter seul optimisé,
checkpoint réduit à l'adapter).

Les poids vivent dans `modelstore/` (gitignored), pré-téléchargés une fois pour
permettre l'inférence **hors-ligne** (CI, RunPod sans réseau) :

```bash
fetch-pretrained --list            # plan, sans réseau
fetch-pretrained                   # backbone + adapter Granite → modelstore/
DOWNSCALING_MODELSTORE=/workspace/models fetch-pretrained   # volume réseau
```

Le loader cherche `modelstore/` d'abord, puis retombe sur HuggingFace.

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

## Bonnes pratiques & développement

### Installation (uv recommandé)

```bash
uv sync                      # pipeline statistique (CPU) seul
uv sync --extra dl           # + deep learning (torch, lightning)
uv sync --extra prithvi      # + Prithvi WxC (huggingface_hub, safetensors)
uv sync --extra all          # tout (stat + dl + prithvi + pmap)
```

L'environnement est **verrouillé** (`uv.lock`) : `uv sync` est reproductible.
Les `[project.optional-dependencies]` isolent les stacks lourdes (torch n'est
requis que pour le DL).

### Tests

Suite `pytest` **sans GPU ni réseau** — le cœur métier (indices, transfos
statistiques) est couvert en dur, la glue ML (Lightning, modelstore) via des
modèles jouets et des `fast_dev_run`. Les tests DL/Prithvi sont gardés par
`pytest.importorskip(...)` : ils sont **sautés** quand l'extra correspondant n'est
pas installé (la CI ne tourne que l'extra `statistical`).

```bash
uv run pytest                # toute la suite
```

### Lint & typage

- **Ruff** (`E`, `F`, `W`, `I`, `UP`, `B`) — garde **bloquante** en CI :

  ```bash
  uv run ruff check downscaling tests
  uv run ruff check --fix downscaling tests   # corrections sûres
  ```

- **mypy** — typage **progressif** et informatif (non bloquant) : on ne vérifie
  pour l'instant que les fonctions annotées (`check_untyped_defs = false`),
  à durcir module par module.

  ```bash
  uv run mypy downscaling
  ```

### Conventions

- **Pas de chemin ni d'hyperparamètre en dur** : tout passe par `configs/` et se
  surcharge en dotlist. Une nouvelle région = un `experiment/`.
- **Compute interchangeable** : ne jamais coder `accelerator`/`precision` en dur —
  les lire depuis le groupe `cluster=`.
- **Imports** : chemins absolus `downscaling.…` pour le code applicatif, relatifs
  (`.module`) au sein d'un sous-package.
- **Modèle injecté** : le réseau est construit en amont (`build_model` /
  `from_pretrained`) puis passé au `LightningModule` ; l'optimiseur est instancié
  dans `configure_optimizers`.
- **Reproductibilité** : commits atomiques, `uv.lock` à jour, tests verts avant
  push.

---

## Références

- Cannon A.J. et al. (2015) *Bias Correction of GCM Precipitation by Quantile
  Delta Mapping*. J. Climate 28, 6938–6959.
- Perez E. et al. (2018) *FiLM: Visual Reasoning with a General Conditioning
  Layer*. AAAI.
- Ronneberger O. et al. (2015) *U-Net*. MICCAI.
- Baño-Medina J. et al. (2020) *Configuration and intercomparison of deep learning
  neural models for statistical downscaling*. Geosci. Model Dev. 13, 2109–2124.
- Höhlein K. et al. (2020) *A comparative study of convolutional neural network
  models for wind field downscaling*. Met. Apps 27, e1961.
- Schmude J. et al. (2024) *Prithvi WxC: Foundation Model for Weather and Climate*.
  arXiv:2409.13598.
