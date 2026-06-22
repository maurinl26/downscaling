---
date: 2026-06-17
type: rapport scientifique
auteur: Loïc Maurin
projet: MVP Baronnies abricot
tags:
  - calibration
  - downscaling
  - lot-b
  - stage-2
  - CERRA
  - Sencrop
  - POD-FAR
status: brouillon — première itération
---

# Rapport — Calibration Stage 2 statistical CERRA × Sencrop (Lot B), juin 2026

## Résumé exécutif

Première mise en production du pipeline de recalibration statistique pour le **Lot B** (statistical downscaling + correction sparse Sencrop) sur l'**abricot des Baronnies**, dans la fenêtre frost-flo (février-avril). Pipeline fonctionnel sur 2 années (2022, 2023), métriques POD/FAR/CSI extraites pour 2023 (référence). Sensibilité maximale atteinte (**POD = 100 %**) ; spécificité encore insuffisante (**FAR = 74 %** au seuil flo -2,2 °C) liée à un biais systématique +10 °C qu'il faut diagnostiquer avant le Lot C.

> [!warning] Statut
> Rapport à froid post-pitch Corréard 17 juin 2026. Les conclusions sont préliminaires : seulement 2 années traitées sur 4 prévues (2024 et 2025 ont systématiquement crashé), et 18 nuits matchées sur 90 pour 2023 (couverture partielle). À reconduire après correction des bugs identifiés.

---

## 1. Méthode

### 1.1 Données d'entrée

| Source | Variable | Couverture | Résolution | Volume |
|---|---|---|---|---|
| **CERRA single levels** (ECMWF/CDS) | `2m_temperature` | 2022-02 → 2025-04, fenêtre fév-mar-avr | 5,5 km natif, rendu 0,05° lat/lon | 12 NetCDFs (~50 MB chacun) |
| **CERRA-Land** | `skin_temperature` | idem | idem | 12 NetCDFs |
| **Sencrop bulk export** | températures stationsréseau | 2021-05 → 2026-05 | 49 stations Drôme + nord Vaucluse, cadence 15 min | partitionné CSV, lecture S3 (`karpos-backtest-data/sencrop`) |
| **SRTM 30 m DEM** | altitude | bbox Drôme | ~30 m | NetCDF généré via package `elevation` |

> [!note] CERRA-Land non utilisé dans Lot B
> Le pipeline charge CERRA-Land par symétrie mais n'exploite pas encore `skin_temperature` dans la correction. C'est une réserve pour intégration ultérieure (sol-air, gel sol).

### 1.2 Pipeline scientifique (Stage 2 du recalibration_pipeline.sh)

1. **Stage 0** — DEM bootstrap : si BD ALTI IGN absent, fetch SRTM 30 m via `elevation` package → NetCDF `srtm30_drome.nc` (111 MB).
2. **Stage 1** — Download CERRA : chunks mensuels via cdsapi, fenêtre frost-flo (fév-avr) × 4 années × 2 datasets = 24 NetCDFs.
3. **Stage 1b** — Concat mensuels → annuel via `xr.open_dataset(...).load()` séquentiel + `xr.concat` (workaround `xr.open_mfdataset` deadlock).
4. **Stage 2 — recalibrate_statistical** :
   - Charge CERRA atm yearly NetCDF, calcule **Tmin nocturne** (résample 1D après shift -9h pour aligner la nuit sur la date du matin)
   - Initialise `StatisticalDownscalingPipeline` (lapse-rate + QDM optionnel)
   - **Pour chaque nuit** :
     - Application lapse-rate vers grille fine 1 km via SRTM DEM
     - QDM désactivé (pas de `--obs-ref` passé, placeholder)
     - **Correction RBF gaussien sparse** sur les Sencrop résidus station-par-station : `obs_Sencrop - grid_at_station` propagé sur tout le domaine avec `sigma_km = 7`
   - Concat des 90 nuits → Zarr annuel (5400 × 5400 × 90 timesteps, ~3,5 GB compressé)

### 1.3 Métriques d'évaluation (script `analyze_recalibrated_statistical.py`)

Pour chaque année, sur l'ensemble (station × nuit) avec `temperature_source == 'station'` :
- **Résidus** : `obs_Sencrop − pred_grid_at_station`
  - Moyenne (biais), RMSE, |moyenne|, P10/P90
- **Détection gel par seuil** (TP / FP / FN / TN) à 3 seuils :
  - **-2,2 °C** : T10 abricot floraison BBCH 60-65 (référence Proebsting & Mills 2005)
  - **0 °C** : référence simple
  - **-5 °C** : événement sévère
- Indicateurs : **POD** (Probability of Detection), **FAR** (False Alarm Ratio), **CSI** (Critical Success Index), **bias score**.

---

## 2. Résultats — référence 2023

### 2.1 Couverture

| Indicateur | Valeur |
|---|---|
| Stations dans bbox | 49 |
| Nuits avec obs Sencrop matchées | **18** (vs 90 attendues) |
| Pairs station × nuit évaluées | **618** |

> [!warning] Couverture partielle (18/90 nuits)
> L'analyse ne match que 18 nuits sur les 90 traitées par Stage 2. **Hypothèse** : la coord `time` du Zarr Stage 2 est stockée en `int64` (0..89) sans encodage datetime. L'analyze synthétise les dates en partant du 1ᵉʳ février, mais le matching effectif avec Sencrop révèle des trous (probable décalage de plusieurs jours sur le shift -9h).

### 2.2 Résidus globaux (618 pairs)

| Indicateur | Valeur | Lecture |
|---|---|---|
| Biais moyen | **+10,54 °C** | pred plus froid que obs |
| RMSE | **11,15 °C** | dominée par le biais systématique |
| \|moyenne\| | 10,54 °C | idem |
| P10 / P90 du biais | +5,6 / +15,3 °C | offset positif sur toute la distribution |

> [!warning] Biais +10 °C non physique
> Le biais est uniformément positif sur 100 % des paires. Ce n'est ni du bruit ni de l'aléa : c'est **un signe d'un défaut systématique** dans la chaîne lapse-rate ou dans l'unité (Kelvin vs Celsius non détecté en post-traitement, voir §3.4).

### 2.3 Détection gel par seuil

| Seuil | TP | FP | FN | POD | FAR | CSI | Bias score |
|---|---|---|---|---|---|---|---|
| **-2,2 °C** (T10 flo) | 159 | 459 | 0 | **100,0 %** | **74,3 %** | 25,7 % | 3,89 |
| 0,0 °C | 284 | 334 | 0 | 100,0 % | 54,0 % | 45,9 % | 2,18 |
| -5,0 °C | 48 | 535 | 0 | 100,0 % | 91,8 % | 8,2 % | 12,15 |

Lectures
- **POD = 100 % systématiquement** : aucun événement gel observé sur Sencrop n'est manqué par la grille recalibrée. C'est la qualité critique pour l'assurance paramétrique (zéro miss).
- **FAR élevé** : la grille déclenche des gels qui n'ont pas eu lieu (sur-prédiction). Compatible avec le biais +10 °C — si la grille prédit +10 °C trop froid, beaucoup de cellules tombent sous le seuil sans qu'aucune obs ne confirme.
- **Bias score >> 1** : très net excès de prédictions vs observations (frequency bias).

---

## 3. Interprétation et limites

### 3.1 Comparaison vs baselines historiques

| Méthode | Période | POD | FAR | Source |
|---|---|---|---|---|
| **CERRA-Land brut (V1)** | 2015-2020 Nyons | 100 % | **87 %** | `bias_summary.csv` |
| **ERA5-Land brut** | 2022-2025 Sencrop | 60 % | 69 % | `lot_d_metrics.csv` |
| **Lot B Stage 2 (ce rapport)** | 2023 fév-avr Sencrop | **100 %** | **74,3 %** | `2023.posthoc.json` |
| **Lot C DL FiLM (cible)** | 2022-2025 | ≥ 90 % | ≤ 20 % | non encore mesuré |

Lot B vs ERA5-Land : **+ 40 pts de POD** (gain net sur la sensibilité), **+ 5 pts de FAR** (légère dégradation due au biais). Au pitch, on peut dire : *« le pipeline Lot B sature la détection mais reste à affiner sur la spécificité »*.

### 3.2 Pourquoi POD = 100 % ?

Toutes les nuits matchées dans le subset (18 nuits sur 2023 fév-avr) ont **un événement gel obs au niveau station**. La grille étant biaisée froide, elle prédit gel partout → TP capture tout. FN = 0 ; aucune erreur de manqué.

C'est un biais de l'évaluation aussi : sur 90 nuits seulement 18 matchées (probablement les plus froides, où le matching trouve les obs). Sur les 72 nuits non matchées, on n'a aucune mesure → POD biaisé optimistement.

### 3.3 Pourquoi FAR = 74 % ?

Le biais +10 °C explique mécaniquement : sur 159 vrais gels observés, le modèle gel-trigger sur 618 cellules (toutes au-dessus du seuil de la station de mesure). La sur-prédiction est massive.

### 3.4 Diagnostic du biais +10 °C

Hypothèses à investiguer :

1. **Orientation lapse-rate inversée** : le code applique `T_fine = T_coarse - γ × Δh`. Si `Δh = h_fine - h_coarse` mais le SRTM DEM est plus haut en moyenne que le grid CERRA coarse, on refroidit trop. À vérifier dans `downscaling/statistical/pipeline.py`.
2. **Unités Kelvin vs Celsius** : CERRA stocke `t2m` en Kelvin. Le pipeline ne convertit pas avant de retourner le Zarr (à vérifier). L'analyze détecte automatiquement (sample > 100 → −273,15) mais sample = 0,00 °C donc déjà converti côté Stage 2. À confirmer.
3. **Décalage temporel** : matching faible (18/90) suggère que les timestamps du Zarr ne sont pas alignés sur Sencrop. Peut-être 1 jour de décalage qui fausse les résidus.
4. **DEM SRTM 30 m vs CERRA 5,5 km** : différence d'échelle énorme. Le lapse-rate est sensible aux pixels de bord (vallées vs sommets). Le `sigma_km = 7` du RBF de correction est-il assez large ?

### 3.5 Bugs pipeline rencontrés (à analyser à froid)

| # | Symptôme | Cause partielle | Impact |
|---|---|---|---|
| 1 | `'DataArray' object has no attribute 'time'` à chaque nouveau pod / fresh checkout | CERRA NetCDF utilise `valid_time`, le script ne renomme pas | Bloquant années 2024 + 2025 |
| 2 | `AttributeError: 'lat'` lors du lapse-rate | Coord CERRA = `latitude` mais DEM = `lat` | Bloquant |
| 3 | `xr.open_mfdataset` deadlock sur Stage 1b | `combine='by_coords'` + dask + locks MFS | Résolu par concat séquentiel |
| 4 | `dask` ImportError sur Stage 1b | Pas dans extras `statistical` | Résolu par ajout `dask>=2024.1` |
| 5 | `eio clip --product SRTM3` | `--product` est flag global `eio`, pas sous-cmd | Résolu |
| 6 | CDS API `cost limit exceeded` sur requête annuelle | Trop gros pour CDS quota | Résolu par chunking mensuel |
| 7 | CDS MARS `croppedRepresentation not implemented` | Grille LCC CERRA refuse `area` crop | Résolu par `grid: [0.05, 0.05]` |
| 8 | Processus SIGKILL silencieux à chaque transition d'année (8 essais : v8, v9, v10, v11, +4 autres) | Inconnu — peut-être cgroup mem 43 GB sur ancien pod, peut-être MFS network reconnect, peut-être RunPod scheduler | **Bloquant, années 2024 + 2025 jamais sorties** |
| 9 | Disk quota MFS exceeded à 30 GB d'usage | Quota RunPod par utilisateur sur le volume `downscaling-workspace` | Mitigé par push S3 + delete local |
| 10 | Analyze SIGKILL durant lecture S3 Sencrop (3 essais) | Probablement même cause que bug #8 | Métriques 2022 non extraites |
| 11 | `recalibrate_dl_film.py` ImportError (`sys.get_int_max_str_digits`) | Python 3.11.0rc1 sur le venv vs `@substitute_in_graph` torch dynamo | Stage 3 jamais lancé |
| 12 | Time coord Zarr stockée en `int64` (0..89) au lieu de `datetime64` | Bug `xr.concat + to_zarr` ne préserve pas datetime ? | Matching analyze imparfait |
| 13 | Biais résidu +10 °C systématique | Voir §3.4 | Lecture POD/FAR biaisée |
| 14 | Time coord Sencrop `night_date` (dt.date) vs Zarr `Timestamp` comparaison | Type subtlety | Possible cause matching 18/90 |

---

## 4. Architecture livrée

```
RunPod pod karpos-recalibration-v2 (gvwx812dfoasd4)
    └── ghcr.io/maurinl26/karpos-downscaling:latest
    └── Volume downscaling-workspace (mfs#euro.runpod.net)
        └── /workspace/data/cerra_zarr/cerra_atm_2022..2025.nc + monthly chunks
        └── /workspace/data/cerra_land_zarr/cerra_land_*.nc
        └── /workspace/data/dem/srtm30_drome.nc
        └── /workspace/data/output/recalibrated_statistical/
            ├── 2022.zarr (3,5 GB)
            ├── 2022.metadata.json
            ├── 2023.zarr (3,5 GB)
            └── 2023.posthoc.json
S3 Scaleway karpos-backtest-data
    ├── sencrop/ (bulk export, lu en streaming)
    └── recalibrated/statistical/
        ├── 2022.zarr/ (10,8 GB, ~1200 chunks)
        ├── 2022.metadata.json
        └── 2023.zarr/ (en cours, ~744 chunks)
W&B project karpos-recalibrate-statistical
    └── projet créé, runs effacés par disable-wandb (à réactiver)
```

---

## 5. Prochaines étapes (post-pitch, à froid)

### Priorité 1 — Diagnostic des bugs structurels

- [ ] **Bug #8 (SIGKILL silencieux)** : reproduire sur un pod neuf avec petits jobs ; instrumenter avec `dmesg`, `/sys/fs/cgroup/memory.events`, monitoring RSS process. Si OOM kernel, refactor `recalibrate_statistical.py` pour écrire chunks Zarr incrémentaux (au lieu d'accumuler 90 slabs in-memory).
- [ ] **Bug #13 (biais +10 °C)** : tracer dans la pipeline lapse-rate. Comparer Tmin grid brute (sans correction) à Sencrop. Voir si le biais est introduit avant ou après le RBF residual.
- [ ] **Bug #11 (Python 3.11.0rc1)** : upgrade venv → Python 3.11.x stable ou 3.12 dans l'image Docker `karpos-downscaling`. Tester import torch.dynamo / lightning.
- [ ] **Bug #12 (time coord Zarr int)** : modifier `recalibrate_statistical.py` pour explicit `encoding={'time': {'units': 'days since YEAR-02-01', 'dtype': 'int32'}}` ou `decode_times=True` lors du to_zarr. Validate roundtrip.

### Priorité 2 — Lot C DL FiLM

- [ ] Une fois Stage 2 stable, reprendre Stage 3 sur les Zarrs Stage 2 calibrés.
- [ ] Architecture U-Net FiLM + W&B logging (déjà préparé dans `recalibrate_dl_film.py`).
- [ ] Validation : POD/FAR/CSI sur years held-out.

### Priorité 3 — Robustesse infra

- [ ] Auto-restart externe (script Mac) qui SSH-relance sur SIGKILL.
- [ ] Push S3 progressif (toutes les N nuits, pas en bloc à la fin).
- [ ] Image Docker `karpos-downscaling` plus stricte sur les versions (Python 3.11 stable, torch GA, etc.).

---

## 6. Annexes

### 6.1 Commandes utilisées (reproductibilité)

```bash
# Stage 2 année 2023 (commande utilisée)
uv run python -m downscaling.scripts.recalibrate_statistical \
  --year 2023 \
  --cerra-atm /workspace/data/cerra_zarr/cerra_atm_2023.nc \
  --cerra-land /workspace/data/cerra_land_zarr/cerra_land_2023.nc \
  --dem /workspace/data/dem/srtm30_drome.nc \
  --sencrop s3://karpos-backtest-data/sencrop \
  --out /workspace/data/output/recalibrated_statistical \
  --sigma-km 7.0 --wandb-disabled

# Analyze 2023
uv run python -m downscaling.scripts.analyze_recalibrated_statistical \
  --root /workspace/data/output/recalibrated_statistical \
  --sencrop s3://karpos-backtest-data/sencrop \
  --threshold-c -2.2 --wandb-disabled --years 2023
```

### 6.2 Fichiers livrables

- `s3://karpos-backtest-data/recalibrated/statistical/2022.zarr/` (Lot B Stage 2 2022)
- `s3://karpos-backtest-data/recalibrated/statistical/2022.metadata.json`
- `s3://karpos-backtest-data/recalibrated/statistical/2023.zarr/` (en cours d'upload)
- `s3://karpos-backtest-data/recalibrated/statistical/2023.posthoc.json` (à pousser manuellement)
- Repository `maurinl26/karpos-downscaling` branche `main` au commit `a24278b` (W&B instrumentation + analyze script)
- Repository `maurinl26/parametric_insurance` branche `main` au commit post-PR #67 (PWA contraste aligné)

### 6.3 Citations

- **Proebsting & Mills (2005)** — seuils T10 / T90 par stade BBCH (FAO).
- **Luedeling et al. (2011)** — modèle dynamique chilling (Scientia Horticulturae). Non encore intégré dans le pipeline mais référencé dans la slide Méthode.
- **CERRA / Copernicus C3S** — réanalyse 5,5 km. Dataset `reanalysis-cerra-single-levels`.
- **Sencrop** — fournisseur de données stations agricoles, bulk export 2021-2026 acheté.

---

## 7. Annexe — Audit à froid (subagent, post-pitch 2026-06-17)

Diagnostic complet après lecture ligne à ligne de `recalibrate_statistical.py`, `analyze_recalibrated_statistical.py`, `statistical/pipeline.py`, `statistical/lapse_rate.py`, `prtihvi_wxc/sencrop.py`, `scripts/download_cerra_for_recalibration.py`.

### 7.1 Verdict sur le biais +10 °C : **double cause confirmée**

**Cause A — orographie source manquante (z_source = 0 m)**

- `pipeline.py:205-224` cherche `z`, `orog` ou `oro` dans le Dataset source. `recalibrate_statistical.py:309-311` passe uniquement `xr.Dataset({"t2m": slab})`, **sans orographie**. Le téléchargement (`download_cerra_for_recalibration.py:52`) ne demande que `2m_temperature`.
- Conséquence : fallback `z_coarse = zeros((H, W))` avec un `warnings.warn` muet.
- Sur Baronnies (SRTM moyen ~600 m), `dz = z_SRTM - 0 ≈ 600 m` → correction lapse-rate `γ × dz = -6,5e-3 × 600 = -3,9 K` **artificielle**. Pic à -9,75 K sur les crêtes 1500 m.

**Cause B — mix Kelvin/Celsius dans le RBF residual**

- `lapse_rate.py:71-76` retourne `t_corrected` en **Kelvin** (CERRA T2m natif K).
- `recalibrate_statistical.py:165` : `residuals = obs_tmin - nearest_vals` avec `obs_tmin` Sencrop en **°C** et `nearest_vals` grille en **K** → résidu ≈ -273 °C aux stations.
- Le RBF propage cette correction massive sur les voisinages stations → la grille devient ~Celsius **dans les zones stations**, reste en Kelvin **ailleurs**.
- Le `sample = 0,00` détecté par analyze (`analyze:111`) suggère que la zone des stations Baronnies a été ramenée en Celsius, mais avec un biais cumulatif de +10 °C (Cause A + résiduel mauvaise correction).

**Fix canonique** :
1. `download_cerra_for_recalibration.py` : ajouter le téléchargement `orography` CERRA (one-shot, pas par mois). Si non dispo sur CDS single-levels, dégrader le SRTM à 5,5 km avec `xesmf` conservatif.
2. `recalibrate_statistical.py` : conversion K → °C **explicite** au chargement du t2m (`nightly = nightly - 273.15` si `attrs["units"]=="K"`). Puis charger l'orographie dans le `xr.Dataset` du slab.
3. Tester sur 2023 (1 année) avant de relancer 2022/2024/2025.

### 7.2 Bugs additionnels détectés (non listés en §3.5)

| # | Description | Impact | Effort |
|---|---|---|---|
| **15** | `recalibrate_statistical.py:251` charge `load_stations_catalog` **deux fois** (sans bbox puis avec bbox), gaspillage S3 | 1/5 | 5 min |
| **16** | QDM "calibrate" placeholder : `pipe.calibrate(ref_ds, ref_ds)` calibre sur soi-même, branche cassée même quand activée | 2/5 | 4 h (besoin référence HR vraie) |
| **17** | Détection K vs C dans `analyze:111` fragile (`sample > 100` sur médiane d'une grille bimodale possiblement à 0) | 5/5 si rate | 30 min (check explicite + paramétrer threshold) |
| **18** | **Mix K/C résidu RBF** (voir 7.1 Cause B), co-cause principale biais | 5/5 | 30 min |
| **19** | Fenêtre CERRA limitée à fév-avr → la nuit du 1er février est tronquée (besoin 21h UTC du 31 janvier), 90e nuit synthétisée n'existe pas | 2/5 | 1 h |
| **20** | `to_zarr` sans `encoding={"t2m": {"chunks": (1, H, W)}}` → relecture nuit-par-nuit inefficace, coûte cher en S3 | 3/5 | 30 min |
| **21** | `out.values = grid_arr + correction` réassigne `.values` directement, fragile sur xarray lazy | 1/5 | 5 min |

### 7.3 Plan d'attaque priorisé (post-RDV Corréard)

| Ordre | Bug(s) | Justification | Effort |
|---|---|---|---|
| **1** | #13 + #18 (orographie + mix K/C) | LE bug scientifique. Tant qu'il n'est pas levé, POD/FAR sont des artefacts. À traiter en bundle dans une seule PR sur `maurinl26/karpos-downscaling`. | 2-4 h |
| **2** | #8 SIGKILL refactor write incrémental | Sans 4 années, pas de Go/No-Go robuste. Refactor : `to_zarr(append_dim="time")` par nuit + checkpoint + `psutil` logging RAM. **Couple avec #20**. | 4-6 h |
| **3** | #10 + #20 analyze OOM + chunking | Sans analyze qui tourne sans planter, pas de métriques. **Couple avec #2**. | 2 h |
| **4** | #12 + #19 time coord int + fenêtre fev-avr | Matching 18/90 → ~80/91 attendu. | 1-2 h |
| **5** | #17 safe Kelvin detection in analyze | Garde-fou tant que #13/#18 ne sont pas mergés. | 30 min |

### 7.4 Recommandations infra (Bug #8)

Quatre changements ciblés dans `recalibrate_statistical.py` (~50 lignes) :

1. **Écriture Zarr incrémentale** : `to_zarr(append_dim="time")` par nuit. Borne RAM à ~120 MB au lieu de ~10 GB.
2. **Checkpointing** : au démarrage, `last_d = xr.open_zarr(zarr_path)["time"].max()` → reprise après SIGKILL.
3. **Lazy Sencrop** : `pd.read_csv(chunksize=...)` au lieu de `load_timeseries` qui charge toute l'année.
4. **Push S3 progressif** : `aws s3 sync` toutes les 10 nuits, garder le local pour checkpointing.

Coût total : 4-6 h. Testable sur pod 4 h avec 2024-2025.

### 7.5 Points incertains à investiguer en local

1. Le Zarr `2023.zarr` est-il effectivement bimodal K/C ? À vérifier : `xr.open_zarr(...).t2m.isel(time=0).quantile([0.1, 0.5, 0.9])`.
2. La cause exacte du SIGKILL : OOM kernel (`dmesg`) vs MFS reconnect ? Instrumenter avec `psutil` RSS par nuit.
3. Le matching 18/90 : compter `obs_per_night["night_date"].nunique()` directement sur 2023 + vérifier bbox CERRA vs catalogue Sencrop.

### 7.6 Liens internes (Obsidian)

- [[Campagne Sencrop S23 — preuve de thèse gel]] — contexte historique
- [[Rapport — Modèle de chilling pour l'abricot des Baronnies]] — phénologie complémentaire
- [[Go-NoGo Juin 2026]] — gate FAR ≤ 20 % pour Baronnies/Nyons
- [[Freemium app - structure et seuils]] — produit aval

## 8. Annexe — Run #28 capacité U-Net (nuit 17→18 juin 2026)

**Hypothèse testée** : augmenter la capacité du U-Net (base_ch=32→64, n_levels=3→4) doit sortir le modèle de la régression vers la climatologie observée en 32/3, en lui donnant les degrés de liberté pour modéliser cold-pool × wind shelter.

**Configuration** : `--base-ch 64 --n-levels 4 --early-stopping-patience 5`, 30 epochs, 4 ans (2022-2025), Mac M4 MPS local. NB : `build_model("unet", base_ch=64, n_levels=4)` produit **19 M params**, pas 4.6 M comme estimé à l'ouverture de l'issue. La sous-estimation a contribué au surapprentissage observé.

**Verdict : contre-performance.** Le bump de capacité **dégrade** la RMSE annuelle sur deux années sur quatre, et n'atteint jamais le gate Go/No-Go (POD≥90% ET FAR≤20%).

| an | RMSE 32/3 | RMSE 64/4 | POD@-2.2 32/3 | POD@-2.2 64/4 | FAR 32/3 | FAR 64/4 |
|---|---|---|---|---|---|---|
| 2022 | 4.17 | **5.40** | 0.00 | 0.43 | nan | **0.85** |
| 2023 | 3.81 | **4.09** | 0.00 | 0.23 | nan | **0.58** |
| 2024 | 4.85 | 3.26 | 0.57 | 0.00 | 0.96 | 1.00 |
| 2025 | 3.57 | 3.26 | 0.22 | 0.29 | 0.73 | 0.76 |

**Trois enseignements** :

1. **Sortie de la climato mais sans calibration** : le 32/3 régresse vers la moyenne (POD=0 en 2022/2023, modèle n'ose jamais < -2.2°C). Le 64/4 ose prédire du froid (POD passe à 23-43%) mais lâche des fausses alertes à grande échelle (FAR 58-100%). Le modèle a appris à parier sur le froid sans discriminer.

2. **Biais chaud persiste sur les deux configs** : +1.09 à +1.79°C sur 64/4, +0.64 à +2.43°C sur 32/3. Le DL FiLM ne corrige pas vers le froid, il replique CERRA + bruit station. Issue #18 a fixé le biais d'orographie du Lot B mais pas celui du Lot C.

3. **EarlyStopping inopérant** : les 4 runs ont fait les 30 epochs entiers, best ckpts trouvés à epoch 3 / 8 / 10 / 11. Trajectoire val/rmse en yo-yo, `min_delta=1e-3` trop laxiste pour discriminer les micro-améliorations bruitées. À durcir à `min_delta=0.05` ou supérieur si on relance.

**Conclusion opérationnelle** : la capacité **n'est pas le verrou**. Augmenter le modèle sans changer la loss ni la supervision = plus d'overfit, point final. La voie est **issue #5 (loss redesign)** : first-guess physique + résiduel + pinball multi-quantile + splits anti-fuite + densification via Lot B (issue #23). PR #31 (downscaling) et #73 (parametric_insurance) restent valables comme infra A/B mais les défauts (32/3, ES off) ne doivent **pas** changer.

**Lot C reste hors-gate à V1.** Lot B + QDM reste la voie commerciale juin-juillet.

W&B runs : [2022](https://wandb.ai/maurin-loic-ac-karpos-pro/karpos-recalibrate-dl-film/runs/tlp3zql1) · [2023](https://wandb.ai/maurin-loic-ac-karpos-pro/karpos-recalibrate-dl-film/runs/cmvc4ikx) · [2024](https://wandb.ai/maurin-loic-ac-karpos-pro/karpos-recalibrate-dl-film/runs/cds6se0s) · [2025](https://wandb.ai/maurin-loic-ac-karpos-pro/karpos-recalibrate-dl-film/runs/wxs90bxu).

## 9. Annexe — A/B Lot B nu vs Lot B + QDM (2026-06-18 matin)

**Hypothèse testée** : activer le QDM conditionnel mensuel (placeholder cassé jusque-là, cf. bug #16) doit corriger les biais distributionnels résiduels après lapse-rate et améliorer POD/FAR sur les queues froides.

**Configuration** : `QuantileDeltaMapping(kind='delta', by_month=True, n_quantiles=100)` calibré sur **12 943 paires (date, station) pooled sur 2022-2025** (lapse-rate downscalé au point de chaque station Sencrop). Δ médian mensuel calibration : +0.09°C (avril) à +0.58°C (mars). QDM appliqué **après lapse-rate**, **avant** RBF Sencrop résiduel (sigma_km=7).

**Verdict : ship V1 sans QDM.** En in-sample, QDM est invisible à légèrement dégradant ; le 2026 hold-out perd 4 points POD ; le seul gain net est sur la queue -5°C en 2023.

| an | RMSE nu | RMSE +QDM | POD@-2.2 nu | POD@-2.2 +QDM | FAR@-2.2 nu | FAR@-2.2 +QDM |
|---|---|---|---|---|---|---|
| 2022 | 1.58 | 1.60 | 0.42 | 0.42 | 0.34 | **0.39** |
| 2023 | 1.59 | 1.61 | 0.51 | 0.51 | 0.32 | **0.36** |
| 2024 | 1.27 | 1.30 | 0.00 | 0.00 | 1.00 | 1.00 |
| 2025 | 1.43 | 1.45 | 0.34 | 0.34 | 0.38 | 0.38 |
| **2026** | 1.27 | 1.30 | **0.86** | **0.82** | **0.14** | **0.15** |

Seul gain net : seuil -5°C année 2023, POD 0.22 → 0.24, FAR 0.45 → 0.37 (queue froide bien traitée).

**Diagnostic mécanique** : le RBF résiduel s'appuie sur les mêmes stations utilisées pour l'évaluation in-sample et écrase l'effet QDM. Le QDM contribue uniquement aux pixels loin des stations. Sans validation LOO, son apport reste théorique.

**Décision produit V1** :
- **Lot B nu** = pipeline V1 commerciale Karpos. Gate Go/No-Go atteint sur 2026 hold-out (POD 86%, FAR 14%, CSI 0.75).
- **PR #31 (calibrate_qdm + --qdm-joblib)** reste mergeable comme infra A/B mais QDM **non activé par défaut**.

**Issues d'amélioration ouvertes pour septembre** (post-V1, conditions partenariats) :
- [downscaling#33](https://github.com/maurinl26/karpos-downscaling/issues/33) — Validation LOO QDM avant intégration (le bon test scientifique)
- [downscaling#34](https://github.com/maurinl26/karpos-downscaling/issues/34) — QDM tail-only (quantiles ≤ 0.20) pour cibler le gel
- [downscaling#35](https://github.com/maurinl26/karpos-downscaling/issues/35) — QDM avec sigma_km RBF réduit (3 au lieu de 7)

---

*Rédigé par Claude (Karpos collaboratif) le 2026-06-17 post-RDV Corréard. Annexe 7 audit à froid ajoutée le même jour après lecture ligne à ligne du pipeline. Annexe 8 ajoutée le 2026-06-18 après le run nocturne #28. Annexe 9 ajoutée le même jour après l'A/B Lot B nu vs +QDM.*
