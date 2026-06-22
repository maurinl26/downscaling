# Architecture de la descente d'échelle — sources, rôles, cascade

> Réponse aux questions : *« MERRA-2 peut-il être remplacé par CERRA ? »* et
> *« Quel est le pipeline MERRA → CERRA → calibration capteurs locaux (Sencrop) ? »*

## TL;DR

- **Non, CERRA ne remplace pas MERRA-2 *en entrée* de Prithvi WxC.** Le backbone
  a été **pré-entraîné sur MERRA-2** : son jeu de variables (160 canaux), ses
  niveaux verticaux, sa grille globale et ses **scalers de normalisation** sont
  figés dans les poids. Lui donner du CERRA = hors-distribution, variables et
  grille incompatibles.
- **MERRA-2, CERRA et Sencrop ne sont pas interchangeables : ils jouent des
  rôles différents.** MERRA-2 = *entrée* du foundation model. CERRA = *cible*
  haute résolution pour entraîner la tête de descente d'échelle. Sencrop/Netatmo
  = *référence terrain* pour la calibration locale finale.
- Pour un produit **centré CERRA** (le socle Karpos), le chemin pragmatique
  n'a pas besoin de Prithvi : `CERRA → 1 km (U-Net/statistique) → calibration
  capteurs`. Prithvi/MERRA-2 est une **couche optionnelle** de contexte
  grande-échelle (cf. §6).

---

## 1. Rôle de chaque source

| Source | Résolution | Couverture | Rôle dans le pipeline |
|--------|-----------|------------|------------------------|
| **MERRA-2** | ~50 km (0.5°×0.625°) | Globale, 1980→ | **Entrée** du backbone Prithvi WxC (160 canaux, figé par le pré-entraînement) |
| **ERA5** | ~31 km | Globale, 1940→ | Entrée alternative des pipelines *non-Prithvi* (statistique / U-Net) |
| **ERA5-Land** | ~9 km | Globale terre | Entrée fine alternative (socle secondaire) |
| **CERRA** | **5.5 km** | Europe, 1984→ | **Cible** haute résolution d'entraînement *ou* entrée des pipelines non-Prithvi (socle principal Karpos) |
| **AROME (analyse)** | 1.3 km | France | Cible la plus fine possible (si disponible) |
| **MNT (COP-DEM)** | 25–90 m | — | **Conditionnement** orographique (élévation, pente, exposition) |
| **Sencrop / Netatmo** | ponctuel | Parcelles | **Référence terrain** pour calibration / fine-tuning local |

**Le point clé** : « basse résolution → haute résolution » n'implique pas que
les sources soient substituables. Une source est soit une **entrée** (ce que le
modèle voit), soit une **cible** (la vérité qu'on lui apprend à reproduire),
soit une **référence de calibration** (l'ancrage terrain). MERRA-2 est une
entrée *parce que* Prithvi a été entraîné dessus ; CERRA est une cible naturelle
*parce que* c'est la meilleure vérité gridée haute résolution sur l'Europe.

---

## 2. Pourquoi CERRA ne peut pas remplacer MERRA-2 *en entrée* de Prithvi

Le backbone `PrithviWxC` (2,3 B paramètres) attend, par construction
(`PrithviWxC.configs.load_model`) :

1. **Un jeu de variables MERRA-2 précis** — 160 canaux = variables de surface
   (T2M, U10M, V10M, PS, QV2M, fluxs EFLUX/HFLUX/SW/LW…) **+** variables
   verticales (T, U, V, QV, OMEGA, H…) sur **14 niveaux modèle**. CERRA expose
   un jeu de variables NWP différent, sur d'autres niveaux.
2. **Une grille globale fixe** 360×576 (~0.5°). CERRA est une grille régionale
   Lambert sur l'Europe à 5.5 km. Incompatible sans reprojection lourde — et
   même reprojeté, le contenu physique (résolution, variables) ne correspond pas.
3. **Des scalers de normalisation MERRA-2** (`climatology/musigma_*.nc`) appliqués
   *à l'intérieur* du `forward`. Ces statistiques sont propres à MERRA-2.
4. **Des poids conditionnés MERRA-2** : le réseau a appris la distribution
   MERRA-2. Lui présenter du CERRA brut = entrées hors-distribution → sorties non
   fiables.

> Conclusion : remplacer l'entrée par CERRA reviendrait à **re-pré-entraîner**
> (ou ré-adapter en profondeur) le foundation model — hors de portée. CERRA
> trouve sa place ailleurs : comme **cible**.

---

## 3. La cascade complète (rôles et résolutions)

```
        ┌──────────────────────────────────────────────────────────────┐
        │  ÉTAGE A — Contexte grande échelle (OPTIONNEL, Prithvi WxC)   │
        │                                                              │
        │   MERRA-2 (~50 km, global, 160 canaux)   ← ENTRÉE figée      │
        │            │                                                 │
        │            ▼  backbone(batch)  [gelé]                        │
        │   Prévision / représentation MERRA-2 (~50 km)                │
        └───────────────────────────┬──────────────────────────────────┘
                                     │  (champ grossier)
                                     ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  ÉTAGE B — Descente d'échelle conditionnée MNT → 1 km        │
        │                                                              │
        │   champ grossier (Prithvi  OU  ERA5/CERRA directement)       │
        │        +  MNT haute résolution (élévation, pente, expo)      │
        │            │                                                 │
        │            ▼  tête apprise (U-Net FiLM  ou  adapter DEM)     │
        │   Champ 1 km                                                 │
        │            ▲                                                 │
        │   CIBLE D'ENTRAÎNEMENT : CERRA 5.5 km (ou AROME 1.3 km)      │
        └───────────────────────────┬──────────────────────────────────┘
                                     │  (champ 1 km gridé)
                                     ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  ÉTAGE C — Calibration locale (capteurs in situ)            │
        │                                                              │
        │   champ 1 km  +  observations Sencrop / Netatmo (ponctuel)   │
        │            │                                                 │
        │            ▼  fine-tuning sparse  /  bias-correction  /  OI   │
        │   Champ calibré parcelle → indices paramétriques             │
        └──────────────────────────────────────────────────────────────┘
```

- **A → B** : MERRA-2 est *l'entrée*, son champ grossier descend en résolution.
- **B (cible)** : **CERRA est la vérité 5.5 km** qui supervise l'apprentissage de
  la tête. Le réseau apprend « grossier + relief → fin » en imitant CERRA.
- **B → C** : le champ 1 km est ensuite **recalé sur les capteurs** (Sencrop) —
  correction du biais résiduel local que ni MERRA-2 ni CERRA ne capturent
  (microclimats, fonds froids de vallée).

---

## 4. Calibration locale (Sencrop / Netatmo)

L'étage C ancre la sortie gridée sur le terrain. Trois méthodes, du plus simple
au plus appris :

1. **Bias-correction / Quantile mapping** — corrige la climatologie locale
   (déjà en place : `statistical/quantile_mapping.py`).
2. **Interpolation optimale (OI)** — assimile les observations ponctuelles dans
   le champ (déjà esquissé : `prtihvi_wxc/optimal_interpolation.py`).
3. **Fine-tuning supervisé sparse** — entraîne la tête de downscaling à matcher
   les stations QC'd, avec régularisation spatiale (déjà en place :
   `prtihvi_wxc/lightning_finetune.py`, supervision Netatmo). Sencrop s'y branche
   de la même façon (obs ponctuelles → loss aux positions des capteurs).

Sencrop et Netatmo jouent le **même rôle** (réseau de capteurs in situ) ; Sencrop
apporte une couverture agricole dédiée et un QC potentiellement meilleur. Les
deux passent par le **même conteneur** (`StationObs`), le même agrégat Tmin et la
même loss sparse — seul le loader change (`load_sencrop` / `load_netatmo_parquet`,
datasets `SencropFineTuneDataset` / `NetatmoFineTuneDataset`).

### Et le MNT dans le fine-tuning capteurs ? (valorisation de l'élévation)

Le MNT n'est **pas** spécifique à Prithvi : il intervient à **deux** niveaux,
indépendants du choix de chemin (A ou B).

1. **Dans le réseau de descente d'échelle** (étage B) : le U-Net est conditionné
   par le MNT (élévation, pente, exposition) via FiLM — c'est le cœur de la
   descente d'échelle orographique. Conservé tel quel dans le chemin CERRA.
2. **Dans la loss de calibration** (étage C) : une station Sencrop est à une
   **altitude précise**, souvent différente de l'altitude moyenne de sa maille
   1 km (fonds froids de vallée surtout). On corrige donc la prévision à
   l'altitude réelle du capteur **avant comparaison**, par lapse-rate :

   ```
   ŷ_station = ŷ_maille + γ · (z_station − z_maille)
   ```

   où `dz = z_station − z_maille` (m) est calculé à l'assignation station→grille
   (`stations.elevation_offset`) et `γ` le gradient thermique (≈ −6.5 K/km diurne,
   ≈ −4 K/km en régime de gel nocturne). C'est l'option `elevation_aware` /
   `obs_dz` de `SparseSupervisedLoss` (cf. `lightning_finetune.py`). **Sans cette
   correction**, le réseau apprendrait à reproduire un biais d'altitude au lieu
   du vrai champ.

> Donc oui, l'élévation de terrain reste pleinement valorisée dans
> CERRA → Sencrop — par le conditionnement MNT du réseau **et** par la correction
> d'altitude au point de calibration.

---

## 5. État d'implémentation

| Étage | Composant | État |
|-------|-----------|------|
| A | Backbone Prithvi WxC réel (`loader.py`) | ✅ câblé + testé (config jouet) |
| A | Pipeline d'entrée **MERRA-2** (`FrostNightDataset → batch`) | 🔴 **manquant** ([#1](https://github.com/maurinl26/karpos-downscaling/issues/1)) |
| B | U-Net FiLM (ERA5/CERRA → 1 km) | ✅ entraînement Lightning |
| B | Tête DEM sur prévision Prithvi | ✅ câblée (forward réel) |
| B | Pipeline statistique (lapse-rate + QDM) | ✅ |
| C | Fine-tuning sparse **Netatmo + Sencrop** | ✅ Lightning, source-agnostique (`StationFineTuneDataset`) |
| C | Calibration **MNT-aware** (correction d'altitude `obs_dz`) | ✅ `SparseSupervisedLoss(elevation_aware)` |
| **C — chemin B** | **Calibration U-Net (CERRA → 1 km → capteurs)** | ✅ `deep_learning/sparse_calibration.py` (`UNetSparseCalibrationModule`) |
| **C — chemin B** | **`coarse_provider` CERRA** (entrées U-Net par nuit) | ✅ `deep_learning/cerra_provider.py` (`CERRACoarseProvider`) |
| C | Interpolation optimale | 🟡 esquisse |

> Le **chemin B est calibrable de bout en bout** : `UNetSparseCalibrationModule`
> appelle le U-Net `(x_met, x_dem)`, sélectionne le canal cible (T2m) et supervise
> sur stations sparse avec correction d'altitude ; `CERRACoarseProvider` fournit
> les entrées réelles, normalisées comme à l'entraînement. Indépendant du blocage
> MERRA-2 (#1).
>
> **Tmin = descente horaire puis min** (`calibration.hourly=true`, défaut) : on
> descend en résolution **chaque heure** de la nuit (le U-Net voit des champs
> instantanés, en distribution), puis on prend le **min des prédictions 1 km**.
> Préférer cette voie à `hourly=false` (réduction du champ coarse avant le U-Net),
> qui présenterait au réseau un champ Tmin agrégé hors-distribution.
>
> En une commande (entry point Hydra) :
>
> ```bash
> run-calibration                                   # config calibration/ par défaut
> run-calibration cluster=cloud calibration.epochs=50
> run-calibration calibration.reduce=mean calibration.elevation_aware=false
> ```
>
> Pré-requis données : fichiers CERRA (`cerra_<date>.nc`), MNT (`data.dem_attrs`),
> exports Sencrop (`sencrop_<date>.csv`), checkpoint U-Net entraîné + stats de
> normalisation (chemins dans `configs/calibration/default.yaml`).
>
> **Onboarding des vraies données** (`CERRACoarseProvider`) :
> 1. **Diagnostiquer** un fichier : `run-calibration calibration.inspect=true` →
>    liste variables présentes/manquantes, grille vs MNT, nb de pas horaires.
> 2. Si les noms diffèrent → `calibration.var_map='{t2m: 2t, u10: 10u, ...}'`.
> 3. Si la grille CERRA ≠ grille MNT → `calibration.regrid=true` (rééchantillonne
>    sur le MNT ; nécessite l'extra `regrid` (xesmf) ou des lat/lon 1D).
> Coordonnées `latitude`/`longitude`/`valid_time` harmonisées automatiquement ;
> variable attendue absente → erreur explicite listant les variables disponibles.

---

## 6. Deux chemins produits — recommandation

> **Décision (parking Prithvi).** Le chemin **B est le chemin produit**. Le chemin
> A (Prithvi) est **parké — recherche, hors chemin critique** ([issue #1](https://github.com/maurinl26/karpos-downscaling/issues/1),
> label `parked`). Raison : Prithvi prévoit au pas global ~50 km, plus grossier que
> CERRA ; il n'apporte pas de finesse et la descente ×50 exigerait l'archi Granite
> pour un gain nul. Le code (backbone + entrée MERRA-2) reste isolé et réactivable.

### Chemin A — *Prithvi-centré* (⏸️ parké, recherche)
`MERRA-2 → Prithvi (2,3 B) → tête DEM → 1 km → calibration capteurs`
- **+** contexte atmosphérique grande échelle appris, cohérence physique.
- **−** MERRA-2 ~50 km **plus grossier** que CERRA ; descente régionale ×50 hors
  tête DEM (⇒ archi Granite) ; GPU lourd. **Pas de valeur sur la chaîne actuelle.**

### Chemin B — *Réanalyse-directe CERRA* (léger, déjà construit)
`CERRA 5.5 km (ou ERA5-Land 9 km) → U-Net FiLM / statistique → 1 km → calibration`
- **+** s'appuie sur le **socle CERRA** (cœur Karpos), pipelines déjà en place,
  pas de dépendance MERRA-2/foundation model, validable rapidement.
- **−** pas de *prior* foundation model ; repose sur la calibration locale pour
  la finesse.

> **Recommandation** : pour le produit centré CERRA, **le chemin B est le socle**.
> CERRA y est l'**entrée** (déjà à 5.5 km, plus fin que MERRA-2), descendue à
> 1 km par le U-Net/statistique conditionné MNT, puis calibrée Sencrop. Prithvi
> (chemin A) reste pertinent comme **brique de recherche** (prior grande échelle,
> variables absentes de CERRA) mais n'est pas sur le chemin critique du produit.
>
> Dans les deux cas : **MERRA-2 = entrée de Prithvi uniquement**, **CERRA = cible
> ou entrée des pipelines directs**, **Sencrop = calibration finale**.

---

## Références

- Schmude et al. (2024) *Prithvi WxC: Foundation Model for Weather and Climate*,
  arXiv:2409.13598 (pré-entraînement MERRA-2).
- Yu et al. (2025) *Fine-Tuning Foundational Models for Downscaling*, NASA NTRS 20250006603.
- CERRA : Copernicus Regional ReAnalysis for Europe (5.5 km).
- MERRA-2 : NASA Modern-Era Retrospective analysis for Research and Applications, v2.
