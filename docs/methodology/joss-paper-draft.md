---
type: trame-publication
project: Karpos
cible-journal: JOSS (Journal of Open Source Software)
status: brouillon-trame
epic: EPIC 1 — Scientifique POD/FAR/Biais
related-code: parametric_insurance
tags: [karpos, recherche, publication, joss, open-source]
created: 2026-06-17
---

# Trame publication JOSS — `parametric_insurance`

> **Objectif** : publier un papier court (250–1 000 mots) au JOSS pour **fixer l'architecture logicielle** de `parametric_insurance`, obtenir un **DOI citable** sur le code, et instituer la **rigueur de tests** comme moat technique (PI : un compétiteur ne pourra pas répondre "moi aussi" à un papier publié et indexé).

## 1. Positionnement et angle

### Pourquoi JOSS et pas JOSS-like (SoftwareX, JORS)
- JOSS = **standard de facto** des packages scientifiques modernes (xarray, scikit-image, fastai, chillR…) ; signal de qualité ingénierie.
- Revue **par les pairs sur le code lui-même** (pas l'usage scientifique — c'est l'objet du papier GMD parallèle).
- Cycle court (3-8 semaines de review en moyenne).
- Gratuit, OA, DOI Zenodo automatique.
- Critères acceptation : substantialité ≥ 3 mois-personne effort, fonctionnalités originales, tests, documentation, communauté potentielle (cf. [JOSS author guide](https://joss.readthedocs.io/en/latest/submitting.html)).

### Angle du papier
**Un toolkit Python intégré pour l'assurance paramétrique agricole climatique**, articulant :
- Calcul d'indices de gel sur séries de température (IND-01 à IND-05),
- Couplage phénologique chilling × forcing (PhenoFlex Luedeling 2021) entièrement réimplémenté en Python pur (sans dépendance R en production),
- Pipeline de downscaling NWP (U-Net résiduel + FiLM·DEM + calibration capteurs out-of-sample),
- Backtesting et metrics standard (POD/FAR/CSI, courbes ROC).

**Ce que la publication "fixe"** : l'API publique (signatures des fonctions phare), la structure modulaire `indices/`, `downscale/`, `phenoflex/`, `backtest/`, et la couverture de tests d'adéquation au modèle de référence chillR (14/14, cf. [[PhenoFlex — Couplage Chilling × Forcing (abricot)]]).

## 2. Cible : qui lit, qui cite

- Chercheurs en agroclim / phéno (INRAE, Univ. Bonn — Luedeling, Wageningen — Dalhaus, ETH — Finger)
- Praticiens assurance paramétrique (porteurs MGA, équipes data assureurs)
- Communauté open-source géosciences (xarray, climpred ecosystem)

→ Citations attendues à 3 ans : 5-15. Pas un blockbuster, mais un **point d'ancrage scientifique** indexable.

## 3. Pré-requis avant soumission (état du code)

| Item | État actuel | Action requise |
|---|---|---|
| Licence open-source (Apache 2 ou MIT) | À vérifier | Choisir Apache 2 (clause brevet, plus protecteur) — ajouter `LICENSE` |
| README.md substantiel (install, quickstart, citation) | À étoffer | Inclure exemples PhenoFlex + downscaling + backtest |
| Documentation API (Sphinx / mkdocs) | Probable absente | Mettre en place `mkdocs-material` + autodoc |
| Tests unitaires + CI | **14/14 PhenoFlex au vert** | Étendre couverture sur `downscale/` + `indices/IND-04/05` |
| `CONTRIBUTING.md` + `CODE_OF_CONDUCT.md` | À créer | Templates GitHub standards |
| Statement of need (papier) | À rédiger | Cf. §5 |
| `paper.md` + `paper.bib` à la racine | À créer | Format Markdown standard JOSS |
| Repo public avec releases tagguées | À vérifier | Première release `v0.3.0` taguée, avec changelog |
| DOI Zenodo lié | Automatique JOSS | À activer post-soumission |

**Décision IP** : open-sourcer `parametric_insurance` = paradoxal vu le NO-GO + IP DL à protéger ? **Non.** L'IP DL = poids du modèle entraîné + calibration capteur OOS + paramètres calés sur données propriétaires Sencrop. Le code méthode peut être ouvert, les poids et la calibration restent propriétaires (mention explicite dans le README : *"trained weights and calibration not included; reach out for licensing"*). C'est le pattern de Meta Llama, Stability AI, etc.

## 4. Structure cible — sections JOSS (template officiel)

JOSS impose une structure stricte ; voici la trame contenu pour chaque section, avec longueur cible.

### `Title` (≤ 250 caractères)

Brouillon : *"parametric_insurance: a Python toolkit for parametric agricultural insurance — frost indices, phenology coupling, and reanalysis downscaling"*

### `Authors`

- Loïc Maurin (EI Karpos, France) — corresponding author. ORCID requis.
- À considérer en co-auteur (à arbitrer) : Christian Kühnlein (ECMWF, héritage PMAP/FVM), Stefano Ubbiali (IAC ETH), si contribution code/idée significative. Sinon → acknowledgements.

### `Summary` (≤ 250 mots)

Brouillon de contenu :
> Parametric agricultural insurance pays out on objective weather thresholds rather than indemnified losses, eliminating the moral hazard and assessment cost that have made traditional crop insurance economically unsustainable in marginal climates. Yet existing implementations rely on coarse-grid reanalyses, fixed phenological calendars, and ad-hoc trigger formulations that produce unacceptable basis risk. `parametric_insurance` is a Python package that integrates three building blocks needed for production-grade parametric frost indices: (1) a kilometre-scale downscaling pipeline combining residual U-Net with FiLM-conditioned digital elevation, calibrated out-of-sample on in-orchard sensor networks; (2) a fully Python implementation of the PhenoFlex chilling × forcing coupling (Luedeling et al. 2021) verified against the reference chillR R package; (3) five frost indices (IND-01 to IND-05) parameterised by FAO-validated phenological-stage thresholds. The package supports end-to-end backtesting (POD/FAR/CSI, ROC) and is used in production to deliver frost indices for tree-fruit crops in the Rhône valley.

### `Statement of need` (≤ 400 mots)

Argument structuré en trois temps :
1. **Le verrou commercial** : les produits d'assurance paramétrique gel publiés (notamment Airbus IPP) reposent sur remote sensing pur (R² 0.71-0.81) sans publication de métriques POD/FAR/CSI ; **angle mort exploitable** (cf. [[Benchmark IPP Airbus - Requirements Indice]]).
2. **Le verrou méthodo** : la prédiction de stade par GDD depuis le 1ᵉʳ janvier (proxy standard) ignore la dormance, produit un **risque de base temporel** identifié par Dalhaus et al. (2018), et n'est ré-utilisable que par une réimplémentation manuelle.
3. **Le vide outil** : aucun package Python intégrant downscaling + phénologie + indices + backtesting n'existe à ce jour ; chillR est R-only et limité à la phéno.

→ `parametric_insurance` comble le vide outil. Cas d'usage actuels : développement d'indices pour arboriculture (abricot, cerise), couplage avec capteurs Sencrop pour calibration locale, génération de rapports historiques parcellaires.

### `Functionality` (≤ 500 mots)

Structurer en sous-sections, une par module :

#### `indices/` — Calcul d'indices de gel
- `compute_frost_damage_index(Tmin, stage)` — croisement T_min × T₁₀/T₉₀ par stade (tables FAO Proebsting & Mills)
- `compute_cumulative_seasonal_index()` — IND-04
- `compute_chill_portions()` → migré vers `phenoflex.chill_portions_timeseries()` (bug IND-05 corrigé, cf. [[PhenoFlex — Couplage Chilling × Forcing (abricot)]])

#### `phenoflex/` — Couplage chilling × forcing
- `phenoflex_run(t_hourly, params) -> {x, y, z, t1}`
- `predict_stage_dates(t_hourly, species, season)` — StageTimeline daté
- `evaluate_frost_night(Tmin_skin, stage)` — double seuil maturation + gel
- Forward 100 % Python, calage offline via `chillR` (R)

#### `downscale/` — Downscaling NWP
- `UNetResidual` + `FiLMDEM` (PyTorch)
- `StationCalibrator` — biais médian par station out-of-sample
- Loader CERRA 5,5 km + ARRA 1,3 km (Zarr)

#### `backtest/` — Évaluation
- `compute_metrics(predictions, truth) -> {POD, FAR, CSI, AUC}`
- `roc_curve(scores, truth)` — version pinball-aware
- Splits anti-fuite (event-wise, leave-one-season-out, blocs spatiaux)

### `Example` (snippet exécutable ≤ 30 lignes)

```python
from parametric_insurance.phenoflex import phenoflex_run, predict_stage_dates
from parametric_insurance.indices import compute_frost_damage_index
from parametric_insurance.io import load_cerra_hourly

t_hourly = load_cerra_hourly(bbox=(4.9, 44.3, 5.3, 44.5), period="2020-09-01/2021-06-30")
timeline = predict_stage_dates(t_hourly, species="apricot_bergeron", season="2020-2021")
damages = compute_frost_damage_index(t_hourly.min("hour"), timeline)
print(damages.where(damages.severity > 0))
```

### `Acknowledgements`

- Projet **PMAP/FVM** (ECMWF / ETH Zurich) — héritage scientifique du noyau dynamique nouvelle génération IFS, à l'origine du savoir-faire downscaling.
- **Eike Luedeling** (Univ. Bonn) — package `chillR` de référence pour la phénologie.
- **Tobias Dalhaus** (Wageningen) — apport méthodologique sur la réduction du risque de base par information phénologique (Dalhaus et al. 2018).
- **Sencrop** — accès aux données capteurs in-verger (collaboration commerciale).

### `References` (BibTeX, ≥ 5)

Citations cœur :
- Luedeling et al. 2021 (PhenoFlex, AGRMET)
- Dalhaus et al. 2018 (basis risk, Sci. Reports)
- Fishman, Erez & Couvillon 1987 (Dynamic chill model, JTB)
- Anderson, Richardson & Kesner 1986 (GDH, Acta Horticulturae)
- FAO 2005 (Proebsting & Mills frost tables)
- Ronneberger et al. 2015 (U-Net, MICCAI)
- Perez et al. 2018 (FiLM conditioning, AAAI)

## 5. Calendrier de rédaction

| Étape | Charge | Échéance |
|---|---|---|
| Choix licence + LICENSE + CONTRIBUTING + CoC | 0,5 j | S26 |
| Étoffer README + ajouter exemples runnables | 1 j | S27 |
| Mettre en place mkdocs-material + autodoc | 1 j | S28 |
| Étendre couverture tests `downscale/` + `indices/` | 2 j | S28-S29 |
| Rédiger `paper.md` + `paper.bib` | 1 j | S29 |
| Tag release `v0.3.0` + changelog | 0,5 j | S30 |
| Soumission JOSS | 0,5 j | **S30 (fin juillet)** |
| Réponse au review (1-2 rounds typiques) | 2-3 j | S33-S36 |
| Acceptation cible | — | S38-S40 (sept.) |

**Total charge rédaction** : ~8 j sur S26-S30 (étalable, pas concentré sur S26).

## 6. Liens

- [[Trame publication GMD — Downscaling-calibration gel arboriculture]] (papier scientifique parallèle, validation)
- [[Trame DEP v0 — note d'analyse technique]]
- [[PhenoFlex — Couplage Chilling × Forcing (abricot)]]
- [[Campagne Sencrop S23 — preuve de thèse gel]]
- [[Produit - 10-06-2026]] (livrable v0.3, source des chiffres)
- [[feedback-perf-framing-sap]] (cadrage Sûr/Actuel/Projection à respecter dans le papier)
- [[Sprint S26 — Landing EI-Karpos (Option B)]]
