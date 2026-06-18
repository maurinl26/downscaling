---
type: trame-publication
project: Karpos
cible-journal: GMD (Geoscientific Model Development, Copernicus / EGU)
status: brouillon-trame
epic: EPIC 1 — Scientifique POD/FAR/Biais
related-code: parametric_insurance
tags: [karpos, recherche, publication, gmd, validation-scientifique]
created: 2026-06-17
---

# Trame publication GMD — Downscaling et calibration de réanalyses pour l'assurance paramétrique gel arboricole

> **Objectif** : publier une validation scientifique complète du dispositif Karpos v0.3 dans **GMD (Geoscientific Model Development, Copernicus/EGU)**. Le papier sert simultanément (i) de caution publique pour la marque blanche / partenaires assureurs, (ii) de réponse anticipée au futur dépôt Comité des Indices (CNES, INRAE, Météo France, IDELE, ISG), (iii) de preuve différenciante face à Airbus IPP qui n'a jamais publié de POD/FAR/CSI.

## 1. Positionnement et angle

### Pourquoi GMD
- **Open Access** (Copernicus), comité de lecture transparent (les reviews et révisions sont publiées), cycle 4-9 mois.
- Lectorat exactement ciblé : modélisation atmosphérique appliquée, downscaling, validation operationnelle.
- Alternatives écartées : *Agric. For. Meteorol.* (cycle 8-14 mois, paywall partiel), *Sci. Reports* (généraliste, signal moins fort en assurance), *Atmospheric Measurement Techniques* (cible plus instrumentale).

### Titre brouillon (3 variantes à arbitrer)

- **V1 — méthodo focus** : *"A residual U-Net downscaling of CERRA reanalyses with in-orchard sensor calibration for parametric frost insurance: validation over apricot orchards in the Rhône Valley"*
- **V2 — produit focus** : *"From 5.5 km reanalysis to operational frost indices: an end-to-end deep-learning pipeline validated against the Sencrop sensor network on apricot orchards"*
- **V3 — assurance focus** : *"Reducing basis risk in parametric frost insurance: phenology-aware downscaling and out-of-sample calibration of CERRA reanalyses"*

Reco : **V3** — c'est l'angle assurance qui maximise la citabilité côté MGA/assureurs et qui amorce le moat commercial.

### Auteurs proposés

- Loïc Maurin (EI Karpos) — *first + corresponding*
- À solliciter (ordre d'approche, avant rédaction) :
  - **Stefano Ubbiali** (IAC ETH Zurich) — caution dynamique downscaling, héritage PMAP/FVM
  - **Christian Kühnlein** (ECMWF) — PI PMAP, autorité IFS
  - **Tobias Dalhaus** (Wageningen, AECP ETH) — caution risque de base, méthodologie phéno-insurance
  - **Robert Finger** (AECP ETH) — caution assurance indicielle
- Hypothèse plausible : 3-4 auteurs au total, ordre Maurin – Ubbiali – Dalhaus – Kühnlein (à négocier).

**Décision préalable à la rédaction** : sécuriser la contribution de chacun par mail avant d'avancer (réunion type 30 min × 3 contacts).

## 2. Question scientifique et contribution

### Question
> *Une chaîne de downscaling deep-learning calibrée out-of-sample sur un réseau de capteurs in-verger peut-elle atteindre des performances de discrimination du gel (POD ≥ 80 %, FAR ≤ 20 %) suffisantes pour soutenir un contrat d'assurance paramétrique sur arboriculture fruitière ?*

### Contributions explicites
1. **Méthodologique** — chaîne intégrée U-Net résiduel + FiLM·DEM + calibration biais médian out-of-sample, applicable à toute paire (réanalyse, réseau capteur).
2. **Empirique** — première validation publiée (à notre connaissance) sur réseau Sencrop densifié in-verger Drôme-Ardèche, avec courbes ROC et table comparative CERRA brut vs ERA5-Land vs downscaling vs calibration.
3. **Opérationnelle** — démonstration que le **socle de réanalyse** (CERRA 5,5 km vs ERA5-Land 9 km) compte davantage que la profondeur du réseau de descente d'échelle, conséquence directe pour le choix d'architecture en production.
4. **Reproductibilité** — pipeline publié en open source ([[Trame publication JOSS — parametric_insurance|JOSS parallèle]]) ; jeu de données capteur Sencrop accessible sur demande (NDA NDA Sencrop à arbitrer).

## 3. Pré-requis avant rédaction (état du dispositif scientifique)

| Item | État | Action |
|---|---|---|
| Test pur 2025 — métriques figées | ✅ v0.3 livrée 10/06 | Conservation gelée jusqu'à publication |
| Documentation méthode (protocole splits, calibration, métriques) | ✅ [[Campagne Sencrop S23 — preuve de thèse gel]] | À synthétiser en papier |
| Comparaison CERRA vs ERA5-Land à calibration égale | ✅ Faite | Table à reprendre |
| Reproductibilité — code + données accessibles | Code OK (cf. JOSS) ; données capteur = NDA Sencrop | Négocier release agrégée (pas station-by-station) avec Martin Ducroquet |
| Étude d'ablation (RAW / U-Net seul / U-Net + calib) | ✅ déjà table v0.3 | OK |
| Comparaison stratifiée par régime radiatif vs advectif | ⚠️ partielle | À compléter — c'est un papier-killer si on tient le radiatif/advectif (cf. [[Downscaling gel - discrimination radiatif-advectif (FiLM)]]) |
| Validation phénologique (PhenoFlex z_c calé) | 🔴 non calé — floraison simulée mi-avril vs observée début mars | **Bloquant** — calage offline chillR sur DIVAE/PHENOCLIM à finir avant rédaction |
| Diagrammes de fiabilité (calibration probabiliste) | À faire | 1-2 j |

**Verrou principal avant rédaction** : calage PhenoFlex z_c. Sans ça, on peut publier la partie downscaling+calibration mais pas le couplage phéno → le papier perd 30 % de sa valeur.

## 4. Structure cible — sections GMD

GMD impose une structure souple mais les sections suivantes sont attendues.

### `Abstract` (≤ 250 mots)

Brouillon de contenu :
> Parametric crop insurance pays out on objective meteorological thresholds rather than indemnified losses, but its commercial viability hinges on minimizing basis risk — the gap between physical event and indemnity decision. We present and validate a deep-learning pipeline that downscales kilometre-scale reanalyses (CERRA 5.5 km, ARRA 1.3 km) and calibrates them against an in-orchard sensor network (Sencrop), with phenology-aware frost indices coupled via the PhenoFlex chilling × forcing model (Luedeling et al. 2021). On a pure 2025 hold-out (4 110 observations, 379 frost events) over apricot orchards in the Rhône Valley, the chain achieves POD = 0.80 at FAR < 0.20 (AUC = 0.987) — a fivefold improvement over raw CERRA (POD = 0.47) and a nine-fold improvement over raw ERA5-Land (POD = 0.09). We further show that the choice of reanalysis backbone dominates downscaling network depth: raw CERRA already outperforms calibrated ERA5-Land. Phenological stage prediction via PhenoFlex reduces the basis risk identified by Dalhaus et al. (2018) by enabling stage-specific frost thresholds (T₁₀/T₉₀ Proebsting & Mills). The full pipeline is open-source [reference to JOSS DOI]; trained weights and station calibration are commercially licensable.

### `1. Introduction` (~1500 mots)

- Contexte assurance climatique : retrait des assureurs sur cultures à forte sinistralité chronique (illustrer avec abricot Baronnies — 80 % pertes structurelles).
- Limites des produits actuels : Airbus IPP remote-sensing pur sans POD/FAR publiés ; MRC piégée par la moyenne olympique.
- Verrou scientifique : risque de base (basis risk) — temporel (stade) et spatial (résolution réanalyse).
- Notre contribution (rappel §2).
- Plan du papier.

### `2. Data and study area` (~1000 mots)

- **Zone d'étude** : Drôme-Ardèche, focus Baronnies provençales — caractéristiques topographiques (vallées encaissées, cold-air pooling), espèce dominante (abricot Bergeron), sinistralité observée.
- **Réanalyses** : CERRA 5,5 km (1984-présent), ERA5-Land 9 km (référence), ARRA 1,3 km (référence cible).
- **Réseau capteurs** : Sencrop ~4 000 stations Drôme-Ardèche, in-verger ; protocole QC ; recouvrement temporel.
- **Vérité terrain** : déclarations de pertes Agreste SAA (si accessibles) + témoignages locaux pour événement 2021.

### `3. Methods` (~2500 mots, le cœur)

#### 3.1 Architecture downscaling
- U-Net résiduel : sortie = entrée + correction (évite régression vers climatologie douce)
- Conditionnement FiLM·DEM : `γ(DEM)·x + β(DEM)` à chaque niveau encodeur
- Pré-entraînement toutes nuits → fine-tuning nuits de gel queue pondérée
- Schéma : pré-entraînement + fine-tuning, perte pinball quantile (pas MSE) — *cf. dégradation MSE en annexe*

#### 3.2 Calibration capteurs out-of-sample
- Biais médian par station Sencrop, ajusté sur 2022-2024
- Repli sur biais global pour stations non vues
- Seuil de décision τ* optimisé sous contrainte FAR ≤ 20 %

#### 3.3 Couplage phénologique PhenoFlex
- Modèle Dynamique (Chill Portions, Fishman 1987 ; paramétrisation chillR de Luedeling)
- GDH Anderson 1986 (sigmoïde 3 paramètres)
- Transition sigmoïde chill → heat, paramètre s1
- Seuils T₁₀/T₉₀ par stade BBCH (Proebsting & Mills, FAO 2005)

#### 3.4 Indices de gel
- IND-01 à IND-05 (cf. [[Phénologie - Indices paramétriques gel]])
- Double seuil maturation + intensité — apport vs proxy GDD depuis 1ᵉʳ jan.

### `4. Validation protocol` (~800 mots)

- **Splits** : train 2015-2021 / dev-calibration 2022-2024 / **test pur 2025** / 2026 réservé.
- **Aucune fuite** : biais et seuil τ* ajustés sur dev, appliqués tels quels sur test.
- **Métriques** : POD = TP/(TP+FN), FAR = FP/(FP+TP), CSI, AUC, diagrammes de fiabilité.
- **Stratification** : par altitude, par régime radiatif/advectif, par stade BBCH.
- **Bootstrap event-wise** pour intervalles de confiance.

### `5. Results` (~2000 mots)

Au moins 4 figures et 2 tables :
- Figure 1 — Zone d'étude + localisation stations Sencrop + DEM
- Figure 2 — Courbes ROC sur test pur 2025, par méthode (RAW CERRA, RAW ERA5, downscaled, downscaled+calibrated)
- Figure 3 — Carte des biais par station Sencrop avant/après calibration
- Figure 4 — Diagrammes de fiabilité par régime radiatif/advectif
- Table 1 — Résumé performances POD/FAR/CSI/AUC par méthode × backbone
- Table 2 — Stratification altitude × régime

Récit attendu :
1. RAW CERRA bat RAW ERA5-Land → le **socle de réanalyse domine** la profondeur du réseau de downscaling.
2. La **calibration capteurs** double encore le POD à FAR fixe.
3. Le **gain est principalement attribuable au radiatif** (cuvettes, cold-air pooling) — c'est ce que le conditionnement FiLM·DEM résout.

### `6. Discussion` (~1500 mots)

- **Comparaison avec IPP Airbus** : remote sensing pur, R² 0.71-0.81 ; pas de métriques de discrimination publiées. Notre POD/FAR comble ce vide ; nos avantages : pas de problème de couverture nuageuse, historique 60+ ans, variables physiques cohérentes.
- **Limite densité capteurs** : POD plafonne là où la calibration repose sur < 5 nuits observées par station → roadmap densification ARRA + Weenat.
- **Limite phénologique** : `z_c` calés ce trimestre (validation DIVAE Gotheron/Toulenne, PHENOCLIM AgroClim).
- **Transférabilité** : la chaîne s'applique à toute culture pour laquelle (i) un réseau capteur dense existe et (ii) des seuils T₁₀/T₉₀ par stade sont publiés. Cultures candidates : cerise, pêche, vigne, kiwi.

### `7. Conclusion` (~500 mots)

- POD 0,80 @ FAR < 20 % sur test pur 2025 = seuil commercial atteint en gel arboricole sur la zone.
- Le couplage downscaling + calibration + phénologie réduit le risque de base à la fois spatial et temporel.
- Pipeline open-source (JOSS), poids commerciables.

### `Code and data availability`

- Code : `parametric_insurance` v0.3.0 — Apache 2.0 — DOI Zenodo via JOSS.
- Poids entraînés et calibration station : disponibles sur licence commerciale (contact corresponding author).
- Données capteur Sencrop : agrégées (commune-mois, anonymisées) sur demande ; data brute sous NDA Sencrop.
- Réanalyses CERRA / ERA5-Land : librement accessibles via CDS Copernicus.

### `Author contributions` · `Acknowledgements` · `References`

(standards, environ 30-40 références au total)

## 5. Calendrier de rédaction

| Étape | Charge | Échéance cible |
|---|---|---|
| Calage offline PhenoFlex z_c (verrou bloquant) | 5-7 j | S26-S30 |
| Stratification radiatif/advectif (papier-killer) | 3-5 j | S28-S32 |
| Validation co-auteurs (Ubbiali, Dalhaus) | 3-4 réunions | S26-S30 |
| Rédaction draft v1 (toutes sections) | 10-12 j | S33-S40 (sept.) |
| Revue interne par co-auteurs (rounds 1-2) | 5 j | S41-S44 (oct.) |
| Soumission GMD | 1 j | **S45 (début nov.)** |
| Review GMD (interactive discussion) | — | S47-S55 (déc.-fév.) |
| Acceptation cible | — | **S10-S15 2027 (mars-avril)** |

**Charge totale rédaction** : ~25-30 j sur S26-S44 (étalable sur 5 mois).

## 6. Risques et garde-fous

| Risque | Garde-fou |
|---|---|
| Calage PhenoFlex bloque indéfiniment (données DIVAE inaccessibles) | Repli proxy PHENOCLIM AgroClim (publique) — moins fin mais publiable |
| Sencrop refuse release agrégée des données | Repli station-by-station anonymisée sur sous-échantillon |
| Co-auteurs ETH ralentissent / refusent | Solo + acknowledgements seulement — papier reste publiable |
| Reviewers GMD demandent extension à 2026 (jeu réservé) | Préparer à l'avance le rejeu 2026 sans toucher au modèle entraîné |
| Concurrence Airbus / Sofar publie un papier équivalent entre temps | Première soumission rapide (nov. 2026) + JOSS en parallèle pour blocage de l'antériorité |

## 7. Articulation avec le reste du dispositif

- **JOSS papier** ([[Trame publication JOSS — parametric_insurance]]) : sort en premier (S30), fixe le code et donne le DOI à citer dans le GMD.
- **DEP v0** ([[Trame DEP v0 — note d'analyse technique]]) : invoque les métriques publiées du papier GMD comme caution scientifique (page "méthodologie").
- **Landing** ([[Sprint S26 — Landing EI-Karpos (Option B)]]) : section "Preuve produit" pointera vers le pré-print HAL/Zenodo en attente de la publication GMD finale.
- **Pitch AG Syndicat septembre** : *"un papier scientifique en cours de soumission sur ces résultats"* = caution forte vs procès en crédibilité du dossier antérieurement fermé.

## 8. Liens

- [[Trame publication JOSS — parametric_insurance]]
- [[Trame DEP v0 — note d'analyse technique]]
- [[Campagne Sencrop S23 — preuve de thèse gel]]
- [[Produit - 10-06-2026]]
- [[PhenoFlex — Couplage Chilling × Forcing (abricot)]]
- [[Rapport — Modèle de chilling pour l'abricot des Baronnies]]
- [[Downscaling gel - discrimination radiatif-advectif (FiLM)]]
- [[Benchmark IPP Airbus - Requirements Indice]]
- [[feedback-perf-framing-sap]] (cadrage Sûr/Actuel/Projection à respecter dans abstract et conclusion)
- [[Sprint S26 — Landing EI-Karpos (Option B)]]
