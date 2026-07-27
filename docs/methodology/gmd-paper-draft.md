---
type: trame-publication
project: Karpos
cible-journal: GMD (Geoscientific Model Development, Copernicus / EGU)
status: brouillon-trame v3 — orienté PRÉVISION, chiffres honnêtes LOO/CSI/REV, sans PhenoFlex
epic: EPIC 1 — Scientifique POD/FAR/CSI + valeur de prévision
related-code: karpos-downscaling
langue: FR (EN régénéré à la demande — cf. workflow GMD)
created: 2026-06-17
updated: 2026-07-27
tags: [karpos, recherche, publication, gmd, prévision, calibration, REV, LOO]
---

# Trame publication GMD — Rendre les prévisions de gel exploitables à la parcelle par calibration in situ

> **Réécriture v3 (27/07/2026) — angle PRÉVISION.** Le papier est réorienté de « assurance / risque de base » vers la **valeur de prévision** : une prévision (réanalyse ou AROME) à maille kilométrique est **aveugle au gel radiatif de parcelle** ; une **calibration par capteurs in situ** la rend **exploitable** pour la décision. Résultat-phare = **×10 de détection** en prévision opérationnelle. L'assurance paramétrique redevient **une application** (discussion). Chiffres **honnêtes en leave-one-station-out** (abandon du « POD 0,80 / AUC 0,987 »). **PhenoFlex retiré** (feature app). Ablation FiLM → **étude compagnon** Risser/ANITI 2027.

## Résumé

*(Rédigé en français ; version anglaise régénérée à la demande — cf. workflow GMD.)*

> La détection du gel de printemps **à l'échelle de la parcelle** est un verrou opérationnel : prévisions et réanalyses à maille kilométrique lissent la queue froide et sont **aveugles au gel radiatif de cuvette**, là où le risque se concentre. Nous présentons une chaîne de descente d'échelle par apprentissage profond qui affine la réanalyse CERRA (5,5 km) et la **calibre sur un réseau de capteurs in-verger (Sencrop)** par assimilation à fonction de base radiale. En validation stricte **leave-one-station-out** (2022-2025, ~12 500 nuits-station, vergers d'abricotiers de la vallée du Rhône, aucune fuite), la chaîne atteint un **CSI de 0,38** au seuil agronomique −2,2 °C (POD 0,48 / FAR 0,36), contre **0,27** pour la seule calibration statistique et **0,17** pour la réanalyse brute — la **supervision in situ porte l'essentiel du gain**. Appliquée à une prévision **AROME opérationnelle à J-1**, la même calibration fait passer la détection à la parcelle de **5 % à 50 % (×10)**, et se transfère d'une saison à l'autre (CSI 0,34, train 2024 → test 2025). Une analyse **coût-perte** (valeur économique relative) montre que, sur la plage coût-perte réelle des exploitants, la chaîne capture **80 à 85 %** de la valeur d'une prévision parfaite (~**460 à 1 960 €·ha⁻¹·an⁻¹** évités). Le levier de la valeur n'est pas le modèle atmosphérique mais le **réseau de capteurs in situ**. Pipeline open source ; poids et calibration commerciables.

## 1. Positionnement et angle

### Pourquoi GMD
Open Access Copernicus, comité transparent, cycle 4-9 mois ; lectorat modélisation atmosphérique appliquée / descente d'échelle / **prévision opérationnelle** — l'angle prévision est pile dans la cible GMD.

### Titre (orienté prévision)
> *« Making frost forecasts actionable at the parcel scale: in-situ sensor calibration of downscaled reanalyses and AROME, validated leave-one-station-out »*

Variante FR : *« De la maille kilométrique à la parcelle : calibration in situ de prévisions de gel descendues en échelle, et sa valeur décisionnelle »*.

### Auteurs et affiliation
- **Loïc Maurin¹** — *premier auteur, corresponding*. ¹ École Nationale de la Météorologie (ENM), Météo-France, Toulouse. ORCID 0009-0004-8117-4850.
  - ⚠️ Affiliation de publication = **ENM / Météo-France seule** ; **Karpos (EI)** en *Acknowledgements*.
- Co-auteurs à confirmer : **Stefano Ubbiali** (IAC ETH — descente d'échelle), **Tobias Dalhaus** (Wageningen/AECP ETH — valeur décisionnelle).
- **Étude compagnon (2027)** : ablation FiLM·MNT avec **Laurent Risser (ANITI/Toulouse-INP)**.

## 2. Question scientifique et contribution

### Question
> *Une prévision de gel descendue en échelle et **calibrée hors-échantillon sur un réseau de capteurs in-verger** devient-elle **exploitable à la parcelle** — c'est-à-dire skillée ET économiquement valable pour la décision — là où la maille kilométrique brute est aveugle ?*

Mesurée en **CSI (leave-one-station-out)** et en **valeur économique relative (REV)**.

### Contributions
1. **Résultat-phare (prévision)** — une prévision AROME brute à J-1 ne détecte que **5 %** des gels à la parcelle ; la calibration Sencrop la porte à **50 %** (**×10**), et elle **généralise d'une saison à l'autre**.
2. **Empirique (validation)** — première validation publiée (à notre connaissance) sur réseau **Sencrop** in-verger densifié, en **strict leave-one-station-out** (aucune fuite) : la **calibration in situ est le composant porteur du skill**, devant le socle de réanalyse et la profondeur du réseau.
3. **Décisionnelle** — **valeur de prévision** par analyse coût-perte (REV, Richardson/Wilks) : quantification de l'euro évité par la prévision sur la plage coût-perte réelle des exploitants.
4. **Méthodologique** — chaîne U-Net résiduel + conditionnement MNT + supervision station hors-échantillon, applicable à toute paire (prévision/réanalyse, réseau capteur).
5. **Reproductibilité** — open source (JOSS), protocole et graines documentés.

## Introduction

Le gel de printemps est, pour l'arboriculture et la viticulture, l'aléa qui peut effacer une récolte en une seule nuit. Sa fréquence augmente paradoxalement sous le réchauffement : des hivers plus doux avancent le débourrement, exposant des organes floraux vulnérables à des gelées tardives inchangées — l'épisode d'avril 2021, qui a ravagé une large part des vergers et vignobles français, en est l'illustration. Face à ce risque, deux décisions dépendent d'une information fiable **à l'échelle de la parcelle** : la **protection active** la nuit même (aspersion, tours à vent, bougies — coûteuse, à n'engager qu'à bon escient) et, en aval, le **transfert de risque** assurantiel.

Or l'information disponible n'est pas à la bonne échelle. Les prévisions et réanalyses opérationnelles raisonnent à la maille kilométrique — AROME à 1,3-2,5 km, CERRA à 5,5 km, ERA5-Land à 9 km. À ces résolutions, la **queue froide est lissée** : le gel radiatif de cuvette, produit d'un refroidissement nocturne local par rayonnement et d'accumulation d'air froid en fond de vallée (*cold-air pooling*), n'est pas résolu — précisément là où le gel se concentre et frappe le plus fort. Les approches existantes ne franchissent pas ce dernier kilomètre : la descente d'échelle statistique corrige un biais moyen sans restituer la structure fine ; les produits de télédétection (par ex. Airbus IPP) rapportent des corrélations (R²) mais **aucune métrique de détection** (POD/FAR) publiée à la parcelle ; l'assurance récolte multirisque repose sur des moyennes olympiques peu résolues spatialement.

La thèse de ce travail est que **le dernier kilomètre ne se franchit pas par un meilleur modèle atmosphérique, mais par une calibration au contact du terrain** — l'assimilation d'un réseau dense de capteurs in-verger. Nous le démontrons en trois temps : (i) une **validation de méthode** en descente d'échelle de réanalyse, en protocole strict *leave-one-station-out* ; (ii) une **valeur en prévision opérationnelle**, en appliquant la même calibration à une prévision AROME à J-1 ; (iii) une **valeur décisionnelle** chiffrée par une analyse coût-perte. Le fil conducteur, empiriquement établi, est que **la supervision par capteurs in situ — non le socle atmosphérique ni la profondeur du réseau — porte l'essentiel du skill utile.** Le reste du papier présente les données et la zone d'étude (§2), la méthode (§3), le protocole de validation (§4), les résultats (§5), puis la discussion et les limites (§6-7).

## Données et zone d'étude

**Zone d'étude.** Les vergers d'abricotiers (variété Bergeron dominante) de la Drôme et de l'Ardèche, autour des Baronnies provençales — un relief de **vallées encaissées** propice au *cold-air pooling* et une sinistralité gel chronique (2017, 2019, 2021). C'est un cas d'étude exigeant : le gel radiatif y domine, à petite échelle spatiale, mal saisi par les mailles grossières.

**Réanalyses et modèles.** La réanalyse **CERRA** (5,5 km, disponible depuis 1984) est le socle canonique d'entrée ; **ERA5-Land** (9 km) sert de référence comparative pour isoler l'effet du socle (§5.2). Les champs de surface **SURFEX** (température radiative de surface `T_skin`) fournissent, avec CERRA, l'**enveloppe physique** de bornage du réseau (§3.1). Pour l'évaluation en prévision (§5.4), on utilise une prévision **AROME** archivée à J-1 (Open-Meteo Historical Forecast, ~2,5 km). Réanalyses CERRA et ERA5-Land sont librement accessibles via le CDS Copernicus.

**Réseau de capteurs (vérité terrain).** Le réseau **Sencrop** de capteurs in-verger de la zone fournit la température, échantillonnée au pas **sub-horaire (~15 min)** ; la température minimale nocturne agrégée est notre vérité terrain. Après contrôle qualité, le jeu couvre **~12 500 nuits-station sur 2022-2025**. 

**Limite de mesure à garder en tête.** La Tmin Sencrop n'est pas une vérité parfaite : le capteur a une **constante de réponse thermique (~20 min)** qui, lors des refroidissements radiatifs rapides — le scénario même du gel sévère — le fait **lire plus chaud que la réalité** (§3.2, §6). Cette vérité terrain porte donc un **biais chaud connu**, corrélé au taux de refroidissement, à corriger (déconvolution) et à garder à l'esprit dans l'interprétation des scores.

**Seuil d'événement.** L'événement « gel » est défini au seuil **agronomique −2,2 °C** (approximativement la température létale LT10 au stade floraison de l'abricot, d'après Proebsting & Mills), seuil auquel toutes les métriques (§4) sont calculées.

## 3. Méthodes

### 3.1 Réseau de descente d'échelle et bornage physique

**Architecture résiduelle.** Le cœur de la chaîne est un U-Net convolutif opérant en **résiduel** : plutôt que de prédire directement la température minimale nocturne, le réseau apprend une **correction** `δ` ajoutée à un *first-guess* physique, soit `T̂min = T_fg + δ`. Le *first-guess* est le champ de réanalyse cible descendu bilinéairement à 1 km, éventuellement corrigé d'un gradient adiabatique de maille `Γ·(z_MNT − z_orographie_coarse)` (variante *lapse*, `Γ = −4 °C·km⁻¹`). Ce choix est déterminant : sans lui, le réseau régresse vers la climatologie douce et le pouvoir de détection du gel s'effondre (POD → 0). Le plancher garanti par le *first-guess* équivaut à la calibration statistique de référence (§3.2) ; le biais de grande échelle est porté par le *first-guess*, le réseau ne façonnant que la structure fine. La correction lapse de **maille** (1 km vs orographie coarse) est distincte de la correction station↔maille (`Γ·dz_obs`) appliquée une seule fois dans la perte — les deux `dz` diffèrent, sans double-comptage.

**Conditionnement par le relief (FiLM).** À chaque niveau de l'encodeur, les cartes de caractéristiques météo `x` sont modulées de façon affine par une couche FiLM, `FiLM(x) = γ·x + β`. Le couple `(γ, β)` — **un par canal** — est produit par un petit perceptron à partir d'un **résumé global du relief** : le MNT est réduit par moyennage spatial global (`AdaptiveAvgPool2d(1)`) en un vecteur, optionnellement concaténé à un vecteur de conditionnement de grande échelle (p. ex. descripteurs de régime synoptique), puis projeté en `(γ, β)`. La couche est initialisée à l'identité (`γ = 1, β = 0`). **Portée du conditionnement — assumée explicitement** : le MNT étant globalement moyenné, cette modulation est **globale par canal** (le même `(γ, β)` s'applique à tous les pixels), et non une correction par pixel dépendant de l'altitude locale. Le conditionnement ajuste donc la réponse du réseau au **contexte topographique d'ensemble** du domaine, pas la structure fine intra-domaine. Nous ne la sur-attribuons pas au réseau : la structure fine (cuvettes, fonds froids radiatifs) est portée par l'assimilation in situ (§3.2), et une variante FiLM **spatiale** (`γ, β` sous forme de cartes) fait l'objet de l'ablation renvoyée à l'étude compagnon (§6).

**Supervision de la queue froide (perte pinball).** La perte totale combine un terme d'attache aux stations et une régularisation, `L = L_obs + λ L_TV`. Le terme d'attache `L_obs` compare la prédiction — ramenée à l'altitude réelle du capteur par la correction adiabatique station↔maille `Γ·dz_obs` — à la Tmin observée. Plutôt qu'une erreur quadratique, on emploie une **perte pinball** (quantile) : pour un résidu `r = T_obs − T̂`, `L_obs = moyenne[ q·r si r ≥ 0 ; (q−1)·r sinon ]`, avec `q = 0,10`. Un résidu `r > 0` (prédiction trop **froide** → risque de fausse alerte) est pondéré `q = 0,1` ; un résidu `r < 0` (prédiction trop **chaude** → **gel manqué**) est pondéré `|q−1| = 0,9`. La perte pénalise donc un gel manqué **neuf fois plus** qu'une fausse alerte, biaisant volontairement les prédictions vers le froid (le quantile 10 %) et privilégiant la détection (POD) — là où une perte MSE symétrique régresse vers la moyenne douce et effondre la détection des extrêmes. Le terme `L_TV` est une **régularisation de variation totale** (`moyenne|∂_x T̂| + moyenne|∂_y T̂|`) imposant la cohérence spatiale du champ reconstruit.

**Bornage physique du prior (clamp).** La tête de sortie borne la prédiction du réseau dans une **enveloppe physique** construite à partir des champs SURFEX et CERRA. Soit `E = {E_k}` la pile des champs d'enveloppe (température de surface radiative SURFEX `T_skin`, température CERRA), `lo = min_k E_k`, `hi = max_k E_k` ; la sortie est `T̂ = c + a·tanh(r)` avec `c = (lo+hi)/2` et `a = (hi−lo)/2 + m` (marge `m`). Elle appartient donc **par construction** à `[lo − m, hi + m]`, tout en restant différentiable, pour un skill quasi inchangé. Le clamp garantit ainsi que la **composante apprise** de l'indice — le *prior* modèle — ne produit **jamais** de valeur non physique. Il ne s'applique **pas** à la correction observationnelle (§3.2), et ce choix est délibéré (voir la décomposition auditable ci-dessous).

### 3.2 Assimilation des observations Sencrop par fonction de base radiale (RBF)

La correction qui porte l'essentiel du skill est une **assimilation** des résidus de stations Sencrop sur le champ descendu, apparentée à une interpolation optimale simplifiée. Pour chaque nuit et chaque maille servie `S`, on calcule un résidu par station donneuse `j`, `ρ_j = T_obs,j − T̂_pré-RBF(cellule_j)`, où `T̂_pré-RBF` est le champ **avant assimilation** (sortie du réseau, §3.1). Ces résidus sont interpolés sur `S` par une **pondération gaussienne** de la distance entre la maille de `S` et la station `j`, `w_j = exp(−d²_{S,j} / 2σ²)`, avec une longueur de corrélation `σ = 7 km`. La correction n'est appliquée que sous garde opérationnelle : au moins **5 stations présentes** cette nuit **et** au moins **3 donneurs valides** ; à défaut, on sert le champ non corrigé.

**Validation hors-échantillon (anti-fuite).** La station évaluée est retirée de l'ensemble des donneurs (*leave-one-station-out*). Comme deux stations très proches portent un résidu quasi identique, on regroupe en outre les stations distantes de moins de `cluster_km` en grappes (union-find) et l'on retire la **grappe entière** (*leave-one-cluster-out*) — mode le plus défendable pour estimer la performance sur une parcelle **sans capteur**. Tous les scores rapportés (§5) proviennent de ce protocole ; les scores in-sample (station présente dans l'assimilation), plus optimistes (FAR ≈ 0,19 vs ≈ 0,45 sur une même année), sont explicitement écartés.

**Articulation avec SURFEX.** SURFEX n'entre pas comme observation mais comme **contrainte physique** : ses champs de température de surface fournissent, avec CERRA, l'enveloppe de bornage du réseau (§3.1). L'assimilation Sencrop (RBF) apporte, elle, la **correction de biais locale** — fond de vallée, exposition, cold-air pooling — que ni la réanalyse ni SURFEX ne résolvent à la parcelle. C'est cette **assimilation in situ**, et non le modèle atmosphérique ou le réseau seul, qui porte l'essentiel du skill (§5).

**Décomposition auditable de l'indice servi.** L'indice final est la somme d'un *prior* modèle **borné** (§3.1) et d'une correction observationnelle : `T̂_servi = clamp(T̂_DL) + Σ_j w_j ρ_j / Σ_j w_j`. La correction **peut** faire sortir l'indice de l'enveloppe physique du prior — et c'est **voulu** : elle est tirée des observations, qui constatent des froids que le modèle physique n'encadre pas (cuvettes radiatives). L'auditabilité de l'indice ne repose donc **pas** sur son appartenance à une enveloppe, mais sur sa **décomposition** : une composante modèle bornée + une composante observationnelle **transparente et reproductible** — moyenne pondérée de résidus de stations, poids `w_j = exp(−d²/2σ²)` dépendant **uniquement de la géométrie stations↔maille**. Pour tout indice servi, on peut donc dire exactement ce qui vient du modèle (borné) et ce qui vient de **quelles stations, avec quels poids** ; la part qui dépasse la physique **est la donnée elle-même**. Le garde-fou d'assurabilité n'est alors pas un bornage physique de l'indice, mais le **contrôle qualité** des observations et l'**agrégation robuste** (≥ 3 donneurs, pondération par distance, exclusion de grappe) : une station isolée aberrante est **diluée**, non propagée.

### 3.4 Deux flux de température (à ne pas mélanger)
- **Prévision/skill gel** : Tmin (T_skin radiatif CERRA / prévision AROME).
- *(Couplage phénologique stade-aware : retiré — feature applicative, #232.)*

## 4. Protocole de validation

- **Juge** : **CSI au seuil agronomique −2,2 °C**, en **leave-one-station-out** (station évaluée jamais dans le calage).
- **Aucune fuite** : les chiffres in-sample sont écartés (ils font passer le FAR de ~0,45 à ~0,19 sur une même année).
- **Deux régimes de validation** :
  1. **Hindcast** (réanalyse CERRA descendue) — 2022-2025, ~12 500 nuits-station, vergers d'abricotiers Drôme/vallée du Rhône : valide la **méthode**.
  2. **Prévision** (AROME archivé J-1, Open-Meteo, 2024-2025, 48 stations) : valide la **valeur opérationnelle** (le cœur du papier), avec **split temporel** train-une-saison → prévoir-l'autre.
- **Métriques** : POD/FAR/CSI par an et *pooled* ; **REV** ; stratification régime/altitude.
- **Reproductibilité** : graine globale + exécution déterministe consignées.
- 2024 = année quasi sans gel (à ne pas moyenner naïvement).

## 5. Résultats

### 5.1 Validation de méthode : skill hindcast

Nous validons d'abord la chaîne en *hindcast* (réanalyse CERRA descendue), en **leave-one-station-out** au seuil agronomique −2,2 °C sur 2022-2025 (~12 500 nuits-station). Les scores sont **micro-agrégés** — contingences (VP/FP/FN) sommées sur les quatre saisons — ce qui donne un poids négligeable à 2024, année quasi sans gel, et évite l'artefact d'une moyenne annuelle tirée vers le bas par une saison dégénérée (Fig. 2).

| Méthode | POD | FAR | CSI |
|---|---|---|---|
| CERRA brute (5,5 km) | — | — | 0,17 |
| Calibration statistique (Lot B) | 0,36 | 0,48 | 0,27 |
| DL + supervision station (Lot C) | 0,48 | 0,36 | 0,38 |

Le CSI progresse à chaque étage de la chaîne, de **0,17** (réanalyse brute) à **0,27** (calibration statistique) puis **0,38** (réseau supervisé par les stations) ; le Lot C améliore **simultanément** la détection (POD 0,36 → 0,48) et le taux de fausses alertes (FAR 0,48 → 0,36). Le skill est homogène entre saisons à événements (CSI Lot C : 0,38 en 2022, 0,42 en 2023, 0,33 en 2025 ; 2024, sans gel, n'est pas informative). Le **saut décisif provient de la supervision par capteurs** : l'apport du socle atmosphérique et de la profondeur du réseau est secondaire devant celui de l'assimilation in situ (§3.2).

### 5.2 Le socle de réanalyse prime sur la profondeur du réseau

À calibration égale, la réanalyse d'entrée pèse davantage que la complexité du réseau : CERRA 5,5 km surpasse ERA5-Land 9 km avant même toute correction apprise, ce qui oriente le choix d'architecture vers le **meilleur socle** plutôt que vers un réseau plus profond. *(Chiffres à recomputer en LOO — l'estimation actuelle est in-sample, #233 ; la conclusion qualitative est attendue robuste.)*

### 5.3 Où le skill se concentre : le régime radiatif

Stratifiés par régime synoptique, les événements de gel se concentrent en régime **radiatif** (nuit claire, calme, cuvette) — environ 46 % du total — et c'est précisément là que la chaîne est la plus skillée (CSI ≈ 0,21 contre ≈ 0,05 en régimes ventés/cycloniques, soit un facteur ~4). C'est physiquement cohérent : le gel radiatif de cuvette est ce que la maille grossière résout le plus mal et ce que les capteurs in situ observent le mieux. *(Stratification actuellement in-sample ; à refaire hors-station, #234.)*

### 5.4 Résultat-phare : valeur en prévision opérationnelle (AROME)

Le hindcast valide la méthode ; la prévision en mesure la valeur. Nous évaluons une prévision **AROME archivée à J-1** (Open-Meteo, 48 stations de la Drôme, 2024-2025), au même seuil −2,2 °C et hors-échantillon (Fig. 4). Brute, la prévision est **quasi aveugle au gel de parcelle** : POD 0,05, CSI 0,05 — le pas de 2,5 km lisse la queue froide et manque le décrochage radiatif local. La **calibration par les stations Sencrop** — un mapping de quantiles par station, appliqué **sans recours à l'observation de la nuit prévue** — porte la détection à **POD 0,50** (× 10) et le CSI à **0,32** (× 6). Surtout, cette calibration **se transfère dans le temps** : apprise sur 2024 et appliquée à 2025 (cadre strictement opératoire, aucune fuite), elle conserve un CSI de **0,34**. Le levier de la valeur n'est donc pas le modèle atmosphérique mais le **réseau de capteurs in situ**, qui rend exploitable une prévision qui ne l'était pas.

*Note de mesure* : la vérité terrain Sencrop est elle-même biaisée **chaud** sur les refroidissements rapides (retard de réponse thermique ~20 min, §6) ; les scores de détection ci-dessus sont donc **conservateurs** — une déconvolution de ce retard devrait les relever sur les gels les plus sévères.

### 5.5 Valeur décisionnelle : analyse coût-perte (REV)

Un skill ne vaut que par la décision qu'il améliore. Nous quantifions la **valeur économique relative** (REV ; Richardson, 2000 ; Wilks) : un décideur protège (coût `C`) ou non (perte `L` en cas de gel), et `V(α)` mesure, sur le ratio coût-perte `α = C/L`, la fraction de la valeur d'une prévision parfaite que capte la chaîne (Fig. 5). L'implémentation est vérifiée par son **ancrage théorique** : au point `α = s` (base rate), `V` égale le score de Peirce (POD − POFD), avec correspondance numérique exacte. Sur la plage coût-perte **réelle** des exploitants (`α ≈ 0,02-0,10`, dérivée de coûts de protection de 0,6 à 2,5 k€·ha⁻¹·nuit⁻¹ et de valeurs de récolte de 15 à 40 k€·ha⁻¹), la chaîne capture **80 à 85 %** de la valeur d'une prévision parfaite (maximum `V = 0,85` à `α ≈ 0,07`), soit **~460 à 1 960 €·ha⁻¹·an⁻¹** de pertes évitées. Ce résultat éclaire le CSI « modeste » de 0,38 : parce que le ratio coût-perte du gel tombe **sur** le base rate, la chaîne opère dans son **régime de valeur maximale**. C'est l'argument que ni le RMSE ni la CRPS ne fournissent.

### Figures et tables

Fig. 2 (CSI par étage, LOO), Fig. 3 (biais résiduel par station avant/après, −45 %), Fig. 4 (AROME brut vs calibré — résultat-phare), Fig. 5 (REV `V(α)` et zone de valeur) sont produites (`scripts/make_gmd_figures.py`, colorblind-safe, PNG + PDF). Restent Fig. 1 (zone d'étude + stations + MNT) et les tables de stratification en LOO (#234).

## 6. Discussion
- **Positionnement prévision** : les systèmes d'alerte gel opèrent à la maille du modèle (AROME 1,3-2,5 km) ; ils manquent le gel radiatif de cuvette. Nous montrons qu'une **couche de calibration in situ** — et non un meilleur modèle atmosphérique — franchit ce dernier kilomètre.
- **La calibration in situ est le composant porteur** : le moat est le **couplage prévision × réseau capteur**, pas le réseau de neurones seul.
- **Applications aval** (dont l'assurance) : une prévision skillée à la parcelle alimente aussi bien l'**alerte agronomique** (protection active) qu'un **indice paramétrique** à faible risque de base — mais l'objet du papier est la **valeur de prévision**, l'assurance n'en est qu'un débouché.
- **vs Airbus IPP** : remote sensing sans métriques de discrimination publiées ; nos POD/FAR/CSI **honnêtes en LOO** comblent ce vide, sans dépendance nuageuse, historique long.
- **Limites honnêtes** : (i) CSI modeste (0,38) ; (ii) densité capteurs (POD plafonne < 5 nuits/station) ; (iii) **retard de réponse thermique des capteurs (~20 min)** → biais chaud de la vérité terrain sur les refroidissements radiatifs rapides (biais ≈ `τ·|dT/dt|`, ~1-2 °C sur les gels rapides), à corriger par déconvolution `T̂_air = T_c + τ·dT_c/dt` (#236) — c'est le principal biais de mesure du dispositif ; (iv) déterminisme d'entraînement à finaliser ; (v) AROME 2,5 km (archive Open-Meteo) → bascule 1,3 km natif à venir.
- **Future work** : (a) AROME natif 1,3 km ; (b) seuils **stade-aware** (couplage phéno, en cours applicatif) pour le risque de base temporel ; (c) **ablation FiLM·MNT** (compagnon Risser/ANITI 2027) ; (d) horizons de prévision > J-1.
- **Transférabilité** : toute culture avec réseau capteur dense + seuils par stade — cerise, pêche, **vigne**, kiwi.

## 7. Conclusion
- En prévision opérationnelle, la calibration in situ transforme une prévision AROME **aveugle** (5 %) en prévision **exploitable** (50 %, ×10), validée hors-station.
- Le skill (CSI 0,38) porte une **valeur décisionnelle élevée** (REV 80-85 %, ~460-1 960 €/ha) sur la plage coût-perte réelle.
- Le levier n'est pas le modèle atmosphérique mais le **réseau de capteurs in situ**. Pipeline open source ; poids et calibration commerciables.

## Code and data availability
- Code : `karpos-downscaling` — Apache 2.0 — DOI Zenodo via JOSS.
- Poids + calibration : licence commerciale (corresponding author).
- Sencrop : agrégées/anonymisées sur demande ; brut sous NDA. Prévision AROME : Open-Meteo Historical Forecast. Réanalyses : CDS Copernicus.

## Author contributions · Acknowledgements · References
- *Acknowledgements* : cadre EI **Karpos** ; données **Sencrop** (M. Ducroquet) ; échanges T. Dalhaus ; étude compagnon FiLM à venir avec **L. Risser (ANITI)**.
- ~30-40 réf. (Richardson 2000, Wilks *Statistical Methods*, U-Net, FiLM, CERRA/ERA5-Land, AROME, Open-Meteo…).

---

## Prérequis avant rédaction (mis à jour)

| Item | État | Action |
|---|---|---|
| Chiffres LOO honnêtes (hindcast + AROME) | ✅ persistés | figer |
| Posthoc Lot C opposable (#222) | ✅ | — |
| Figures GMD F2-F5 (CSI, biais, AROME, REV) | 🔴 #235 | 2-3 j |
| CERRA vs ERA5-Land **en LOO** | 🔴 #233 | 1-2 j |
| Régimes **hors-station** | 🔴 #234 | 2-3 j |
| Seed + `deterministic=True` | 🔴 #228 | 0,5 j |
| ~~PhenoFlex z_c~~ | ✅ retiré (feature app, #232) | — |

**Aucun verrou bloquant** : le papier est rédigeable, angle prévision.

## Calendrier (recadré — S31, 27/07/2026)

| Étape | Charge | Cible |
|---|---|---|
| Recompute LOO + seed + figures (REV, AROME) | ~6-8 j | août-sept |
| Co-auteurs (Ubbiali, Dalhaus) | 2-3 réunions | sept |
| Draft v1 (prose) | 10-12 j | sept-oct |
| Revue interne | 5 j | oct |
| Soumission GMD | 1 j | **nov 2026** |

## Liens
- [[Méthodo — Valeur économique des prévisions gel (REV + risque de base)]]
- [[Métriques trackées — Lot B vs Lot C]]
- [[Audit — Architecture downscaling gel (clamp + Sencrop-SURFEX)]]
- [[Produit - 27-07-2026]]
- [[Trame publication JOSS — parametric_insurance]]
- [[feedback-perf-framing-sap]]
