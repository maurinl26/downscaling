---
title: "Modélisation du froid hivernal et du forçage printanier pour l'abricotier des Baronnies provençales — couplage chilling × forcing et croisement avec le risque de gel"
aliases:
  - rapport chilling abricot Baronnies
  - chilling Baronnies
tags:
  - karpos/recherche
  - karpos/phenologie
  - karpos/gel
  - phenoflex
created: 2026-06-10
status: draft-v1
project: "[[AgroFrost-CEI]]"
related:
  - "[[PhenoFlex — Couplage Chilling × Forcing (abricot)]]"
  - "[[Phénologie - Indices paramétriques gel]]"
  - "[[2026-05-22 Réunion Tobias Dalhaus]]"
---

# Modélisation du froid hivernal et du forçage printanier pour l'abricotier des Baronnies provençales

*Rapport technique — Karpos — v1, 10 juin 2026*

> [!abstract] Résumé
> Nous décrivons et appliquons un modèle phénologique à couplage **froid → chaleur** (PhenoFlex, Luedeling et al. 2021) pour prédire les stades de développement floral de l'abricotier (*Prunus armeniaca* L., cv. **Bergeron**) dans les **Baronnies provençales** (Drôme/Nyonsais), et le croiser avec le risque de gel printanier. Le modèle enchaîne (i) une accumulation de froid par le **modèle Dynamique** (Chill Portions, CP), (ii) une accumulation de chaleur par **Growing Degree Hours** (GDH, Anderson 1986), reliées par une transition sigmoïde, puis (iii) un **double seuil** : *seuil de maturation* (le bourgeon a-t-il atteint un stade vulnérable ?) et *seuil de gel* (T_min < seuil critique du stade). L'implémentation Python est validée par 14 tests d'adéquation au modèle de référence chillR. Les seuils de **vulnérabilité au gel par stade** (T10/T90) sont issus de sources vérifiées (FAO/Proebsting & Mills) ; le **besoin en froid** de Bergeron (≈ **62–65 CP**) est documenté dans la littérature ; les **besoins en chaleur** (GDH par stade) restent **à caler** sur données phénologiques observées — en l'état, les dates de floraison simulées (mi-avril) accusent un retard de ~2–4 semaines sur l'observation (début mars), ce qui circonscrit l'usage actuel à la méthodologie et non à la prédiction opérationnelle des dates.

---

## 1. Introduction

L'abricotier est l'espèce arboricole emblématique et économiquement dominante des Baronnies provençales. Sa **floraison précoce** (historiquement mi-mars, désormais début mars) l'expose de façon chronique aux **gelées tardives** : les producteurs locaux rapportent *« sept ans sans récolte normale »*, avec des vagues de froid « noir » d'origine polaire survenues trois fois en cinq ans là où elles étaient vicennales (France Bleu Drôme, 2024 ; PNR des Baronnies provençales, 2023).

Deux risques climatiques distincts coexistent :

1. **Gel printanier** sur organes floraux — risque aigu, dépendant du **stade phénologique** : un même −3 °C est anodin en dormance et destructeur en pleine floraison.
2. **Déficit de froid hivernal** — risque chronique émergent : Bergeron, variété exigeante, *« manque de froid pour différencier de bons bourgeons »* (PNR Baronnies), condition jugée « mal satisfaite à partir de 2050 ».

La **vulnérabilité au gel étant fonction du stade**, prédire correctement le **calendrier phénologique** est le prérequis de tout indice paramétrique de gel robuste — c'est le levier de réduction du **risque de base temporel** identifié par Dalhaus et al. (2018) et discuté avec T. Dalhaus (Wageningen) le 22/05/2026 ([[2026-05-22 Réunion Tobias Dalhaus|CR]]).

Ce rapport documente la brique de modélisation correspondante et l'état de **vérification de ses seuils**.

---

## 2. Zone d'étude

| Caractéristique | Valeur |
|---|---|
| Territoire | Baronnies provençales (Drôme — Nyonsais, Buis-les-Baronnies) |
| Espèce / cultivar cible | *Prunus armeniaca* L., cv. **Bergeron** (groupe **tardif**) |
| Altitude vergers | ~250–600 m |
| Régime de gel dominant | Radiatif nocturne (nuits claires, vallées encaissées) + advectif (« froid noir » polaire) |
| Fenêtre de risque | Débourrement → nouaison (≈ fév.–mai) |

Bergeron relève du **groupe de précocité tardif** ; les variétés précoces (type Orangered) fleurissent jusqu'à ~4 semaines plus tôt et requièrent une calibration distincte (Andreini et al. 2014 ; Audergon 1993).

---

## 3. Matériel et méthodes

### 3.1 Données de température

| Usage | Variable | Source cible | Rôle |
|---|---|---|---|
| Phénologie (froid + chaleur) | **T air horaire** | CERRA 5,5 km **descente d'échelle** calibrée **Sencrop** | Les modèles de froid/chaleur sont calibrés sur T air |
| Déclenchement gel | **T_skin** (T de surface radiative) | CERRA-Land | Capte le refroidissement radiatif nocturne (avantage PoC : POD 0 °C 49 %→95 %) |

> [!warning] Deux variables de température distinctes
> La phénologie consomme la **T air** ; le déclenchement du gel la **T_skin radiative**. Les deux flux ne doivent pas être confondus (cf. note d'architecture, décision D5).

### 3.2 Modèle de froid — Chill Portions (modèle Dynamique)

Le modèle **Dynamique** (Fishman, Erez & Couvillon 1987 ; paramétrisation chillR de Luedeling) accumule le froid en **Chill Portions** (CP), unité physiologique irréversible. À chaque heure ($T_K = T_{°C}+273$) :

$$ x_s = \frac{a_0}{a_1}\,e^{(e_1-e_0)/T_K}, \quad k_1 = a_1\,e^{-e_1/T_K}, \quad \xi = \frac{e^{f}}{1+e^{f}},\ f=\text{slp}\cdot t_{etmlt}\frac{T_K-t_{etmlt}}{T_K} $$

Le précurseur $E$ relaxe vers $x_s$ au taux $k_1$ ; dès que $E \ge 1$, une fraction $\xi$ est convertie en CP (irréversible) et $E$ est consommé d'autant. Paramètres : $e_0{=}4153{,}5$ ; $e_1{=}12888{,}8$ ; $a_0{=}139500$ ; $a_1{=}2{,}567\times10^{18}$ ; $\text{slp}{=}1{,}6$ ; $t_{etmlt}{=}277$.

Propriété clé (vérifiée, §3.6) : **réponse en température en cloche, optimum ~6 °C**, accumulation nulle aux températures chaudes ($x_s<1$ ⇒ pas de conversion) comme très froides ($\xi\to0$), avec **réversibilité partielle** (les après-midi doux annulent une partie du froid nocturne).

### 3.3 Modèle de forçage — Growing Degree Hours (Anderson 1986)

Fonction de chaleur sigmoïde à trois paramètres ($T_b$ base, $T_u$ optimum, $T_c$ critique) :

$$ \text{GDH}(T)=\begin{cases} 0 & T\le T_b \text{ ou } T\ge T_c\\[2pt] \frac{T_u-T_b}{2}\!\left[1+\cos\!\big(\pi+\pi\tfrac{T-T_b}{T_u-T_b}\big)\right] & T_b< T\le T_u\\[2pt] (T_u-T_b)\!\left[1+\cos\!\big(\tfrac{\pi}{2}+\tfrac{\pi}{2}\tfrac{T-T_u}{T_c-T_u}\big)\right] & T_u< T< T_c \end{cases} $$

### 3.4 Couplage PhenoFlex (chilling × forcing)

Le forçage thermique n'est comptabilisé qu'une fois le froid (quasi) satisfait, via une **transition sigmoïde** du froid cumulé $C(t)$ autour du besoin critique $y_c$ :

$$ \text{transition}(t)=\frac{1}{1+e^{-s_1\,(C(t)-y_c)}}, \qquad H(t)=\sum_{\tau\le t}\text{GDH}(\tau)\cdot\text{transition}(\tau) $$

La **date d'un stade** est le premier $t$ tel que $H(t)\ge z_c^{\text{stade}}$ (un besoin en chaleur $z_c$ par stade). Ceci remplace le proxy historique « GDD depuis le 1ᵉʳ janvier », dont la date de départ arbitraire biaise le stade prédit en hiver doux.

### 3.5 Croisement gel × stade — double seuil

Pour chaque nuit de la fenêtre de risque :

1. **Seuil de maturation** : le bourgeon a-t-il franchi le 1ᵉʳ stade vulnérable (gonflement) à cette date ? Sinon → pas de déclenchement (dormance peu sensible).
2. **Seuil de gel** : $T_{min} < T_{10}/T_{90}$ du **stade actif**.

Le paiement n'est positif que si **les deux** conditions sont réunies — c'est l'apport demandé : *« un seuil de maturation en plus du seuil de gel »*.

### 3.6 Implémentation et vérification

Code : `parametric_insurance/backtest/src/indices/{phenoflex.py, phenology.py}`. Forward **100 % Python** (aucune dépendance R en production) ; le package R **chillR** est réservé au **calage offline** des paramètres. **14 tests d'adéquation** au modèle de Luedeling (`tests/test_phenoflex.py`) : valeurs analytiques exactes du GDH d'Anderson, courbe de réponse des CP (optimum ~6 °C), réversibilité partielle, gate de maturation, intégration multi-années, et comparaison numérique directe à `chillR::Dynamic_Model`/`GDH` (activée en CI). **14/14 au vert.**

> Cette étude a révélé et corrigé un défaut de l'implémentation antérieure du modèle Dynamique (accumulation non discriminante en température — issue #3, corrigée).

---

## 4. Paramètres et seuils — statut de vérification

> [!important] Légende du statut
> 🟢 **Vérifié** (source primaire) · 🟡 **Littérature** (valeur publiée, à confirmer localement) · 🔴 **À caler** (placeholder, calage requis).

### 4.1 Besoin en froid (levée de dormance), $y_c$

| Paramètre | Valeur | Source | Statut |
|---|---|---|---|
| Bergeron — besoin en froid | **62–65 Chill Portions** | UC ANR Fruit&Nut chill requirements ; cohérent Fadón et al. 2021 | 🟡 |
| Bergeron — équiv. heures < 7 °C | **~800 h** | PNR Baronnies provençales (2023) | 🟡 |
| Valeur retenue dans la config | $y_c = 45$ CP | placeholder prudent | 🔴 → relever vers ~63 CP au calage |

### 4.2 Besoins en chaleur par stade (GDH), $z_c$

| Stade (BBCH) | $z_c$ config (GDH) | Statut |
|---|---|---|
| Gonflement (51) | 2500 | 🔴 |
| Sépales (53) | 4000 | 🔴 |
| Bouton rose (55) | 5500 | 🔴 |
| Floraison (60–65) | 7000 | 🔴 |
| Nouaison (69–71) | 9500 | 🔴 |

Ordre de grandeur cohérent avec les besoins en chaleur publiés pour l'abricot (plusieurs milliers de GDH jusqu'à floraison ; Ruiz, Campoy & Egea 2007), mais **non calés** : ils produisent une floraison simulée trop tardive (§5).

### 4.3 Seuils de vulnérabilité au gel par stade (T10/T90) — *vérifiés*

Températures critiques détruisant 10 % (T10, seuil d'activation) et 90 % (T90, paiement maximal) des organes floraux ; tables Proebsting & Mills (FAO 2005), recoupées Agrobiotop.

| Stade Baggiolini / BBCH | T10 (°C) | T90 (°C) | Statut |
|---|---|---|---|
| A / 50 — bourgeon dormant | −17 | −25 | 🟢 |
| B / 51 — gonflement | −8 | −12 | 🟢 |
| C / 53 — sépales | −5 | −7 | 🟢 |
| D / 55 — bouton rose | −3 | −5 | 🟢 |
| E–F / 60–65 — floraison | −2,2 | −4 | 🟢 |
| G / 67 — chute pétales | −0,8 | −2 | 🟢 |
| H–I / 69–71 — nouaison | −0,5 | −1,5 | 🟢 |

---

## 5. Résultats préliminaires

Calculs réalisés avec l'implémentation Karpos sur une **climatologie horaire synthétique** type Baronnies (moyennes mensuelles Nyonsais + cycle diurne ±4 °C ; saison sept. 2020 → juin 2021), à climat actuel et sous un réchauffement uniforme +2 °C (~2050).

| Scénario | CP accumulés (nov–fév) | Besoin Bergeron (63 CP) | Déficit de froid | Levée de dormance | Floraison simulée |
|---|---|---|---|---|---|
| Actuel | **89 CP** | satisfait | 0 | 12 déc. | 13 avr. |
| +2 °C (~2050) | **88 CP** | satisfait | 0 | 29 déc. | 27 mars |

Lectures :

- **Froid** : sous le modèle Dynamique, l'hiver des Baronnies **satisfait encore largement** le besoin de Bergeron (≈ 63 CP), **y compris à +2 °C** (88 CP). Ce résultat nuance le discours « heures < 7 °C » (alarmiste dès 2050) : le métrique CP (Luedeling), plus robuste en hiver doux, déplace l'échéance du risque de déficit. Le **choix du modèle de froid change matériellement le diagnostic** — point méthodologique central.
- **Avancement de la floraison** : +2 °C **avance** la floraison simulée (27 mars vs 13 avr.) → fenêtre de gel décalée vers une période encore exposée aux descentes froides, cohérent avec l'observation locale (floraison désormais début mars).

> [!caution] Limite majeure — dates non calées
> La floraison simulée (mi-avril à climat actuel) accuse **~2–4 semaines de retard** sur l'observation Baronnies (début mars). Cause : les $z_c$ (GDH par stade) sont des **placeholders** (🔴). **Les dates ci-dessus illustrent la mécanique du modèle, pas une prédiction validée.** Le calage (§6) est le verrou avant tout usage opérationnel.

---

## 6. Discussion

- **Calage prioritaire des besoins en chaleur** ($z_c$) et ajustement de $y_c$ vers ~63 CP, par recuit simulé (`chillR::phenologyFitter`) sur dates de floraison observées : **PHENOCLIM AgroClim INRAE**, **DIVAE Gotheron/Toulenne** (cf. [[Phénologie - Indices paramétriques gel]] §Sources), idéalement SEFRA (station d'expérimentation Drôme) pour Bergeron local.
- **Modèle de froid décisif** : CP (Dynamique) vs Chilling Hours donnent des diagnostics de déficit divergents. Le **CP est recommandé** (transférable entre scénarios climatiques ; Luedeling) et adopté comme référence unique (décision D4).
- **Groupes de précocité** : trois jeux de paramètres (précoce/intermédiaire/tardif) sont nécessaires ; un modèle unique à l'échelle de l'espèce ne converge pas (Andreini et al. 2014).
- **Données** : remplacer la climatologie synthétique par la **CERRA descente d'échelle calibrée Sencrop** (T air horaire) pour la phénologie, et **T_skin** pour le gel ; valider les dates de stades contre les observations de terrain.
- **Risque de déficit de froid (IND-05)** : désormais calculable correctement (modèle corrigé) ; à instruire spécifiquement sur Bergeron sous trajectoires +2/+4 °C avec effet des **after-midi doux** (réversibilité), que le métrique « heures de froid » ignore.

---

## 7. Conclusion

La brique de couplage **froid × chaleur** est implémentée, vérifiée contre le modèle de Luedeling, et croise correctement **stade de maturation** et **seuil de gel** pour l'abricotier. Les **seuils de gel par stade sont vérifiés** (FAO/Proebsting & Mills) ; le **besoin en froid de Bergeron est documenté** (≈ 62–65 CP) ; les **besoins en chaleur restent à caler**, ce qui interdit pour l'instant une lecture opérationnelle des dates de floraison. Prochain jalon : **calage chillR sur observations PHENOCLIM/DIVAE/SEFRA** puis rejeu sur séries CERRA-Sencrop réelles, condition de bascule du démonstrateur vers un produit.

---

## Références

- Anderson J.L., Richardson E.A., Kesner C.D. (1986). *Validation of chill unit and flower bud phenology models for 'Montmorency' sour cherry.* Acta Horticulturae, 184, 71–78.
- Andreini L., García de Cortázar-Atauri I., Chuine I., et al. (2014). *Understanding dormancy release in apricot flower buds using several process-based phenological models.* Agricultural and Forest Meteorology, 184, 210–219.
- Audergon J.-M. (1993). *Recherches sur la biologie de l'abricotier.* Thèse, INRA Avignon.
- Dalhaus T., Musshoff O., Finger R. (2018). *Phenology Information Contributes to Reduce Temporal Basis Risk in Agricultural Weather Index Insurance.* Scientific Reports, 8, 46. DOI : 10.1038/s41598-017-18656-5.
- Fadón E., Fernández E., Behn H., Luedeling E. (2021). *Reducing the uncertainty on chilling requirements for endodormancy breaking of temperate fruits by data-based parameter estimation of the dynamic model: a test case in apricot.* Tree Physiology, 41(4), 644–656. DOI : 10.1093/treephys/tpaa164.
- FAO (2005). *Frost protection: fundamentals, practice and economics, Vol. 1.* Tables Proebsting & Mills.
- Fishman S., Erez A., Couvillon G.A. (1987). *The temperature dependence of dormancy breaking in plants: mathematical analysis of a two-step model involving a cooperative transition.* Journal of Theoretical Biology, 124(4), 473–483.
- Luedeling E., Schiffers K., Fohrmann T., Urbach C. (2021). *PhenoFlex – an integrated model to predict spring phenology in temperate fruit trees.* Agricultural and Forest Meteorology, 307, 108491. DOI : 10.1016/j.agrformet.2021.108491. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S016819232100174X)
- Luedeling E. (2018+). *chillR — Statistical Methods for Phenology Analysis in Temperate Fruit Trees.* CRAN. [package](https://cran.r-project.org/package=chillR) · [vignette PhenoFlex](https://cran.r-project.org/web/packages/chillR/vignettes/PhenoFlex.html)
- Ruiz D., Campoy J.A., Egea J. (2007). *Chilling and heat requirements of apricot cultivars for flowering.* Environmental and Experimental Botany, 61(3), 254–263. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0098847207000883)
- UC ANR — Fruit & Nut Research and Information Center. *Crop Chill Portions Requirements.* [lien](https://ucanr.edu/sites/fruitandnut/Weather_Services/chilling_accumulation_models/CropChillReq)
- Parc naturel régional des Baronnies provençales (2023). *L'abricot à l'heure du changement climatique.* [article](https://www.baronnies-provencales.fr/actualite/labricot-a-lheure-du-changement-climatique/)
- France Bleu Drôme-Ardèche (2024). *Abricot des Baronnies : « ça fait sept ans qu'on n'a pas eu une récolte normale ».*

---

## Annexe — Reproductibilité

- Dépôt : `parametric_insurance` — commits `f4896812` (PhenoFlex) et `4073cc81` (correction IND-05, closes #3).
- Modèle : `backtest/src/indices/phenoflex.py` (`predict_stage_timeline`, `evaluate_frost_night`, `compute_phenoflex_triggers`).
- Tests d'adéquation : `backtest/tests/test_phenoflex.py` (14/14).
- Config abricot (paramètres + seuils) : `backtest/config_drome_ardeche.yaml` → `phenology.abricotier`.
- Note d'architecture : [[PhenoFlex — Couplage Chilling × Forcing (abricot)]].
