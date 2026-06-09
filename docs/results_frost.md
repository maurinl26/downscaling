# Résultats — détection du gel, éval vs Sencrop (ground truth)

Protocole : **train 2015-2021 / dev 2022-2025 / test pur 2026** (jamais touché).
Métrique : POD/FAR @ 0 °C + ROC-AUC + meilleur point de fonctionnement, sur nuits
held-out, Tmin downscalé échantillonné aux stations (correction d'altitude).
Cible métier : **FAR < 0,20 et POD > 0,75**.

## Tableau d'ablation (dev 2022-2025, 481 nuits, 16 768 obs-station)

| Configuration | POD@0°C | FAR@0°C | AUC | POD @FAR<0,20 | FAR @POD>0,75 |
|---|---|---|---|---|---|
| U-Net **absolu** (sans résiduel) | ~0 | ~0 | ~0,5 | — | — |
| Baselines calibration au point (RAW/EQM/KF, médiane) | 0,58-0,77 | 0,25-0,53 | — | jamais les deux | jamais les deux |
| **U-Net résiduel** (t2m) | 0,781 | 0,396 | **0,964** | 0,316 | 0,378 |
| U-Net résiduel + **calibration/station** (oracle, borne sup.) | — | — | 0,975 | **0,611** | 0,294 |
| *Réf. avril 2022 seul (optimiste, 1 épisode)* | — | — | 0,98 | 0,775 | 0,198 |

## Enseignements

1. **Le résiduel est le débloqueur** (`out = entrée + correction`) : passe du *mean collapse*
   (POD≈0) à POD 0,78 / AUC 0,96. Sans lui, rien ne marche (issue #7).
2. **La discrimination est déjà excellente** (AUC 0,96-0,98). Le problème n'est pas le
   pouvoir prédictif mais le **point de fonctionnement** : au seuil 0 °C le modèle
   **sur-déclare le gel** (FAR 0,40).
3. **La calibration par station (étage C) est nécessaire mais pas suffisante** : elle
   double presque le POD à FAR<0,20 (0,316 → 0,611) mais reste sous la cible 0,75.
   **En t2m seul, même une calibration parfaite ne ferme pas le gap.**
4. **Le levier décisif restant = les prédicteurs radiatifs** (vent 10 m, point de rosée,
   rayonnement/nébulosité) : ils séparent le gel radiatif des nuits froides ventées/
   nuageuses que t2m seul confond → c'est ce qui doit faire tomber le FAR sous 0,20.
5. **Avril 2022 seul est trompeur** (un épisode de gel franc) : toujours valider sur
   plusieurs saisons.

## Multi-variable [t2m, rosée, u10] → CERRA : N'AIDE PAS (résultat clé)

Éval 481 nuits 2022-2025 : POD@0°C 0,886 / FAR 0,494 / **AUC 0,963** / POD@FAR<0,20 **0,256**
(vs t2m-only 0,316) / +calibration 0,574 (vs 0,611). **Régression légère, AUC inchangé.**

**Pourquoi** : l'étage A reconstruit le **t2m de CERRA** (déjà AUC 0,96). CERRA t2m encode
déjà l'atmosphère → ajouter rosée/vent en entrée n'aide pas à mieux reconstruire CERRA ;
le modèle est saturé à la skill de CERRA. **Le plafond = CERRA vs vérité terrain, pas
l'information d'entrée.** Les prédicteurs radiatifs ne paient que là où la supervision est
la STATION (gel radiatif local que CERRA lisse) → **étage C**, pas étage A.

→ **Pivot stratégie** : (1) étage C avec entrée multi-var (calibration Sencrop + rosée/vent) ;
(2) cible plus fine (CERRA 5,5 km en entrée) ou supervision directe station.

## Décision d'architecture (pour le 15 juin)

`résiduel (acquis) + prédicteurs radiatifs multi-variables + FiLM DEM (cuvettes) +
calibration sparse Sencrop (étage C)`, point de fonctionnement calé sur dev, reporté
sur le test pur 2026. La calibration au point seule (MOS/EQM/KF) est écartée :
n'atteint jamais les deux cibles et ne scale pas avec la densité (cf. roadmap §7).

## Prochaines briques (par leverage attendu sur la cible)

1. **Multi-variable** (découpler in/out channels : entrée [t2m, td, u10, v10] → sortie t2m)
   — le plus gros levier FAR.
2. **Étage C** calibration Sencrop (train dev, test 2026) — confirme le gain ~+0,3 POD réel.
3. **DEM cuvettes** (TPI/sky-view) en FiLM.
4. **CERRA** 5,5 km en cible/entrée (dès DL 2022-2026).
