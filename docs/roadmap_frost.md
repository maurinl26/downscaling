# Roadmap — descente d'échelle du gel comme problème de queue froide

> **Reframing.** Le gel n'est pas un problème de champ moyen mais de **queue froide
> conditionnelle** : les minima rares/localisés des cuvettes en régime radiatif. La
> métrique métier (POD/FAR) vit *entièrement* dans la queue. Optimiser la RMSE globale
> est quasi-orthogonal à l'objectif (un modèle peut avoir une RMSE excellente ET un POD
> nul — le *mean collapse* observé, cf. issue #7).
>
> **Stratégie : ancrer le champ (résiduel) puis aiguiser la queue** (objectif tail-aware
> + prédicteurs physiques + terrain de cuvette). Cible : **FAR < 20 %, POD > 75 %**.

---

## Phase 0 — Recadrer objectif & métriques (immédiat)
- Headline = **POD / FAR / CSI**, biais gel, loss quantile queue froide. RMSE = diagnostic secondaire.
- **ROC / sweep de seuil** sur les baselines (RAW/EQM/KF) + corr 0,97 → connaître le
  *plafond de discrimination* à données constantes et le meilleur point de fonctionnement.
- W&B logge déjà val/pod, val/far (PR #10). Ajouter CSI + biais gel.

## Phase 1 — Architecture : ancrage + tail-aware
- **Résiduel global** : `out = upsample(coarse_t2m) + head(x)` → supprime le mean collapse
  (la base = la température d'entrée). *Prérequis n°1, débloque le POD.*
- **Objectif tail-aware** (remplace le MSE pur) :
  - **loss quantile** (pinball) sur quantiles bas (q05/q10) → apprend la queue, pas la moyenne ;
  - et/ou **tête de classification d'exceedance** (gel oui/non au seuil) avec **focal loss** ;
  - garder un petit terme champ-moyen pour l'ancrage spatial.
- **Sur-échantillonnage des nuits gélives** (sampler pondéré) — ne pas noyer l'extrême
  dans la saison chaude. (La pondération `frost_alpha` seule ne suffit pas — cf. #7.)

## Phase 2 — Prédicteurs physiques (ce qui *crée* l'extrême)
- **Multi-variable** : découpler in/out channels (entrée N canaux → sortie t2m).
- Entrées du gel radiatif : **vent 10 m (u,v)** + **point de rosée** (déjà en local) +
  **rayonnement LW/SW descendant / nébulosité** (à télécharger ERA5-Land). Tue surtout le **FAR**
  (les nuits ventées/nuageuses ne gèlent pas en surface).
- **SURFEX** : **CERRA-Land `skin_temperature`** (déjà en zarr) en entrée — budget radiatif de
  surface. Plus tard : assimilation SURFEX offline + EKF capteurs (version physique de l'étage C).

## Phase 3 — Terrain qui *localise* l'extrême (FiLM enrichi)
- Conditionnement FiLM DEM enrichi : **TPI** (position topographique), **sky-view factor**,
  **TWI**, **profondeur/encaissement de vallée** — descripteurs de cuvette froide, bien plus
  prédictifs du gel radiatif que elevation/slope/aspect seuls.
- (Optionnel) **FiLM de régime** : descripteurs synoptiques scalaires (vent moyen domaine,
  nébulosité) → bascule advectif ↔ radiatif.

## Phase 4 — Cible plus fine + calibration in-situ
- **CERRA 5,5 km** (vs ERA5-Land 9 km) comme cible/entrée dès le DL 2022-2026 fini — résout
  des cuvettes plus fines. Trancher l'entrée canonique (issue #8).
- **Étage C** (calibration sparse Sencrop, bugs corrigés PR #9) : ancre la queue sur la
  vérité terrain — c'est là que l'extrême est recalé au réel (elevation-aware).

## Phase 5 — Décision & produit
- Sortie = **probabilité de gel** ; **seuil calé sur la ROC** par station/régime pour viser
  FAR<20 %/POD>75 %.
- **Index basis-risk-aware** : arbitrage misses (basis risk assuré) vs fausses alertes
  (coût assureur) selon l'asymétrie de coût.

## Phase 6 — Validation & ops
- Validation **événementielle** (nuit) + **EVT / périodes de retour**, pas seulement pixel.
- Monitoring W&B POD/FAR (fait), entraînement CI/CD RunPod + S3 Scaleway (PR #10).

---

## Phase 7 — Sensibilité à la densité Sencrop (valeur de l'information)
- L'étude peut débloquer une **densité Sencrop plus grande** (arbitrage commercial). L'archi
  à 2 étages **scale avec la densité** : étage A (prior DEM, indépendant des stations) extrait
  la valeur du réseau clairsemé actuel + généralise aux points non instrumentés ; étage C
  (calibration sparse) s'améliore monotonement avec le nombre de stations. → argument
  **downscaling vs calibration au point** (cette dernière ne généralise/scale pas) pour le client.
- **Analyse de sensibilité** : courbe **POD/FAR vs nombre de stations** (sous-échantillonnage
  10/20/30/48, leave-station-out) → quantifie la **valeur marginale d'une station** → nourrit
  l'arbitrage densité ET fait un slide client (« +N stations = +X pts de POD »).
- Plus de stations **en cuvette** = validation de l'hypothèse cold-pooling (TPI/sky-view) +
  réduction du **basis risk**.

## Ordre d'exécution recommandé
1. **ROC/seuil** (plafond à données constantes) — 1 script.
2. **U-Net résiduel** + smoke (POD > 0 ?) — débloqueur.
3. **Objectif tail-aware** (quantile / focal) + sur-échantillonnage gel.
4. **Multi-var** (vent/rosée) + télécharger rayonnement.
5. **DEM cuvettes** (TPI/sky-view) en FiLM.
6. **CERRA fin** + étage C sur Sencrop.
7. **Décision ROC métier** → FAR<20 %/POD>75 %.

Le combo qui peut réellement atteindre la cible : **résiduel + objectif queue-froide +
prédicteurs radiatifs + DEM cuvettes + CERRA + calibration Sencrop**, validé en POD/FAR.

---

## ✅ DÉCISION ARCHI FIGÉE (2026-06-10)

La comparaison à calibration égale a tranché : **CERRA 5,5 km ≫ ERA5-Land 9 km**, et
**CERRA maille + calibration station franchit la cible métier**, sans U-Net.

**Archi produit retenue : `CERRA 5,5 km (maille native) + calibration station OOS`**
(biais médian par station + seuil τ*, fit 2022-2024 → test pur 2025).

| Variante (test pur 2025) | AUC | POD@FAR<20 % | Point opérationnel |
|---|---|---|---|
| CERRA RAW | 0,976 | 0,47 | — |
| **CERRA + calibration station** | **0,987** | **0,80** | **POD 0,55 / FAR 0,12** |

Reproductible en une commande : `uv run frost-product` → `reports_product/frost_product_metrics.json`
+ `frost_product_roc.png`. **Le U-Net (étage A dense) n'a plus d'objet** avec CERRA en entrée
(pas de vérité dense sous 5,5 km) ; le U-Net station-supervisé devient un levier *post-15*
« couverture parcelles non instrumentées ». Voir [[project_downscaling_trainings]].

## Timeline finalisée (2026-06-10 — produit du 15 SÉCURISÉ)

Tout l'amont data/ML est fait : CERRA téléchargé/converti (2015-2026), protocole figé,
chiffre produit packagé. Reste surtout de la **restitution** (demain) et du bonus.

| Échéance | Reste à faire | État |
|---|---|---|
| ✅ Fait (10/06) | CERRA tranché, `frost-product` reproductible, rapport v2 + figures | **fait** |
| Demain (restitution) | **App de présentation des résultats**, **présentation + pricing**, début **modèle chilling** | à lancer |
| Avant le 15 | Éval finale **TEST PUR 2026** quand Sencrop+CERRA 2026 complets (mars-mai en attente CDS) | bloqué saison en cours |
| **15 juin** | Présentation client — archi figée = CERRA + calibration station | — |
| Bonus (post-15) | U-Net station-supervisé (parcelles non instrumentées), tail-aware, sensibilité densité | — |

**Filet de sécurité — relevé** : l'archi figée mesure **POD@FAR<20 % 0,80** (op 0,55 @ FAR 0,12)
sur test pur 2025, calibration out-of-sample. Plus besoin d'upside pour défendre le 15 ;
le U-Net station-supervisé et le tail-aware sont désormais des **améliorations**, pas des prérequis.

## Phase déploiement (après/parallèle au gel)

Architecture produit (cf. [[project_deploy_decision]], `infra_pro.md`) :
- **Back + ML sur Scaleway** : API produit (`api.karpos.pro`), inférence du modèle gel
  (FrostField), artefacts/poids sur Object Storage S3, entraînements RunPod, monitoring W&B.
- **Front sur Vercel** : carte (BFF proxy → API Scaleway) + landing-page.

Le modèle gel figé le 15 alimente l'API Scaleway ; le front Vercel consomme l'indice.
