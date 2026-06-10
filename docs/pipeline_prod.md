# Pipeline de production — indice gel & alerte temps réel

> Principe directeur : **un cœur unique source-agnostic, deux régimes de données.**
> Le moat (descente d'échelle + calibration station) est le même code ; seule la
> **source amont** change selon qu'on veut l'indice climatique (réanalyse, batch) ou
> l'alerte temps réel (prévision, streaming).

## La bascule clé : réanalyse ≠ prévision

| | Indice climatique / pricing | **Alerte gel temps réel** |
|---|---|---|
| Source | **CERRA 5,5 km** (réanalyse) | **AROME 1,3 km / ECMWF** (prévision) |
| Latence source | ~3,5 mois | heures (4 runs/jour) |
| Cadence | batch mensuel/saisonnier | J0/J+1, à chaque run modèle |
| Calibration station | biais médian/station (offline, OOS) | **Kalman par station (online)** vs obs Sencrop live |
| Usage | période de retour, prime, basis risk | « gèlera-t-il cette nuit ? » |
| Public (GTM) | assurance / IGP / industriels | **signal B2B → Sencrop / assureurs** (pas d'app fermier Karpos) |

**Point dur** : CERRA est une réanalyse à 3,5 mois de retard → **inutilisable pour
l'alerte**. L'alerte exige une **prévision** (AROME, ou ECMWF open-data). Et le biais
appris contre CERRA **ne se transfère pas** à un modèle de prévision → il faut le
**re-dériver online** (filtre de Kalman par station, script `run_sencrop_kf_bias.py`
déjà écrit) au fil des observations Sencrop.

## Architecture en couches

```
        ┌──────────────── SOURCES (adaptateurs) ────────────────┐
        │ CERRA 5,5 km réanalyse   (download-cerra, batch)       │ → indice
        │ AROME 1,3 km / ECMWF      (adaptateur prévision)        │ → alerte
        │ Sencrop API               (obs live + historique)       │ → calib + vérif
        └────────────────────────────┬───────────────────────────┘
                                      ▼
              ┌──────────── CŒUR (source-agnostic) ────────────┐
              │ regrid → grille 1 km · Tmin nocturne            │   = le moat
              │ calibration station (médiane offline | KF online)│   frost_eval_core
              │ règle de décision τ* → indice / probabilité gel │   (généralisé)
              │ émet FrostField (contrat OTA + badge source/res)│
              └────────────────────────┬───────────────────────┘
                                       ▼
   ┌─────────────────┬─────────────────┼──────────────────┬──────────────────┐
   ▼                 ▼                 ▼                  ▼                  ▼
CALIB STORE     SERVING (API Scaleway)              FRONT Vercel        ALERTE B2B
biais+τ*        api.karpos.pro /api/v1/frost/*       app.karpos.pro      push → Sencrop
versionné       FrostField (contrat stable)          carte (OTA swap)    / assureurs
/source         climatique + /forecast (nouvel ep.)                      (webhook/API)
   │
ORCHESTRATION : RQ workers + cron (CERRA mensuel · prévision 4×/jour) · RunPod (retrain) ·
                monitoring W&B/Grafana + boucle de vérif (prévision vs Sencrop réalisé → POD/FAR live)
```

## Mapping sur la stack existante (rien à réinventer)

- **API = le produit** (`backtest/api`, FastAPI, `/api/v1/*`, JWT, S3) sur **Scaleway**
  `api.karpos.pro`. Le pipeline alimente l'endpoint `FrostField`. Cf. [[project_deploy_decision]].
- **Front carte = Vercel** `app.karpos.pro`, contrat `FrostField` **OTA** : on remplace la
  donnée fake par la sortie calibrée CERRA, **sans toucher le front** (badge `source` :
  `cerra-5.5km` → `downscale-parcelle-v1`). Cf. [[project_frost_data_ota]].
- **TimescaleDB + RQ + Redis** (Scaleway, Phase 2) : obs Sencrop, indices, jobs async.
- **S3** `karpos-parametric-results` : champs, artefacts de calibration, poids.
- **Sencrop = canal de diffusion** des alertes fermier (pas d'app fermier Karpos). Karpos
  garde l'IP de l'indice. Cf. [[project_gtm_sencrop_channel]].

## Le cœur à généraliser (`frost_eval_core` → service)

`frost_eval_core.py` contient déjà la logique source-agnostic (échantillonnage maille→station,
ROC, calibration OOS). Pour la prod, l'industrialiser en service :
1. **Adaptateur de source** = interface `(date) → champ Tmin nocturne sur grille 1 km`.
   Implémentations : CERRA (fait), AROME/ECMWF (à écrire), ERA5-Land (fait).
2. **Calibration** = interface `predict + obs → correction/station`. Offline médiane (fait,
   indice) ; online Kalman (existe, à brancher pour l'alerte).
3. **Décision** = seuil τ* → `FrostField` (severity, tmin, payout). Contrat déjà défini
   `app/lib/frost.ts`.

## Régime ALERTE — faisabilité & garde-fous

Faisable avec la stack. Ce qui change vs l'indice :
1. **Source prévision** : AROME 1,3 km (Météo-France, 4 runs/jour, haute-rés, idéal cuvettes)
   ou ECMWF IFS/AIFS open-data (gratuit, global). À arbitrer (AROME = meilleure rés, MF API).
2. **Calibration online** : Kalman par station vs Sencrop live (le biais prévision dérive
   du biais réanalyse → re-fit obligatoire). `run_sencrop_kf_bias.py` est la brique.
3. **Budget latence** : run AROME J0 matin → Tmin prévu nuit → alerte l'après-midi. Le gel
   radiatif est prévisible 12-36 h à l'avance → OK.
4. **Boucle de vérification** : chaque matin, prévision vs Tmin Sencrop réalisé → POD/FAR
   opérationnel suivi en continu → ajuste le seuil. C'est le contrôle qualité de l'alerte.
5. **Diffusion (GTM)** : l'alerte est un **signal B2B** injecté dans Sencrop / chez l'assureur,
   pas un push Karpos→fermier.

## Séquencement recommandé

| Étape | Contenu | Échéance |
|---|---|---|
| **1. Indice en prod** | brancher CERRA+calibration sur l'API `FrostField` (swap Track B), front carte réelle | viser le 15 |
| **2. Service cœur** | extraire `frost_eval_core` en service avec interface adaptateur/calibration/décision | foulée |
| **3. Alerte (pull Phase 4)** | adaptateur AROME/ECMWF + Kalman online + job 4×/jour + boucle vérif + webhook B2B | fast-follow |

Indice et alerte partagent ~80 % (cœur + calibration + serving). Le delta alerte =
adaptateur prévision + calibration online + diffusion push. La Phase 4 « temps réel »
de [[project_architecture]] (prévue ~2027) devient **avançable** maintenant que la
calibration est démontrée.
