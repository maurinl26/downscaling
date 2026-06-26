# Passage à une infra d'entraînement mature

Cible : **monitoring des scores (W&B)** + **entraînement déclenchable depuis CI/CD**
sur **RunPod** (GPU), données & artefacts sur **Scaleway Object Storage (S3)**.
Remplace le flux manuel actuel (upload tuiles sur volume RunPod, logger CSV/MLflow,
checkpoints locaux sur le Mac).

---

## 1. Architecture cible

```
 GitHub (push tag / workflow_dispatch)
        │  GitHub Actions (.github/workflows/train-downscaling.yml)
        ▼
   launch-dl-job  ──provisionne──▶  RunPod pod (A100/L4, image karpos/downscaling:latest)
                                          │  1. pull tuiles + DEM  ◀── Scaleway S3 (s3.fr-par.scw.cloud)
                                          │  2. run-dl-train --override cluster=cloud
                                          │  3. log scores en direct ──▶ Weights & Biases
                                          │  4. push best_model.ckpt  ──▶ Scaleway S3 (artifacts/)
                                          ▼
                                   W&B run (val/rmse, POD/FAR, hyperparams, courbes)
```

Principe : **le code et la config sont versionnés (git)** ; **les données et les
poids vivent sur S3** ; **les scores vivent sur W&B** ; **le déclenchement est CI/CD**.
Rien d'important ne reste sur le Mac.

---

## 2. Weights & Biases — monitoring des scores

**Déjà câblé** (`deep_learning/train.py::_build_logger`) : si `WANDB_PROJECT` (ou
`WANDB_API_KEY`) est défini, Lightning logge dans W&B (sinon MLflow, sinon CSV).

À faire :
- `uv add wandb` dans `downscaling/pyproject.toml` (extra `dl`), et l'ajouter à l'image Docker.
- Variables : `WANDB_API_KEY`, `WANDB_PROJECT=karpos-downscaling`, `WANDB_ENTITY=<org>`.
- Métriques loggées aujourd'hui : `train/loss`, `train/rmse`, `val/loss`, `val/rmse`.
  **À ajouter** : POD/FAR de validation gel (callback custom comparant la sortie aux
  stations Sencrop — cf. issue #7), pente MOS, et les hyperparams (base_ch, lr, données).
- Comparaison de runs (baselines au point vs U-Net) dans un même projet → un tableau
  W&B unique RAW/MOS/EQM/KF/U-Net sur POD/FAR.

---

## 3. Scaleway Object Storage (S3) — données & artefacts

Endpoint : `https://s3.fr-par.scw.cloud` (région `fr-par`). Compatible S3 → `boto3`
ou `s3cmd`/`rclone`. Bucket suggéré : `karpos-downscaling`.

```
s3://karpos-downscaling/
├── training/drome_ardeche/2015-2021/   coarse/*.nc  fine/*.nc  dem_attributes.nc
├── sencrop/                            sencrop_{date}.csv  (étage C)
├── cerra_fine/                         cerra_{date}.nc     (étage C, source canonique)
└── artifacts/<run_id>/                 best_model.ckpt  normalization_stats.json
```

À faire :
- Helper `downscaling/scripts/s3_sync.py` (boto3) : `push <local> <s3uri>` / `pull <s3uri> <local>`,
  endpoint+creds via `SCW_ACCESS_KEY`/`SCW_SECRET_KEY`/`SCW_S3_ENDPOINT`.
- `launch_dl_job` : avant `run-dl-train`, `pull` les tuiles depuis S3 vers `/workspace/data/training/` ;
  après, `push` le checkpoint vers `artifacts/<run_id>/`. Supprime l'étape manuelle d'upload sur le volume réseau.
- Côté Mac, produire les tuiles (`build_downscaling_tiles.py`) puis `s3_sync push` une seule fois.

---

## 4. CI/CD — entraînement déclenchable (GitHub Actions)

`.github/workflows/train-downscaling.yml` :
- Déclencheurs : `workflow_dispatch` (bouton + inputs : `epochs`, `base_ch`, `gpu`) et
  optionnel `push: tags: ['train-*']`.
- Job léger (ubuntu) : `pip install runpod`, puis `uv run launch-dl-job --task unet-train`
  (qui provisionne le pod RunPod, lequel fait pull S3 → train → log W&B → push S3).
- Le job CI **ne fait pas le GPU** : il orchestre RunPod et attend (`--status`) ou rend la main.

Secrets GitHub (repo → Settings → Secrets → Actions) :
`RUNPOD_API_KEY`, `SCW_ACCESS_KEY`, `SCW_SECRET_KEY`, `WANDB_API_KEY`,
`SCW_S3_ENDPOINT` (= https://s3.fr-par.scw.cloud), `WANDB_PROJECT`.

L'image Docker `registry.fr-par.scw.cloud/karpos/downscaling:latest` doit embarquer
`wandb`, `boto3`, et le code à jour (rebuild après merge).

---

## 5. Étapes de migration (ordre)

1. `uv add wandb boto3` (extra `dl`) + rebuild image Docker.
2. `s3_sync.py` (push/pull S3) + créer le bucket `karpos-downscaling`.
3. `s3_sync push` des tuiles 2015-2021 + DEM.
4. Adapter `launch_dl_job` : pull S3 avant train, push checkpoint après.
5. Workflow GitHub Actions `train-downscaling.yml` + secrets.
6. 1er run CI/CD → vérifier W&B (scores) + checkpoint sur S3.
7. (Plus tard) callback POD/FAR de validation (issue #7) loggé dans W&B.

---

## 6. Espace disque — nc → zarr

NetCDF4 est déjà compressé (HDF5/zlib) → la conversion zarr (zstd) gagne ~0-30 %
en octets, mais surtout : **store chunké consolidé, accès lazy, déportable kDrive/S3**.
- Réanalyses : `uv run convert-nc-to-zarr --source cerra` (idem era5land/cerra-land),
  **puis supprimer les `.nc`** sources. C'est le vrai gain disque (les `.nc` mensuels CERRA).
- Artefacts régénérables (tuiles `data/training/*.nc`, nuits `cerra_fine/`) : ne pas les
  stocker durablement en local — les pousser sur S3 et régénérer à la demande.
- Sur le Mac (SSD ~5 Gi libres, contrainte réelle) : checkpoints + tuiles → S3 ;
  garder en local le minimum.

---

## 7. Annexe — créer les ressources Scaleway (pas-à-pas)

Aucun projet Scaleway n'existe encore. À faire une fois, dans la console
[console.scaleway.com](https://console.scaleway.com) :

### 7.1 Projet + bucket Object Storage
1. **Organisation → Projects** : créer (ou choisir) un projet, ex. `karpos`.
2. **Storage → Object Storage → Create bucket** :
   - Nom : `karpos-downscaling` (globalement unique par région).
   - Région : **fr-par** (Paris) — cohérent avec l'image Docker `registry.fr-par.scw.cloud`.
   - Visibilité : **Private**.
3. (Optionnel) Créer les « dossiers » logiques `training/`, `sencrop/`, `cerra_fine/`,
   `artifacts/` — ou laisser `s3-sync` les créer au 1er `push`.

### 7.2 Clé API (IAM)
1. **IAM → API keys → Generate an API key** (attachée à une application IAM dédiée,
   ex. `downscaling-ci`).
2. Donner à cette application la **policy** `ObjectStorageFullAccess` (ou restreinte au
   bucket `karpos-downscaling`).
3. Récupérer **Access Key ID** → `SCW_ACCESS_KEY` et **Secret Key** (affichée une seule
   fois !) → `SCW_SECRET_KEY`.

### 7.3 Endpoint
- `SCW_S3_ENDPOINT = https://s3.fr-par.scw.cloud`  ·  `SCW_S3_REGION = fr-par`.

### 7.4 Test local
```bash
export SCW_ACCESS_KEY=... SCW_SECRET_KEY=...
export SCW_S3_ENDPOINT=https://s3.fr-par.scw.cloud
uv run s3-sync push downscaling/data/dem/dem_attributes.nc s3://karpos-downscaling/test/
uv run s3-sync pull s3://karpos-downscaling/test/ /tmp/s3test/
```

### 7.5 Secrets / variables GitHub Actions
Repo `downscaling` → **Settings → Secrets and variables → Actions** :
- **Secrets** : `RUNPOD_API_KEY`, `SCW_ACCESS_KEY`, `SCW_SECRET_KEY`, `WANDB_API_KEY`.
- **Variables** : `SCW_S3_ENDPOINT` (= https://s3.fr-par.scw.cloud), `WANDB_PROJECT`
  (= karpos-downscaling), `WANDB_ENTITY` (= ton org W&B).

### 7.6 W&B
1. Créer le projet `karpos-downscaling` sur [wandb.ai](https://wandb.ai).
2. `WANDB_API_KEY` depuis wandb.ai/authorize → secret GitHub + env local si run sur le Mac.

### 7.7 Pousser les données une fois
```bash
uv run s3-sync push downscaling/data/training/ s3://karpos-downscaling/training/drome_ardeche/2015-2021/
```
Ensuite : workflow *Launch DL Job* (dispatch) → le pod pull depuis S3, entraîne, logge W&B, push le checkpoint.

> Alternatives CLI au besoin : `rclone` (remote type `s3`, provider `Scaleway`) ou
> `s3cmd` — mêmes clés/endpoint. `s3-sync` (boto3) suffit pour le flux tuiles/poids.
