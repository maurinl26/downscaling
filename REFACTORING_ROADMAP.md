# Roadmap de refactoring — alignement infra sur `weather_routing`

> Document de cadrage. Rédigé le 2026-05-29. À traiter plus tard.
> Objectif : mettre `downscaling` « au propre » en réutilisant les patterns
> éprouvés du repo sœur `weather_routing` (Hydra + Lightning + uv + tests + infra).

## Verdict

**Oui, le repo s'y prête bien** — et c'est même un meilleur candidat que prévu :
le découpage en package (`downscaling/{shared,statistical,deep_learning,prtihvi_wxc}`)
est déjà sain, sans notebooks, et l'infra GPU (RunPod + Terraform + MLflow + split
Mac/cloud) est par endroits **plus mûre** que celle de `weather_routing`.

Les manques sont concentrés sur **3 axes** : configuration (Hydra), boucle
d'entraînement (Lightning), et **tests (actuellement zéro)**.

Caveats importants — tout ne se transpose pas (cf. §« Ne s'applique pas »).

---

## État actuel (constat)

| Dimension | État | Détail |
|---|---|---|
| Layout package | ✅ bon | `downscaling/` installable, sous-modules clairs, 0 notebook, 34 `.py` |
| Build/deps | 🟡 mixte | `setuptools` + `[tool.uv.sources]` ; extras propres (`statistical`/`dl`/`prithvi`/`pmap`) mais build non-uv |
| Configuration | 🔴 manque | `argparse` dans **10** fichiers + **1 YAML monolithique** (`config/drome_ardeche.yml`). Pas de Hydra/OmegaConf. Ajouter une région = copier tout le fichier |
| Boucle d'entraînement | 🔴 manque | Boucle torch **manuelle** (`deep_learning/train.py` : SpectralLoss, warmup+cosine, early stopping, best-ckpt, tensorboard — tout réinventé). Pas de Lightning |
| Tests | 🔴 **absent** | **Aucun** dossier `tests/`, 0 test. Produit propriétaire sans filet |
| Tracking | ✅ présent | MLflow (`MLFLOW_TRACKING_URI`) câblé dans les jobs |
| Infra compute | ✅ mûr | RunPod via Terraform (`.github/workflows/deploy-runpod-infra.yml`), `launch_dl_job.py`, split Mac/MPS (`run_on_mac.py`) vs cloud |
| Modèles | 🟡 hétérogène | U-Net FiLM maison (`nn.Module`, portable Lightning) + Prithvi WxC (foundation NASA/IBM, checkpoints à charger) + PMAP-LES (JAX, hors-stack) |

---

## Ce qui se transpose proprement depuis `weather_routing`

1. **Composition Hydra** (`configs/` avec groupes `domain/`, `data/`, `cluster/`,
   `model/`, `experiment/`). Le repo a déjà la bonne granularité conceptuelle dans
   son YAML — il « suffit » de l'éclater en groupes et d'en faire des `experiment=`
   par région. ⇒ scaling multi-régions (le sens de croissance du produit).
   - Réutiliser le helper `CONFIG_DIR` absolu (`src/wxrouting/paths.py`) : sans lui,
     lancer via un entry point console fait chercher à Hydra un module `configs`
     inexistant (bug rencontré et corrigé côté `weather_routing`).
2. **`LightningModule`** pour le U-Net FiLM et le fine-tune Prithvi : remplace la
   boucle manuelle. `configure_optimizers` (warmup linéaire → cosine),
   `ModelCheckpoint(monitor=val/...)`, early stopping = callbacks natifs.
   - Garder le wrapping « modèle injecté » comme `ArchesGenFinetune` :
     `instantiate(cfg.module, _recursive_=False)` pour que optimizer/scheduler
     restent des `DictConfig` instanciés dans `configure_optimizers`.
3. **Groupes `cluster=`** (local Mac MPS / cloud GPU) : formalise le split déjà
   présent (`run_on_mac.py` vs `launch_dl_job.py`) en `accelerator: auto` +
   `precision: 32-true` sur Mac (MPS n'aime pas bf16 partout).
4. **Pattern `modelstore/` + script `fetch_pretrained.py`** pour les poids Prithvi
   WxC (cf. câblage ArchesWeatherGen : download HF → arbo locale, `.gitignore`).
5. **Tests « dérisquants »** façon `tests/test_finetune.py` (modèle jouet patché,
   valide toute la glue sans GPU/réseau) + `test_hydra_compose.py` (la config
   compose et s'instancie). C'est le plus gros ROI vu l'absence totale de tests.
6. **Entry points unifiés** (`[project.scripts]`) : `downscaling-train`,
   `downscaling-infer` pilotés Hydra, en remplacement des 10 `argparse`.

## Ce qui NE s'applique PAS (à ne pas forcer)

- **Pipeline statistique** (lapse-rate + QDM, `statistical/`) : pas un réseau de
  neurones → **pas de Lightning**. Hydra pour la config oui, mais la logique reste
  numpy/scipy/sklearn. Ne pas tordre en `LightningModule`.
- **PMAP-LES** (JAX + GT4Py + PHYEX, extra `pmap`) : stack totalement distincte.
  L'infra torch/Lightning ne s'y applique pas. Garder isolé derrière son extra.
- **Indices d'assurance** (`shared/indices.py` : `spring_frost_index`,
  `frost_hours`, `growing_degree_days`, `heatwave_index`…) : logique métier xarray,
  pure et bien isolée. **Ne pas mélanger** à l'infra ML — juste la couvrir de tests
  (c'est le cœur de valeur du produit, et le plus testable unitairement).
- **Terraform** : l'infra cible est **RunPod**, pas Scaleway L40S. Ne pas copier le
  `infra/training` de `weather_routing` — le pattern (IaC + launch script + fallback
  Mac) est déjà là et plus avancé. Tout au plus harmoniser la forme.

---

## Roadmap par phases (priorité décroissante par ROI)

### Phase 0 — Filet de sécurité (avant tout refactor) ✅ terminée
- [x] Créer `tests/` + config pytest dans `pyproject.toml`.
- [x] Tests unitaires des **indices d'assurance** (`shared/indices.py`) — valeurs
      connues, cœur métier, aucune dépendance lourde.
- [x] Tests des transfos statistiques (lapse-rate sur DEM jouet, QDM sur distrib
      synthétique à mapping connu).
- [x] CI `tests.yml` (GitHub Actions) lançant `pytest` sur push/PR (matrice
      Python 3.11/3.12, `uv sync --extra statistical` + `uv run pytest`).
> Sans tests, tout refactor d'infra est aveugle. À faire en premier.
> 35 tests passent. Bonus : bug QDM corrigé (terme delta de Cannon 2015 — voir
> `quantile_mapping.py`, ajout de `_mod_ppf`) découvert en écrivant les tests.

### Phase 1 — uv + packaging
- [x] Figer `uv.lock` (généré par `uv sync`). URL git `pmap` corrigée en forme
      `ssh://` (la forme SCP `git@github.com:` n'est pas parsée par uv).
- [ ] Passer le build à uv (build-backend reste `setuptools`), garder les extras.
- [ ] `[project.scripts]` : exposer les entry points (prépare la Phase 2).

### Phase 2 — Hydra (configuration) 🟡 en cours
- [x] Éclater `config/drome_ardeche.yml` en groupes : `configs/{domain,data,
      statistical,dl,indices,cluster}/` + `experiment/`.
- [x] `drome_ardeche` devient `experiment=drome_ardeche` (pattern experiment
      `# @package _global_`) ; nouvelle région = `experiment/<region>.yaml`
      + `domain/<region>.yaml` + `data/<region>.yaml`, plus de copie intégrale.
- [x] Helper `CONFIG_DIR` absolu (`downscaling/paths.py`) pour fiabiliser
      l'entry point console (sinon Hydra cherche un module `configs`).
- [x] Entry point Hydra `downscaling-run` (`scripts/run_downscaling.py`) pilotant
      le pipeline statistique via `@hydra.main`. `hydra-core` ajouté aux deps.
- [x] Test `test_hydra_compose` : config racine + `experiment=` composent,
      overrides `cluster=` et dotlist atteignent les feuilles (5 tests).
- [x] Migrer les `argparse` restants vers la config Hydra : `pipeline.py` perd
      sa CLI (pilotée par `downscaling-run`) ; `train.py`, `inference.py`,
      `run_dl_inference.py`, `run_era5land_downscaling.py` lisent désormais la
      config via `downscaling.config.load_config(overrides)` (flag `--override`
      au lieu de `--config <monolithe>`). `shared.loaders` réexporte
      `load_config` pour les imports historiques.
- [x] Retirer le monolithe `config/drome_ardeche.yml` et
      `run_statistical_downscaling.py`. Deploys basculés : `launch_dl_job.py`
      (RunPod), `runs/scripts/orchestrate.py` et `scripts/run_campaign.py`
      n'envoient plus `--config` ; README + `runs/README.md` mis à jour.
- [x] Concaténation multi-fichiers des saisons : `data.era5_sl` accepte une
      liste (`data.era5_sl=[avril.nc,mai.nc,…]`) ; l'entry point exécute le
      pipeline par fichier et concatène sur l'axe temporel
      (`run_downscaling._source_list` / `_time_dim`). `run_campaign` (branche
      stat) passe tous les mensuels d'une saison. Couvert par
      `tests/test_run_downscaling.py`.

### Phase 3 — Lightning (entraînement DL)
- [ ] `LightningModule` enveloppant le U-Net FiLM (`build_model`) ; migrer
      SpectralLoss / gradient loss dans `training_step`.
- [ ] `configure_optimizers` : warmup linéaire → cosine (réutiliser le pattern
      `SequentialLR` de `weather_routing`).
- [ ] Callbacks natifs : `ModelCheckpoint(monitor="val/rmse")`, early stopping,
      logger MLflow (déjà en place).
- [ ] `cluster=local` (Mac MPS) / `cluster=cloud` (RunPod A100).
- [ ] Smoke test : 1 epoch sur micro-jeu, comme `scripts/run_local.sh`.
- [ ] Idem pour `prtihvi_wxc/finetune.py`.

### Phase 4 — Checkpoints Prithvi WxC (`modelstore/`)
- [ ] `scripts/fetch_pretrained.py` : récupère les poids Prithvi WxC (HF) dans une
      arbo locale `modelstore/` ; `.gitignore`.
- [ ] Loader tolérant à l'absence (patchable en CI), gère device CPU→accelerator.

### Phase 5 — Finitions
- [ ] Ruff + mypy (config alignée sur `weather_routing`).
- [ ] README : section « agnostique du compute » (même config, `cluster=` variable).
- [ ] Harmoniser docstrings / supprimer code mort éventuel.

---

## Notes de transposition (pièges déjà rencontrés côté `weather_routing`)

- **Hydra + entry point console** : `config_path` relatif casse (Hydra cherche un
  module Python). ⇒ chemin **absolu** via `Path(__file__).parents[N] / "configs"`.
- **`instantiate` récursif** : construit l'optimizer sans `params`. ⇒
  `_recursive_=False` sur le module ; instancier optimizer/scheduler dans
  `configure_optimizers`.
- **`.gitignore`** : pas de commentaire *inline* (`outputs/  # ...` ne matche rien).
- **Foundation model conditionné** : vérifier les dépendances de checkpoints
  (ArchesWeatherGen tirait 4 backbones déterministes). Prithvi WxC peut avoir des
  artefacts annexes (stats de normalisation, masques) à embarquer.
- **MPS** : forcer `precision=32-true`, charger le modèle sur CPU puis laisser
  Lightning déplacer vers l'accelerator.
