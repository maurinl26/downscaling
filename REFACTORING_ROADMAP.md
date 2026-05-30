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
| Build/deps | ✅ bon | Build piloté par uv (`uv build`, backend `setuptools`, `[tool.uv] package`) ; extras propres (`statistical`/`dl`/`prithvi`/`pmap`) ; 6 console_scripts résolus |
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

### Phase 1 — uv + packaging ✅ terminée
- [x] Figer `uv.lock` (généré par `uv sync`). URL git `pmap` corrigée en forme
      `ssh://` (la forme SCP `git@github.com:` n'est pas parsée par uv).
- [x] Build piloté par uv : `uv build --wheel` produit
      `downscaling-*.whl` (build-backend `setuptools`, `[tool.uv] package = true`),
      extras conservés. Les 6 `console_scripts` sont bien embarqués dans le wheel.
- [x] `[project.scripts]` : les 6 entry points **résolvent réellement**. Bug
      corrigé : `launch_dl_job.py` et `run_on_mac.py` vivaient hors package
      (`scripts/` racine) alors que les entry points pointaient sur
      `downscaling.scripts.*` → déplacés dans `downscaling/scripts/`.
      `run_on_mac.main` réparé (résidus Phase 2 : `CONFIG_DEFAULT` supprimé,
      `--config` retiré, appels recâblés sur `load_config`). Docstrings + workflow
      `launch-dl-job.yml` basculés sur `uv run <entry-point>`. Vérifié :
      `run-on-mac --task smoke-test` tourne de bout en bout (MPS).

### Phase 2 — Hydra (configuration) ✅ terminée
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

### Phase 3 — Lightning (entraînement DL) ✅ terminée
- [x] `LightningModule` enveloppant le U-Net FiLM (`build_model`) :
      `DownscalingLitModule` (`deep_learning/lightning_module.py`). SpectralLoss /
      gradient loss appliqués dans `training_step` via `DownscalingLoss`
      (réutilisée, source unique). Boucle `torch` manuelle (`train.Trainer`)
      supprimée — `train.main` construit désormais un `pl.Trainer`.
- [x] `configure_optimizers` : warmup linéaire → cosine (`cosine_with_warmup`,
      `LambdaLR`, `interval: epoch`). Optimiseur instancié dans
      `configure_optimizers` (pattern « modèle injecté »).
- [x] Callbacks natifs : `ModelCheckpoint(monitor="val/rmse")`, `EarlyStopping`,
      logger MLflow si `MLFLOW_TRACKING_URI` sinon `CSVLogger` (sans dép).
- [x] `cluster=local` (Mac MPS, `32-true`) / `cluster=cloud` (RunPod GPU,
      `bf16-mixed`) : `build_trainer` lit accelerator/precision/num_workers du
      groupe `cluster=`. `launch_dl_job` (unet-train) passe `cluster=cloud`.
- [x] Smoke test : `fast_dev_run` sur modèle jouet + dataset aléatoire, sans GPU
      ni réseau (`tests/test_lightning.py`, gardé par `importorskip`).
- [x] Ajout `lightning>=2.2` aux extras `dl` et `prithvi` + `uv.lock` régénéré.
- [x] Idem pour `prtihvi_wxc/finetune.py` : `PrithviFinetuneLitModule` +
      `PrithviFinetuneDataModule` (`prtihvi_wxc/lightning_finetune.py`). Modèle
      injecté, **adapter seul optimisé**, backbone gelé en `eval()`
      (`on_train_epoch_start`), checkpoint réduit à l'adapter
      (`on_save_checkpoint`), supervision sparse Netatmo. `PrithviWxCFinetuner.run`
      délègue à un `pl.Trainer` (callbacks `val/rmse`, logger CSV).
      Smoke test `tests/test_lightning_prithvi.py` (4 invariants : fast_dev_run,
      backbone gelé, optim adapter-only, checkpoint adapter-only).
      Bonus : imports cassés du package `prtihvi_wxc` réparés
      (`downscaling.deep_learning.prithvi_wxc` → relatifs ; `spring_frost` →
      `spring_frost_index`) ; `__init__` tolérant à l'absence de l'extra `prithvi`.

### Phase 4 — Checkpoints Prithvi WxC (`modelstore/`) ✅ terminée
- [x] `fetch_pretrained.py` (entry point `fetch-pretrained`) : `snapshot_download`
      des poids Prithvi WxC + adapter Granite (et annexes : config, scalers,
      climatologie) dans `modelstore/` (cf. `downscaling.paths.MODELSTORE`,
      surchargeable par `DOWNSCALING_MODELSTORE`). Manifeste éditable, `--list`
      (dry-run sans réseau), `--only <key>`, idempotent. `modelstore/` gitignored.
- [x] Loader tolérant à l'absence : `resolve_artifact` cherche `modelstore/`
      d'abord puis retombe sur HuggingFace (tourne hors-ligne une fois
      `fetch-pretrained` passé). `resolve_device("auto")` → cuda/mps/cpu.
      `from_pretrained(device="auto")`. Le chargement Granite reste tolérant
      (try/except) — adapter aléatoire si indispo.
- [x] Tests `tests/test_modelstore.py` (9) : manifeste/plan/`fetch_one`
      (snapshot monkeypatché), résolution modelstore-first + repli HF,
      `resolve_device` — sans réseau, gardés `importorskip` côté loader.

### Phase 5 — Finitions ✅ terminée
- [x] Ruff (`E`,`F`,`W`,`I`,`UP`,`B`) configuré + dépôt **clean** (`ruff check`
      passe) ; per-file-ignores tests (E402). mypy configuré en **typage
      progressif** (`check_untyped_defs=false`, informatif). CI : job `lint`
      (Ruff bloquant, mypy non bloquant).
- [x] README : section **Architecture** (layout, composition Hydra, entry points)
      + **Agnostique du compute** (même config, `cluster=local/cloud`) +
      **Bonnes pratiques & développement** (install/extras, tests `importorskip`,
      lint/typage, conventions).
- [x] Code mort supprimé / harmonisé : imports inutilisés, bloc `encoding` mort
      (prithvi inference), `raise ... from`, `zip(strict=)`, `stacklevel`.
      Bonus correctif : coordonnée `lon` manquante dans la sortie DL
      (`deep_learning/inference.py`).

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
