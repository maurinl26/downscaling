# Plan — downscaling stress thermique (Tmax diurne)

Étendre le downscaling gel (Tmin nocturne, chemin B/étage C) au **stress thermique**
(Tmax diurne), pour alimenter les indices canicule / échaudage à 1 km. Même
architecture (FiLM U-Net, entrée CERRA/AROME grossière, cible Sencrop station),
la seule bascule scientifique est **min nocturne → max diurne**.

## Ce qui est prêt (config additive)

- `configs/calibration/heat.yaml` — `reduce: max`, `lapse_rate: -6.5e-3` (gradient
  diurne bien mélangé au lieu de `-4e-3` inversions nocturnes), checkpoint distinct.
- `configs/experiment/drome_ardeche_heat.yaml` — mêmes groupes région, date estivale.
- `configs/indices/default.yaml` — bloc `heat:` (seuils 35 °C, canicule ≥ 3 jours) ;
  l'étage indices supporte déjà `heat_stress_days` / `heatwave_index`.

Commande cible (une fois les prérequis code livrés) :

```
uv run run-calibration experiment=drome_ardeche_heat calibration=heat
```

## Prérequis code (issues dédiées) — sans eux, la cible reste nocturne

La définition « nocturne » est portée par du **code hardcodé**, pas seulement par
`reduce`. Il faut donc, en rétro-compatible (défauts = nuit/min, gel inchangé) :

1. **Fenêtre diurne paramétrable** — la fenêtre `20h→08h` est en dur dans
   `prtihvi_wxc/stations.py` (`dataframe_to_station_obs`) et `netatmo_qc.py`
   (`load_netatmo_parquet`). Ajouter une fenêtre diurne (ex. `06h→20h`).
2. **Agrégateur `tmax_daytime`** — analogue `np.nanmax` de `tmin_nocturnal`
   (`netatmo_qc.py`), routé par `night_station_targets` (`stations.py`).
3. **QC diurne** — relever `T_MAX_PLAUSIBLE_C` (25 °C, calé « nuit la plus chaude »)
   et traiter le **biais radiatif diurne** (le QC actuel suppose un biais solaire nul :
   c'est un vrai point méthodologique, pas une simple constante).
4. **Métriques hot-tail** — POD/FAR sont câblés côté froid (`obs < seuil`,
   `lightning_module.py` / `train.py`). Ajouter l'analogue chaud (`obs > seuil`,
   ex. 35 °C).
5. **Perte tail-aware (option)** — exposer `loss_quantile` dans `run_calibration.py`
   et viser un **quantile haut** (queue chaude) au lieu de la queue froide actuelle.
6. **Tâche RunPod + identité W&B** — pas de tâche `run-calibration`/heat dans
   `launch_dl_job.py` ; loguer explicitement la variante (min/max, seuil, lapse_rate)
   pour distinguer gel/chaleur dans W&B (`karpos-downscaling`).

## Données

Le bulk Sencrop couvre l'été (mai→sept), donc la cible Tmax diurne est disponible
(contrairement au pipeline gel qui s'arrête à la fenêtre nocturne). Choisir un
épisode chaud de référence (canicule) pour le premier run.

## Caveat scientifique

Le downscaling diurne est intrinsèquement plus dur que nocturne : forçage radiatif,
biais d'exposition des capteurs, couche limite convective. La QC diurne (point 3)
et la métrique/queue (points 4-5) doivent être validées avant de présenter un skill.
