# Canonical input source — CERRA (2026-06-26)

> ADR court, en réponse à [downscaling#8](https://github.com/maurinl26/karpos-downscaling/issues/8)
> ("Cohérence de la source d'entrée étages A/C : ERA5-Land vs CERRA").

## Décision

La source d'entrée canonique du chemin produit Karpos est **CERRA**.

ERA5-Land est conservé uniquement comme **baseline comparative** dans les
rapports scientifiques (cf. `gmd-paper-draft.md`, `karpos-slr-calibration-report.md`)
— il n'alimente plus aucun pipeline de production.

## Raisons

1. **KarposSLR (calibration QDM) tourne déjà sur CERRA atm.** Le workflow
   `calibrate-qdm.yml` (côté `karpos-engine`) consomme `cerra_atm_{year}.nc` et
   produit le joblib transfer-function exposé à la production via
   `vars.QDM_JOBLIB_URI`. C'est la source de vérité de la chaîne d'indices
   aujourd'hui.

2. **CERRA bat ERA5-Land en RAW.** L'audit KarposSLR documenté dans
   `karpos-slr-calibration-report.md` (table POD/FAR Sencrop 2022-2025) montre que
   **CERRA brut détecte +40 pts de POD** vs ERA5-Land brut. Le socle de
   réanalyse domine la profondeur du réseau de descente d'échelle.

3. **Cohérence d'entraînement.** Les évolutions étage A (U-Net) et étage C
   (calibration sparse Sencrop) sont désormais alignées sur des inputs CERRA
   (cf. `CERRACoarseProvider`, `configs/calibration/default.yaml`).

## Migration prévue

**AROME-native** (1.3 km) est planifié pour la **release de septembre 2026**
(cf. [downscaling#70](https://github.com/maurinl26/karpos-downscaling/issues/70)).
La bascule remplacera CERRA par AROME en entrée de QDM KarposSLR, sans changer
l'architecture (le `CERRACoarseProvider` deviendra source-agnostique ou un
`AROMECoarseProvider` parallèle).

ERA5-Land restera la baseline comparative dans les papiers, pas dans le code
produit.

## Ce qui a changé concrètement

- `configs/calibration/default.yaml` — `file_template_cerra: "cerra_{date}.nc"`
  (était `era5land_{date}.nc`)
- `docs/infra/infra_pro.md` — arborescence S3 reflète `cerra_{date}.nc`
- `docs/architecture.md` — section "chemin B" précise CERRA canonique, ERA5-Land
  rétrogradé baseline

Aucun changement de code applicatif : le `CERRACoarseProvider` acceptait déjà
les deux templates via le paramètre `file_template`. Seule la valeur par défaut
de la config + les docstrings/docs étaient désynchronisées du chemin produit
réel.
