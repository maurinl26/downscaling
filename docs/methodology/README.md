# Methodology — open methodology behind karpos-downscaling

This folder contains the **scientific methodology** behind `karpos-downscaling`,
released open source under Apache 2.0 alongside the code.

The goal is twofold :

1. **Scientific credibility** : anyone can reproduce, audit, or cite the
   methodology independently of the proprietary commercial integration.
2. **Cooperation** : partners (Météo-France, INRAE, ANITI, Sencrop) and
   academic collaborators have a citable methodological reference.

## Contents

| Document | Purpose |
|---|---|
| `karpos-slr-calibration-report.md` | Detailed methodology of the statistical downscaling pipeline (KarposSLR) : CERRA → 1 km via lapse-rate, QDM, RBF Sencrop residual ; bias diagnostics ; regime-stratified POD/FAR/CSI evaluation. 9 annexes covering bug catalog, cold audit, run #28, QDM A/B verdict. |
| `chilling-model-baronnies.md` | Phenological complement : chilling unit accumulation model for apricot in Drôme Baronnies (Corréard, Plaisians, Mirabel-aux-Baronnies). |
| `gmd-paper-draft.md` | Skeleton outline of the GMD (Geoscientific Model Development) paper — "A regime-aware statistical downscaling framework for spring frost forecasting in mountainous arboriculture". Target submission : autumn 2026. |
| `joss-paper-draft.md` | Skeleton outline of the JOSS (Journal of Open Source Software) paper — `karpos-downscaling: A toolkit for sparse-network anchored statistical downscaling of frost risk in arboriculture`. Target submission : late summer 2026. |
| `canonical-input.md` | Short ADR (2026-06-26) documenting **CERRA** as the canonical input source of the production chain. ERA5-Land kept as a comparative baseline. AROME-native migration planned September 2026 (downscaling#70). |

## Boundary with the proprietary stack

The methodology, code, and aggregate evaluation metrics are **open source**
(Apache 2.0). They are sufficient to reproduce the science end-to-end on any
region with similar inputs (CERRA + a sparse station network).

The downstream **productive integration** stays in the parent repository
`karpos/parametric_insurance` and is **not** open source. It includes :

- the PWA (Next.js) and landing page (Astro),
- the FastAPI BFF with Stripe + Supabase integrations,
- the calibrated artefacts (`*.joblib`, `*.zarr`) trained on the
  Drôme/Baronnies data,
- the parcel registry and customer data,
- the actuarial calibration of the insurance index,
- the legal opinion stack for the DEP (Déduction pour Épargne de Précaution)
  certificate.

This separation lets external contributors and reviewers engage with the
science freely while preserving the commercial moat of Karpos.

## Citing

If you use `karpos-downscaling` or the methodology described here in
publications, please cite :

```bibtex
@software{karpos_downscaling_2026,
  author       = {Loïc Maurin},
  title        = {karpos-downscaling: a regime-aware statistical
                  downscaling framework for spring frost forecasting},
  year         = 2026,
  publisher    = {Karpos},
  url          = {https://github.com/maurinl26/karpos-downscaling},
  license      = {Apache-2.0}
}
```

Updated bibliographic reference will be provided once the JOSS and GMD
manuscripts are published.

## Contact

Loïc Maurin — loic.maurin@karpos.pro · [@maurinl26](https://github.com/maurinl26)
