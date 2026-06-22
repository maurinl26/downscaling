---
license: cc-by-4.0
language: en
library_name: pytorch
pipeline_tag: image-to-image
tags:
  - super-resolution
  - meteorology
  - downscaling
  - climate
  - FiLM
  - U-Net
  - radiative-frost
  - viticulture
  - parametric-insurance
datasets:
  - proprietary
metrics:
  - POD@FAR<0.20
  - ROC-AUC
  - RMSE
---

# karpos-downscaling — Baronnies v1

Reference weights for the **residual U-Net with FiLM-conditioned terrain modulation** described in:

- *karpos-downscaling: A meteorological super-resolution framework bridging numerical weather prediction and machine learning* (JOSS, in submission)
- *FiLM-conditioned residual U-Net downscaling of frost-risk temperature fields* (GMD, in submission)

Code repository: <https://github.com/maurinl26/karpos-downscaling>

## Model description

Four-level U-Net (≈ 19 M parameters; exactly 19,053,030 across 246 tensors) with FiLM modulation `x ← γ(DEM) · x + β(DEM)` applied at every encoder and decoder level. A small pyramidal DEM encoder produces spatially-varying `(γ_ℓ, β_ℓ)` maps shared between encoder and decoder. The output is **residual**: `ŷ = t2m_input + δ(x_met, DEM)`. Training combines a reanalysis target (stage A) with tail-weighted in-situ supervision against Sencrop minima (stage C).

## Training data

- **ERA5-Land** 9 km (Copernicus CDS, open access) — meteorological inputs
- **CERRA** 5.5 km (Copernicus CDS, open access) — stage-A target
- **COP-DEM** 30 m → 1 km (ESA, open access) — terrain conditioning
- **Sencrop** 48 stations, Baronnies Provençales (proprietary usage agreement) — stage-C ground truth

Training period: 2015–2021 (TRAIN). DEV held-out: 2022–2025. PURE TEST 2026 reserved for a single end-of-cycle evaluation.

## Intended use

- Reproducing the JOSS and GMD figures and ablations
- Benchmarking new super-resolution architectures on the same temporal splits
- Research on FiLM-conditioned image-to-image regression in meteorology
- Building blocks for AIWP fine-tuning pipelines that need locally-supervised target fields

## Out-of-scope use

- **Direct operational frost detection for parametric insurance** — for production use, see the Karpos commercial service. The released weights are MVP-grade reference checkpoints, not production-calibrated.
- **Extrapolation outside the Baronnies relief class** without retraining. Out-of-domain behaviour is documented as a known limitation (cf. paper §7.2).
- **Real-time operational decisions** without a documented downstream calibration and quality-monitoring pipeline.

## Performance (DEV held-out 2022–2025)

| Metric | Value |
|---|---|
| ROC-AUC | 0.964 |
| POD @ FAR < 0.20 | 0.316 (raw output) / 0.611 (with per-station in-sample debias, indicative upper bound) |
| RMSE (Tmin) | reported in the GMD companion paper |

All figures averaged over 5 seeds with bootstrap confidence intervals. See the GMD paper §6 for the complete table, ablations, and pairwise McNemar tests against the QDM statistical baseline.

## How to use

```python
from downscaling.hub import load_baronnies_v1

model = load_baronnies_v1(device="cuda")
# model: torch.nn.Module ready for inference
# expected input shape: (B, C_meteo + C_dem, H, W) with H × W = 168 × 120
```

For the full evaluation harness, baselines, and Hydra configurations, see the GitHub repository.

## Limitations and known biases

- **Discriminative ceiling at AUC ≈ 0.96**, imposed by the CERRA reanalysis target. Adding meteorological predictors (dew point, wind) does not raise the ceiling on this dataset.
- **MSE-based in-situ calibration degrades detection** (mean-regression pathology). Tail-weighted supervision is required for the cold tail; see the GMD paper §5.
- **Trained on the Baronnies pre-alpine basins**. Behaviour on hillside vineyards (Côtes du Rhône, Bourgogne, …) is the subject of ongoing follow-up work.
- **No surface-state conditioning yet**. SURFEX integration (soil moisture, canopy temperature, vegetation type) is planned for v2; see paper §7.3.

## Citation

Software release:

```bibtex
@software{karpos_downscaling_baronnies_v1,
  author       = {Maurin, Loïc},
  title        = {karpos-downscaling — Baronnies v1},
  year         = 2026,
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.20783563},
  url          = {https://huggingface.co/karpos26/karpos-downscaling-baronnies}
}
```

Companion papers — see `CITATION.cff` in the GitHub repository.

## License

Weights released under **Creative Commons Attribution 4.0 (CC-BY 4.0)**. Code in the companion GitHub repository released under **Apache 2.0**. The Sencrop sensor data used during training is proprietary and is **not** redistributed with these weights; retraining requires an independent Sencrop usage agreement.

## Authors and acknowledgements

Loïc Maurin (External Lecturer in Stochastic Filtering, École Nationale de la Météorologie, Météo-France, Toulouse).

This work was conducted in the context of the Karpos project. The author thanks Sencrop for sensor-network access, Copernicus for CERRA and ERA5-Land, and the SURFEX/AROME development teams at Météo-France/CNRM.

## Release strategy

The Baronnies weights are released as a **public reference MVP**: they support reproducibility of the JOSS and GMD papers and academic follow-up work, without exposing any commercially calibrated configuration. Region-specific production weights (Côtes du Rhône and beyond) will be released publicly **after the commercial moat is built** in each region (typically T+18 to T+24 months after the first signed insurance contract, one validated frost season, and a renewed customer cycle).
