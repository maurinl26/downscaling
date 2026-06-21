---
title: 'karpos-downscaling: A meteorological super-resolution framework bridging numerical weather prediction and machine learning'
tags:
  - Python
  - PyTorch
  - meteorology
  - climate
  - downscaling
  - super-resolution
  - statistical post-processing
  - AI weather prediction
  - FiLM
  - radiative frost
  - viticulture
authors:
  - name: Loïc Maurin
    orcid: 0009-0004-8117-4850
    affiliation: 1
affiliations:
  - name: External Lecturer in Stochastic Filtering, École Nationale de la Météorologie, Météo-France, Toulouse, France
    index: 1
date: 21 June 2026
bibliography: paper.bib
---

# Summary

`karpos-downscaling` is a Python/PyTorch **framework for meteorological super-resolution** bridging classical numerical weather prediction (NWP) infrastructure and machine-learning post-processing. Given a coarse-resolution input field and a digital elevation model, it produces a high-resolution (~1 km) thermal field supervised by in-situ agricultural sensors. The reference architecture is a **residual U-Net with FiLM-conditioned terrain modulation** [@perez2018] targeting the detection of rare radiative-frost events, but the data pipeline, evaluation harness, and statistical baselines are model-agnostic. All training configurations are Hydra-frozen and seed-controlled for apples-to-apples comparison. Detailed methodology and ablation results are reported in the companion paper [@maurin2026gmd].

# Statement of need

Frost detection in complex terrain is a long-standing challenge for meteorological post-processing. Dynamical downscaling with high-resolution NWP models such as AROME [@seity2011] carries documented warm biases under stable nocturnal boundary layers [@sandu2013; @couvreux2020]. Statistical methods (MOS, Kalman filter, Quantile Delta Mapping [@cannon2015]) operate per station and ignore terrain structure beyond elevation. Deep-learning super-resolution [@vandal2017; @stengel2020] captures spatial structure but suffers from documented *mean collapse* on rare extremes when trained with pixel-wise mean-squared error.

Existing open-source tools cover parts of this space. `MBC` and `downscaleR` (R) implement quantile-mapping families [@cannon2018] without spatial structure or deep-learning baselines. The few deep-learning downscaling codebases (`DeepSD`-derivatives, `ClimaX` adaptations) are research artefacts tied to a specific dataset, with neither a fixed evaluation protocol nor packaged statistical baselines on the same splits.

The community therefore lacks an **open, reproducible benchmark** for thermal super-resolution in complex terrain: existing publications report results on private splits and proprietary networks, making cross-paper comparison effectively impossible.

## Positioning relative to AI weather prediction frameworks

A second, fast-growing context is the rise of **AI weather prediction (AIWP)** models such as GraphCast [@graphcast], Pangu-Weather [@pangu], and FourCastNet [@fourcastnet], together with the open-source training and inference stack ANEMOI [@anemoi] around ECMWF's AIWP roadmap. These efforts operate *upstream*, at the global native-grid scale, and do not provide a last-mile terrain-aware sensor-supervised post-processing layer. Image-to-image super-resolution efforts in the same lineage (e.g., transformer-based hierarchical upscaling such as ArchesWeather-SR) address grid-to-grid upscaling but stop short of in-situ supervision and method-agnostic baselines. `karpos-downscaling` is **complementary**: it builds the downstream layer that connects an NWP or AIWP output to local, sensor-anchored applications.

Symmetrically, the framework doubles as a **data-generation pipeline for AIWP fine-tuning**. A key enabler is its **surface-state assimilation hook**, which ingests in-situ Sencrop observations into the SURFEX land-surface scheme [@masson2013] along the lines of LDAS-Monde sequential assimilation [@albergel2017]. This hook materialises a concrete sensor-anchored physical loop: a coarse-scale AIWP forecast can be downscaled, scored against in-situ truth, and used to update the surface boundary condition that the next forecast inherits. It is the building block needed by emerging **physical-loop fine-tuning** approaches (RLPF — Reinforcement Learning from a Physical Loop), adaptations of RLHF [@christiano2017] in which the human feedback signal is replaced by an instrumented physical loop, aimed at the global-to-local gap of large AIWP models.

`karpos-downscaling` addresses this gap with four design choices:

1. **Anti-leak temporal splits and seed-controlled training.** Train / development / pure-test splits are hash-fixed; every experiment runs five seeds by default, with reported means ± standard deviations and bootstrap confidence intervals.
2. **Automated leave-one-out cross-validation** along three axes — per station (spatial generalisation), per year (interannual robustness), per frost episode (out-of-distribution event robustness) — with a single Hydra override.
3. **Built-in statistical baselines** (MOS [@glahn1972], Kalman-filter [@galanis2002], QDM [@cannon2015] / MBCn [@cannon2018]) evaluated on the same splits, with McNemar paired-comparison tests and Holm-corrected p-values.
4. **Sencrop evaluation harness** with lapse-rate altitude correction between grid cell and sensor, stratified scoring by relief class (cuvette / slope / ridge) and by synoptic regime (radiative / advective).

The package targets meteorological and climate ML researchers working on super-resolution and bias correction in complex terrain, NWP post-processing teams, and agro-meteorology groups evaluating parametric-insurance indices. It is **method-agnostic**: alternative architectures (concatenation, cross-attention, diffusion-based super-resolution [@mardani2024]) plug into the same pipeline without rewriting data, splits, or metrics. The conditioning interface is similarly extensible: synoptic regime descriptors (cloud cover, low-level wind) and surface state from `SURFEX` (soil moisture, surface temperature, canopy temperature, vegetation type) can be added as parallel FiLM branches without retraining the base architecture.

# Example usage

A complete experiment — building tiles, training the FiLM·DEM U-Net, calibrating against Sencrop, and evaluating with leave-one-out — runs from a single configuration:

```bash
# 1. build training tiles from raw reanalysis + DEM
python -m karpos_downscaling.build_tiles dataset=baronnies

# 2. train stage A (downscaling) + stage C (in-situ calibration), 5 seeds
karpos-train experiment=film_mnt_residual seed=0,1,2,3,4

# 3. evaluate against Sencrop with leave-one-out by station
karpos-eval split=loo_station model=film_mnt_residual metric=pod_at_far20

# 4. run statistical baselines on the same splits for comparison
karpos-baseline method=qdm,mos,kf split=dev_held_out
```

All configurations are versioned in `configs/` and produce a self-describing run directory (model weights, metric tables, bootstrap confidence intervals, pairwise McNemar tests) that the companion paper [@maurin2026gmd] references directly.

# Method outline

The core model is a four-level U-Net with **FiLM modulation** $x \leftarrow \gamma(\mathrm{DEM}) \odot x + \beta(\mathrm{DEM})$ applied at every level. A small pyramidal DEM encoder produces spatially-varying $(\gamma_\ell, \beta_\ell)$ maps shared between encoder and decoder. The output is **residual**:

$$\hat{y}(x,y) = t_{2m}^{\mathrm{in}}(x,y) + \delta(x_{\mathrm{met}}, \mathrm{DEM})(x,y),$$

which prevents the mean-collapse pathology of absolute U-Nets [@stengel2020] and confines the network to the local thermal deviation. Supervision combines a reanalysis target (stage A) and a tail-weighted in-situ supervision against Sencrop minima (stage C), with optional growing-degree-day regularisation to bound the cumulative bias. The companion paper [@maurin2026gmd] gives the full architecture, empirical results on the Baronnies dataset, and ablations isolating the contribution of FiLM, residual formulation, and tail weighting. The implementation uses PyTorch, Hydra for configuration, and Zarr for tile storage; inference runs in seconds on a single GPU over the French metropolitan domain.

# Acknowledgements

This work was conducted in the context of the **Karpos** project (https://github.com/maurinl26/downscaling), targeting parametric frost insurance for French viticulture and arboriculture. The author thanks Sencrop for sensor network access, the Copernicus Climate Change Service for CERRA and ERA5-Land reanalyses, and the SURFEX and AROME development teams at Météo-France and CNRM. Collaborator acknowledgements will be added on submission.

# References
