# Changelog

All notable changes to `karpos-downscaling` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Forthcoming.

## [1.0.0] — 2026-06-22

Baronnies v1 — first public reference release of the calibrated U-Net.

### Added

- **Reference U-Net checkpoint** : residual U-Net with FiLM-conditioned terrain
  modulation (FiLM·DEM), 19,053,030 parameters across 246 tensors, supervised
  by Sencrop in-situ minima on the Baronnies domain (2022-2025 frost seasons).
- **Hub loader** (`downscaling/hub.py`) : loads released checkpoints by version
  tag from the model registry.
- **Release workflow** (`.github/workflows/release-to-hub.yml`) : packages and
  publishes a trained checkpoint + model card to the hub on tag creation.
- **Model card** : reproducibility envelope (data, hyperparameters, metrics)
  shipped alongside the released weights.
- **S3 persistence** for recalibrated outputs (`--out s3://`).
- **JOSS paper draft** : `docs/methodology/joss-paper-draft.md` + `paper.bib`
  bibliography (Issue #43 covered).
- **Zenodo DOI** : concept DOI `10.5281/zenodo.20783563` referenced in
  `CITATION.cff` and README badge.

### Fixed

- Smoke-test hardening from fresh clone (issue #48) — config defaults, missing
  imports, doc references.

## [0.3.0] — 2026-06-21

First public release with full JOSS-readiness scaffolding.

### Added

- **Community files** : `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor
  Covenant 2.1), `SECURITY.md`, `CITATION.cff`.
- **Documentation skeleton** (Sphinx + myst-parser + Read the Docs) :
  - Configuration `docs/conf.py` with `autodoc`, `napoleon`,
    `sphinx-rtd-theme`, `sphinxcontrib-bibtex`, MathJax, intersphinx, and
    extensive `autodoc_mock_imports` covering all heavy scientific
    dependencies (torch, scikit-learn, dask, elevation, etc.).
  - User-facing sections : Getting started (Installation, Quickstart,
    Examples), User guide (statistical pipeline, DL FiLM, Prithvi, regime
    stratification, calibration), Architecture, Infrastructure,
    Methodology, API reference, Community.
  - `docs/getting-started/installation.md` and
    `docs/getting-started/quickstart.md` written end-to-end (real
    walkthroughs with CDS + Sencrop + SRTM + multi-year examples).
  - `docs/community/docstring-conventions.md` codifying the NumPy style
    convention used across the codebase.
  - `docs/references.bib` with the foundational scientific references
    (Perez 2018 FiLM, Park 2022 Weather4Cast, Chapman 2024 RUFCO,
    Ronneberger 2015 U-Net, Luedeling 2021 PhenoFlex, Hersbach 2020
    ERA5, Proebsting & Mills 1978 frost thresholds, Dalhaus 2018 basis
    risk, CERRA 2022).
- **API reference** : Sphinx autodoc wired on `downscaling.shared.*`,
  `downscaling.deep_learning.*`, `downscaling.config`, `downscaling.paths`
  modules.
- **Read the Docs configuration** : `.readthedocs.yaml` with `uv`-based
  install of the `docs` extra only (no heavy scientific stack needed in
  the build environment).
- **Docstrings** : 100 % module coverage, 100 % public class coverage,
  ~53 % public function coverage (gap concentrated on PyTorch Lightning
  hooks and CLI `main()` entry points by convention).

### Changed

- `pyproject.toml` : added `[docs]` optional dependency group
  (`sphinx`, `myst-parser`, `sphinx-rtd-theme`, `sphinx-autodoc-typehints`,
  `sphinxcontrib-bibtex`, `linkify-it-py`).
- `downscaling/__init__.py` and `downscaling/scripts/__init__.py` : added
  package-level docstrings describing modules and CLI entry points.
- `downscaling/shared/loaders.py` : numpydoc-style docstrings on
  `CERRALoader.load_sl` and `CERRALoader.load_pl`.
- `downscaling/deep_learning/model.py` : expanded `build_model` docstring
  with full parameter and return type documentation.
- `docs/infra_pro.md` moved to `docs/infra/infra_pro.md`.

## [0.2.x] and earlier

Pre-release development versions. Not formally tagged. Refer to the git
history for details.
