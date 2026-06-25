# Contributing to karpos-downscaling

Thank you for considering contributing to **karpos-downscaling**, an open-source
Python package for atmospheric reanalysis downscaling (ERA5, CERRA) calibrated
on sparse in-situ sensor networks, with applications to parametric agricultural
insurance.

We welcome contributions from the climate science, machine learning,
agricultural engineering, and actuarial communities. This document describes
how to participate.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Code style and quality](#code-style-and-quality)
- [Submitting changes](#submitting-changes)
- [Commit message conventions](#commit-message-conventions)
- [Reporting bugs and requesting features](#reporting-bugs-and-requesting-features)
- [Scientific contributions](#scientific-contributions)
- [Contact](#contact)

## Code of conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report
unacceptable behavior to **loic.maurin@karpos.pro**.

## Ways to contribute

We welcome the following kinds of contributions:

- **Bug fixes** — corrections to existing pipelines or models
- **New downscaling methods** — additional statistical or deep-learning methods,
  ideally with a published reference
- **Validation studies** — POD/FAR/CSI/RMSE on new geographies, new crops, new
  reanalysis backbones
- **Documentation** — clarifications, tutorials, methodology references
- **Tests** — new unit tests, integration tests, regression tests on reference
  outputs
- **Calibration datasets** — annotated frost events, phenological observations,
  sparse sensor benchmarks (subject to licensing compatibility with Apache 2.0)
- **Performance** — speed-ups, memory reductions, GPU optimizations

If you are unsure whether a contribution fits the scope of the project, please
open an issue first to discuss.

## Development setup

This project uses [`uv`](https://docs.astral.sh/uv/) as its package and
environment manager. We recommend `uv` over `pip` / `conda` for reproducibility
and speed.

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone git@github.com:maurinl26/karpos-downscaling.git
cd downscaling
```

### 3. Install dependencies

The project exposes several **optional dependency groups** matching the
scientific stages:

| Group | Contents | When to install |
|---|---|---|
| `statistical` | `scikit-learn`, `elevation`, `dask` | Lapse-rate + QDM + RBF Sencrop pipeline |
| `dl` | `torch`, `pytorch-lightning`, `wandb` | U-Net FiLM deep-learning pipeline |
| `prithvi` | `prithviwxc` | NASA/IBM foundation model fine-tuning |
| `viz` | `matplotlib`, `cartopy`, `jupyter` | Plotting and notebooks |
| `regrid` | `xesmf` | Conservative regridding |
| `all` | everything above | Full development setup |

For most contributors:

```bash
# Minimal setup for statistical pipeline
uv sync --extra statistical

# Full development setup
uv sync --extra all
```

### 4. Verify the installation

```bash
uv run pytest --collect-only -q | head -20
```

You should see a list of collected tests.

## Running tests

Tests live in `tests/` and use `pytest` with coverage reporting.

```bash
# Run the full test suite
uv run pytest

# Run a specific test file
uv run pytest tests/test_film_regime_conditioning.py

# Run with verbose output and coverage
uv run pytest -v --cov=. --cov-report=term-missing
```

Tests that depend on optional extras (DL, Prithvi) use `pytest.importorskip(...)`
to skip gracefully when dependencies are absent — so a minimal install can still
run the relevant subset of tests.

Continuous integration runs on every PR via GitHub Actions (`.github/workflows/tests.yml`).
Please ensure your changes pass CI before requesting review.

## Code style and quality

We enforce a consistent style via [`ruff`](https://docs.astral.sh/ruff/) and
[`mypy`](https://mypy-lang.org/).

### Linting

```bash
# Check for style issues
uv run ruff check .

# Auto-fix where possible
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type checking

```bash
uv run mypy .
```

Note: typing is **progressively rolled out**. Functions without annotations are
not yet checked. Please type-annotate any new function signatures you add. The
goal is to enable `check_untyped_defs = true` per module over time.

### Configuration

- Line length: **100 characters**
- Python target: **3.11+**
- Rules enabled: `E`, `F`, `W`, `I`, `UP`, `B` (see `pyproject.toml` for
  per-file overrides)

## Submitting changes

### 1. Open an issue first

For non-trivial changes, please **open an issue first** to discuss the
approach. This helps avoid duplicated effort and ensures alignment with the
project's direction.

### 2. Create a branch

Branch names follow this convention:

- `feat/<short-description>` — new feature
- `fix/<short-description>` — bug fix
- `chore/<short-description>` — maintenance, tooling, refactor
- `docs/<short-description>` — documentation only

Example: `feat/qdm-loo-validation`, `fix/cerra-orography-units`.

### 3. Make your changes

- Write or update tests covering your change
- Update documentation (README, docstrings, `docs/methodology/` if relevant)
- Run the test suite locally before pushing

### 4. Open a pull request

- Target the `main` branch
- Use a descriptive title and reference any related issue (e.g. `Closes #42`)
- In the description, explain **what** you changed and **why**
- For scientific changes, include a brief description of the validation
  (POD/FAR/CSI, RMSE, comparison to baseline)
- Be patient — review may take a few days

### 5. Review process

- A maintainer will review your PR and may request changes
- All comments must be resolved before merge
- CI must pass (tests + linting)
- For scientific contributions, a second reviewer may be asked to validate the
  methodology

## Commit message conventions

We use a simplified [Conventional Commits](https://www.conventionalcommits.org/)
style:

```
<type>(<scope>): <short imperative summary>

<optional body explaining what and why, not how>

<optional footer with issue references or co-authors>
```

Types:

- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation only
- `chore` — tooling, build, dependencies
- `refactor` — code change that neither fixes nor adds a feature
- `test` — adding or fixing tests
- `perf` — performance improvement

Examples:

```
feat(dl-film): regime conditioning via FiLM layers in U-Net encoder

Closes #41

Co-Authored-By: ...
```

```
fix(stat): correct +10°C bias from CERRA orography unit (K to °C)

The bias was traced to the orography file being interpreted in Kelvin
rather than meters. This fix explicitly converts and adds a regression test.

Closes #18
```

## Reporting bugs and requesting features

### Bugs

Please open an issue with:

1. A clear, descriptive title
2. Your environment: OS, Python version, `uv` version, relevant optional
   extras installed
3. Minimal reproduction steps
4. Expected vs actual behavior
5. Full error traceback if applicable

### Feature requests

Please open an issue with:

1. A clear description of the desired functionality
2. The use case or scientific motivation
3. References to relevant publications if applicable
4. Whether you are willing to contribute the implementation

## Scientific contributions

This project is research-grade open-source software. We particularly welcome:

- **New benchmark datasets** for downscaling validation (sparse sensor networks,
  phenological observations, frost event catalogs)
- **Reproducibility studies** of published downscaling methods on our test
  geographies (Drôme, Vallée du Rhône, others)
- **Cross-validation analyses** that strengthen the statistical foundations of
  our calibration approach
- **Documentation of failure modes** — regimes, regions, or conditions where
  current methods underperform, with diagnostic analyses

For scientific contributions, please include in your PR:

- A brief methodology description (or link to a published reference)
- The validation protocol (train/dev/test splits, cross-validation, hold-outs)
- The metrics used and their values vs a clear baseline
- Any data, code, or trained weights necessary to reproduce the results

If your contribution leads to a publication, please cite this software using
the metadata in `CITATION.cff` (forthcoming for the JOSS paper).

## Contact

- **Project maintainer**: Loïc Maurin, [loic.maurin@karpos.pro](mailto:loic.maurin@karpos.pro)
- **Issues and feature requests**: [GitHub issue tracker](https://github.com/maurinl26/karpos-downscaling/issues)
- **Pull requests**: [GitHub PRs](https://github.com/maurinl26/karpos-downscaling/pulls)

For discussion of methodology, results, or research collaborations, you are
welcome to reach out by email.

---

Thank you for contributing to open-source climate science.
