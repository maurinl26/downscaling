"""Tests de l'entry point Hydra ``downscaling-run`` (descente statistique).

Couvre la migration multi-fichiers : une saison de ``run_campaign`` passe la
liste de ses mensuels via ``data.era5_sl=[...]`` ; l'entry point exécute le
pipeline par fichier et concatène les sorties sur l'axe temporel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from downscaling.paths import CONFIG_DIR
from downscaling.scripts import run_downscaling as rd

# ---------------------------------------------------------------------------
# Helpers purs
# ---------------------------------------------------------------------------


def test_source_list_normalises_single_and_many():
    assert rd._source_list("a.nc") == ["a.nc"]
    assert rd._source_list(["a.nc", "b.nc"]) == ["a.nc", "b.nc"]
    # ListConfig (forme réelle issue d'un override Hydra) accepté.
    cfg = OmegaConf.create({"era5": ["a.nc", "b.nc"]})
    assert rd._source_list(cfg.era5) == ["a.nc", "b.nc"]


def test_time_dim_prefers_named_axis_then_falls_back():
    ds_time = xr.Dataset({"t2m": (("time", "y", "x"), np.zeros((2, 3, 3)))})
    assert rd._time_dim(ds_time) == "time"

    ds_valid = xr.Dataset({"t2m": (("valid_time", "y", "x"), np.zeros((2, 3, 3)))})
    assert rd._time_dim(ds_valid) == "valid_time"

    # Aucun nom canonique : on retombe sur la seule dim non spatiale.
    ds_other = xr.Dataset({"t2m": (("step", "lat", "lon"), np.zeros((2, 3, 3)))})
    assert rd._time_dim(ds_other) == "step"


# ---------------------------------------------------------------------------
# Wiring multi-fichiers de run_statistical
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Pipeline jouet : une sortie 1-pas par source, datée selon l'appel."""

    calls: list[str] = []

    def __init__(self, **kwargs):
        type(self).calls = []

    def calibrate(self, *a, **k):  # pragma: no cover - non sollicité ici
        raise AssertionError("calibrate ne doit pas être appelé sans obs_ref+mod_ref")

    def run(self, source, variables=None):
        type(self).calls.append(str(source))
        # Un pas de temps distinct par fichier → la concat doit en empiler N.
        t = pd.date_range("2021-04-01", periods=1, freq="D") + pd.Timedelta(
            days=len(self.calls) - 1
        )
        return xr.Dataset(
            {"t2m": (("time", "y", "x"), np.full((1, 2, 2), float(len(self.calls))))},
            coords={"time": t},
        )


def _cfg(tmp_path, sources):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=["experiment=drome_ardeche"])
    with open_dict(cfg):
        cfg.data.era5_sl = sources
        cfg.data.obs_ref = None  # pas de calibration QDM dans ce test
        cfg.run.out = str(tmp_path / "out.nc")
        cfg.run.compute_indices = False
    return cfg


def test_run_statistical_concatenates_multiple_files(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "StatisticalDownscalingPipeline", _FakePipeline)
    sources = ["s_avril.nc", "s_mai.nc", "s_juin.nc"]
    out = rd.run_statistical(_cfg(tmp_path, sources))

    assert out.exists()
    assert _FakePipeline.calls == sources  # un appel pipeline par mensuel
    ds = xr.open_dataset(out)
    assert ds.sizes["time"] == len(sources)  # sorties empilées temporellement


def test_run_statistical_single_file_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "StatisticalDownscalingPipeline", _FakePipeline)
    out = rd.run_statistical(_cfg(tmp_path, "solo.nc"))

    assert _FakePipeline.calls == ["solo.nc"]
    assert xr.open_dataset(out).sizes["time"] == 1
