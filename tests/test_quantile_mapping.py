"""Tests de la correction de biais par cartographie des quantiles.

(``downscaling.karpos_slr.quantile_mapping``)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscaling.karpos_slr.quantile_mapping import (
    EmpiricalQuantileMapping,
    QuantileDeltaMapping,
)


def _series(values, start="2000-01-01"):
    values = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=values.shape[0], freq="D")
    return xr.DataArray(values, dims=["time"], coords={"time": time}, name="t")


def test_eqm_identity_when_distributions_match():
    # modeled == observed → la fonction de transfert est l'identité.
    rng = np.random.default_rng(0)
    data = rng.normal(280.0, 5.0, size=400)
    modeled = _series(data)
    observed = _series(data.copy())
    eqm = EmpiricalQuantileMapping(n_quantiles=100, by_month=False)
    eqm.fit(modeled, observed)
    out = eqm.transform(modeled)
    np.testing.assert_allclose(out.values, modeled.values, atol=1e-6)
    assert out.attrs["bias_correction"] == "EQM"


def test_eqm_corrects_constant_bias():
    # observed = modeled + 3 K partout → EQM doit retirer ~3 K.
    rng = np.random.default_rng(1)
    base = rng.normal(280.0, 4.0, size=500)
    modeled = _series(base)
    observed = _series(base + 3.0)
    eqm = EmpiricalQuantileMapping(n_quantiles=200, by_month=False)
    eqm.fit(modeled, observed)
    out = eqm.transform(modeled)
    # La moyenne corrigée s'approche de la moyenne observée.
    np.testing.assert_allclose(float(out.mean()), float(observed.mean()), atol=0.2)


def test_eqm_transform_before_fit_raises():
    eqm = EmpiricalQuantileMapping(by_month=False)
    with pytest.raises(RuntimeError):
        eqm.transform(_series([1.0, 2.0, 3.0]))


def test_qdm_delta_identity_on_reference():
    # mod_ref == obs_ref, future == ref → QDM-delta ≈ identité.
    rng = np.random.default_rng(2)
    data = rng.normal(285.0, 3.0, size=400)
    ref = _series(data)
    qdm = QuantileDeltaMapping(kind="delta", n_quantiles=100, by_month=False)
    qdm.fit(ref, _series(data.copy()))
    out = qdm.transform(ref)
    np.testing.assert_allclose(out.values, ref.values, atol=1e-6)
    assert out.attrs["bias_correction"] == "QDM-delta"


def test_qdm_delta_corrects_bias_preserving_trend():
    # obs_ref = mod_ref - 2 K ; un futur décalé de +1 K doit garder son anomalie.
    rng = np.random.default_rng(3)
    base = rng.normal(280.0, 2.0, size=600)
    mod_ref = _series(base)
    obs_ref = _series(base - 2.0)
    qdm = QuantileDeltaMapping(kind="delta", n_quantiles=200, by_month=False)
    qdm.fit(mod_ref, obs_ref)
    future = _series(base + 1.0)
    out = qdm.transform(future)
    # Correction du biais (-2) tout en conservant la dérive (+1) → ~ -1 K net.
    np.testing.assert_allclose(float(out.mean() - future.mean()), -2.0, atol=0.3)


def test_qdm_ratio_keeps_precipitation_nonnegative():
    rng = np.random.default_rng(4)
    mod = np.abs(rng.gamma(2.0, 2.0, size=500))
    obs = mod * 1.5  # observations plus humides
    qdm = QuantileDeltaMapping(kind="ratio", n_quantiles=100, by_month=False, wet_threshold=0.1)
    qdm.fit(_series(mod), _series(obs))
    out = qdm.transform(_series(mod))
    assert bool((out.values >= 0.0).all())
    assert out.attrs["bias_correction"] == "QDM-ratio"
    # correction multiplicative → moyenne tirée vers le haut
    assert float(out.mean()) > float(_series(mod).mean())


def test_qdm_invalid_kind_raises():
    with pytest.raises(ValueError):
        QuantileDeltaMapping(kind="nope")


def test_qdm_transform_before_fit_raises():
    qdm = QuantileDeltaMapping(kind="delta", by_month=False)
    with pytest.raises(RuntimeError):
        qdm.transform(_series([1.0, 2.0]))
