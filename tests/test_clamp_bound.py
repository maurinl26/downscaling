"""#81 : la tête bornée doit garder la sortie DL dans l'enveloppe [lo, hi]."""

import torch
import torch.nn as nn

from downscaling.deep_learning.sparse_calibration import UNetSparseCalibrationModule


class _BigModel(nn.Module):
    """Renvoie des valeurs énormes (±1000) : la borne doit tenir malgré tout."""

    def __init__(self, h, w):
        super().__init__()
        self.h, self.w = h, w
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x_met, x_dem, *a):
        b = x_met.shape[0]
        return torch.randn(b, 1, self.h, self.w) * 1000.0 + self.p


def _batch(h, w, seed=0):
    torch.manual_seed(seed)
    cerra = torch.rand(h, w) * 10 - 5  # [-5, 5] °C
    surfex = torch.rand(h, w) * 10 - 8  # [-8, 2] °C
    x_env = torch.stack([cerra, surfex], 0).unsqueeze(0)  # (1, 2, H, W)
    batch = {
        "x_met": torch.randn(1, 1, h, w),
        "x_dem": torch.randn(1, 3, h, w),
        "x_env": x_env,
    }
    lo = torch.minimum(cerra, surfex)
    hi = torch.maximum(cerra, surfex)
    return batch, lo, hi


def test_output_strictly_inside_envelope():
    h, w = 16, 12
    mod = UNetSparseCalibrationModule(
        _BigModel(h, w), clamp=True, clamp_margin=0.0, kelvin_to_celsius=False
    )
    batch, lo, hi = _batch(h, w)
    for _ in range(100):  # sortie CNN aléatoire à chaque passe
        pred = mod._predict_target(batch)[0, 0]
        assert bool((pred >= lo - 1e-4).all()), "sortie sous la borne basse"
        assert bool((pred <= hi + 1e-4).all()), "sortie au-dessus de la borne haute"


def test_margin_widens_envelope():
    h, w = 8, 8
    m = 1.5
    mod = UNetSparseCalibrationModule(
        _BigModel(h, w), clamp=True, clamp_margin=m, kelvin_to_celsius=False
    )
    batch, lo, hi = _batch(h, w, seed=3)
    pred = mod._predict_target(batch)[0, 0]
    assert bool((pred >= lo - m - 1e-4).all())
    assert bool((pred <= hi + m + 1e-4).all())


def test_bounded_head_is_differentiable():
    h, w = 8, 8
    mod = UNetSparseCalibrationModule(_BigModel(h, w), clamp=True, kelvin_to_celsius=False)
    batch, _, _ = _batch(h, w, seed=1)
    mod.zero_grad()
    mod._predict_target(batch).sum().backward()
    assert mod.model.p.grad is not None
