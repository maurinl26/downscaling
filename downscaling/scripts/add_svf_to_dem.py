"""Compute the Sky View Factor (SVF) from a DEM and add it as a new variable.

SVF formula for a horizontal surface (Dozier & Frew 1990) :

    SVF = (1 / 2π) ∫_0^2π cos²(α_h(θ)) dθ

where ``α_h(θ)`` is the horizon angle above horizontal in azimuth θ. SVF = 1
on a peak or flat plain (full hemisphere visible), → 0 in a deep canyon.

This is the key topographic predictor for nighttime radiative cooling :
- low SVF (narrow valley, north-facing slope) → less radiative loss → less frost
- high SVF (peak, open plain) → strong radiative loss → more frost

The 1 km Drôme grid is small (167 × 118) so the brute-force horizon walk
runs in ~1 s, no need for tile-based parallelism.

Usage::

    uv run python -m downscaling.scripts.add_svf_to_dem \\
        --in /tmp/regen_cerra/dem_attributes.nc \\
        --out /tmp/regen_cerra/dem_attributes_svf.nc \\
        --pixel-m 1000 \\
        --n-dirs 16 \\
        --max-dist-px 25

The output NetCDF inherits all coords and variables of the input, plus a
``svf`` DataArray with the same (y, x) shape, dtype float32, range [0, 1].

Refs :
- Dozier & Frew (1990), IEEE TGARS — analytical SVF for tilted surfaces
- Issue downscaling #5 (item 4 : FiLM regime + topo features)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger("add_svf_to_dem")


def compute_svf(
    elev: np.ndarray,
    pixel_m: float = 1000.0,
    n_dirs: int = 16,
    max_dist_px: int = 25,
) -> np.ndarray:
    """Compute the Sky View Factor on a regular grid.

    Parameters
    ----------
    elev : (H, W) float array
        Elevation in meters.
    pixel_m : float
        Pixel size in meters (assumed isotropic).
    n_dirs : int
        Number of azimuth directions to sweep.
    max_dist_px : int
        Maximum distance (in pixels) to look for the horizon. ~25 km is
        usually more than enough for a 1 km grid in mid-latitude terrain.

    Returns
    -------
    svf : (H, W) float array
        SVF in [0, 1]. Border cells (within ``max_dist_px`` of the edge)
        use whatever horizon they have inside the grid.
    """
    H, W = elev.shape
    svf = np.zeros_like(elev, dtype=np.float32)
    yy, xx = np.indices((H, W))

    for k in range(n_dirs):
        theta = 2.0 * np.pi * k / n_dirs
        dy = -np.sin(theta)  # image y axis points south, sin(theta) east-up convention
        dx = np.cos(theta)
        max_alpha = np.full_like(elev, -np.inf, dtype=np.float32)
        for r in range(1, max_dist_px + 1):
            iy = (yy + r * dy).round().astype(int)
            ix = (xx + r * dx).round().astype(int)
            inside = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
            iy_c = np.clip(iy, 0, H - 1)
            ix_c = np.clip(ix, 0, W - 1)
            dh = elev[iy_c, ix_c] - elev
            d_m = r * pixel_m
            alpha = np.arctan(np.maximum(dh, 0.0) / d_m).astype(np.float32)
            alpha = np.where(inside, alpha, -np.inf)
            max_alpha = np.maximum(max_alpha, alpha)
        # Cells with no valid horizon → clear sky → alpha=0
        max_alpha = np.where(np.isfinite(max_alpha), max_alpha, 0.0).astype(np.float32)
        svf = svf + np.cos(max_alpha) ** 2

    return (svf / n_dirs).astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description="Compute SVF from a DEM and write a new NetCDF")
    p.add_argument("--in", dest="inp", type=Path, required=True, help="Input DEM NetCDF")
    p.add_argument("--out", type=Path, required=True, help="Output NetCDF with svf added")
    p.add_argument(
        "--elev-var", default="elevation", help="Elevation variable name (default: elevation)"
    )
    p.add_argument("--pixel-m", type=float, default=1000.0)
    p.add_argument("--n-dirs", type=int, default=16)
    p.add_argument("--max-dist-px", type=int, default=25)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ds = xr.open_dataset(args.inp)
    if args.elev_var not in ds:
        raise ValueError(f"Variable '{args.elev_var}' not in {args.inp}. Got: {list(ds.data_vars)}")

    elev = ds[args.elev_var].values.astype(np.float32)
    log.info(
        "Computing SVF on %dx%d grid (%d dirs × %d px max)",
        *elev.shape,
        args.n_dirs,
        args.max_dist_px,
    )
    svf = compute_svf(elev, pixel_m=args.pixel_m, n_dirs=args.n_dirs, max_dist_px=args.max_dist_px)
    log.info(
        "SVF range: [%.3f, %.3f], mean=%.3f", float(svf.min()), float(svf.max()), float(svf.mean())
    )

    # Preserve all input vars + coords, add svf
    out = ds.copy()
    out["svf"] = xr.DataArray(svf, dims=ds[args.elev_var].dims, coords=ds[args.elev_var].coords)
    out["svf"].attrs["long_name"] = "Sky View Factor"
    out["svf"].attrs["units"] = "fraction"
    out["svf"].attrs["description"] = (
        f"Sky View Factor (Dozier & Frew 1990), {args.n_dirs} azimuth directions, "
        f"max walk {args.max_dist_px} px ({args.max_dist_px * args.pixel_m / 1000:.0f} km)."
    )
    out.attrs["svf_added_by"] = "downscaling.scripts.add_svf_to_dem"
    out.attrs["pixel_m"] = args.pixel_m

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.out)
    log.info("Wrote %s with %d variables", args.out, len(out.data_vars))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
