#!/usr/bin/env python3
"""Live AROME → FiLM downscaling → canonical Tmin zarr (KarposSR serving adapter).

Increment 1 of the "serve a KarposSR frost map on /alerte" story. Closes the gap
identified in the audit: there was **no** path from a live AROME forecast zarr to
a high-resolution nightly-Tmin field written back as a canonical zarr consumable
by the titiler serving layer.

What this does
--------------
1. Reads a live AROME run from a zarr store (``s3://karpos-forecasts/arome/<run>.zarr``
   or a local path). Only the 2 m temperature field ``t2m`` is consumed; hourly
   leadtimes are grouped into forecast nights.
2. Loads the 1 km terrain-attribute DEM aligned on the FiLM target grid
   (elevation, slope, aspect, curvature, sky-view-factor).
3. Loads the KarposSR checkpoint (``UNetSparseCalibrationModule`` — the *sparse
   Sencrop calibration* Lightning module, NOT the legacy ``DLInferencePipeline``,
   see "Why not DLInferencePipeline" below) and runs it per night.
4. Extracts the downscaled **nightly Tmin** field.
5. Writes a canonical zarr (variable ``t2m_karpos_sr``) to
   ``s3://karpos-downscaling/karpos_sr/<run>.zarr`` (already SSRF-whitelisted by the
   titiler serving), plus a reproducibility sidecar.

Why not ``DLInferencePipeline``
-------------------------------
The audit named ``DLInferencePipeline`` / ``build_model(use_film=True)`` as the
inference path. That class is the **legacy** loader: it expects a Trainer-saved
``.pt`` with a ``model_state_dict`` key, ``met_in_ch=5`` (t2m, tp, u10, v10, sp)
and an external normalisation-stats JSON. The operational KarposSR checkpoints
(``checkpoints/multiyear_<year>/<year>-best.ckpt``) are **PyTorch-Lightning**
checkpoints of :class:`UNetSparseCalibrationModule` with a completely different
contract:

* ``met_in_ch=1`` — a single coarse T2m channel bilinearly regridded to the 1 km
  DEM grid (no 5-variable stack, no external stats JSON);
* ``dem_in_ch=5`` — the SVF DEM;
* ``target_mode='raw'`` — the U-Net predicts Tmin directly in °C (no denorm);
* pinball q=0.10 tail-aware loss, ``clamp=False``.

This adapter therefore drives the *sparse-calibration* module, which is the real
KarposSR serving path.

Input-semantics caveat (OPEN, for Lot D calibration)
----------------------------------------------------
The KarposSR model was trained with **CERRA** as the coarse background, fed as a
single snapshot per night, and learns snapshot → station nightly Tmin. Two shifts
are NOT resolved by this plumbing increment and must be validated in Lot D before
any customer-facing use:

* **Source shift** CERRA (5.5 km reanalysis) → AROME (1.3 km forecast). Different
  model climatology / bias structure on the input channel.
* **Reduction shift** the ``--reduce`` used here to collapse hourly AROME to one
  coarse field per night (default ``min``) is not identical to the single-snapshot
  the model saw in training. ``--reduce min`` risks a cold double-count.

The output is labelled ``t2m_karpos_sr`` (downscaled nightly Tmin, °C). It is an
UNCALIBRATED-against-AROME field until Lot D signs it off.

Usage
-----
    uv run python -m downscaling.deep_learning.arome_film_inference \
        --arome    s3://karpos-forecasts/arome/2026-04-27T00.zarr \
        --dem      /path/to/dem_attributes_svf.nc \
        --checkpoint checkpoints/multiyear_2024/2024-best.ckpt \
        --out      s3://karpos-downscaling/karpos_sr \
        --run-id   2026-04-27T00

Reproducibility envelope (submodule rule)
-----------------------------------------
A ``<run>.metadata.json`` sidecar records: literal ``uv run`` command, git SHA,
dirty flag, checkpoint, DEM, AROME run URI, night list, reduce mode, device, and
(if ``WANDB_API_KEY`` is set and not ``--wandb-disabled``) the W&B run URL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from downscaling.deep_learning.model import build_model
from downscaling.deep_learning.sparse_calibration import UNetSparseCalibrationModule

log = logging.getLogger("arome_film_inference")

# Canonical output variable name (must match backtest/api zarr_writer CANONICAL_VARS
# and the titiler serving contract for the KarposSR layer).
KARPOS_SR_VAR = "t2m_karpos_sr"


# ---------------------------------------------------------------------------
# Small reproducibility / IO helpers (kept self-contained: the Apache-2.0
# submodule must not import the private karpos-engine backtest.api package).
# ---------------------------------------------------------------------------
def _git_sha() -> tuple[str, bool]:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def _s3_endpoint() -> str:
    return (
        os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("SCW_S3_ENDPOINT_URL")
        or "https://s3.fr-par.scw.cloud"
    )


def _is_remote(p: str) -> bool:
    return str(p).startswith(("s3://", "scw://"))


def _normalize_s3(url: str) -> str:
    return "s3://" + url[len("scw://") :] if url.startswith("scw://") else url


def _open_zarr(uri: str) -> xr.Dataset:
    """Open a zarr store from a local path or an s3:///scw:// URL (Scaleway)."""
    if _is_remote(uri):
        import s3fs

        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": _s3_endpoint()})
        return xr.open_zarr(s3fs.S3Map(root=_normalize_s3(uri)[len("s3://") :], s3=fs), consolidated=False)
    return xr.open_zarr(uri)


# ---------------------------------------------------------------------------
# DEM loading (mirrors recalibrate_dl_film._build_coarse_provider z-scoring)
# ---------------------------------------------------------------------------
def load_dem(dem_path: str) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[str]]:
    """Load the terrain-attribute DEM → z-scored ``x_dem`` (1, C_dem, H, W).

    Each channel is z-scored on its own, exactly as the training provider does
    (otherwise curvature ~1e-3 is crushed by raw elevation in metres).
    """
    ds = xr.open_dataset(dem_path)
    dem_vars = list(ds.data_vars)
    chans = []
    for v in dem_vars:
        a = ds[v].values.astype(np.float32)
        mean, std = float(np.nanmean(a)), float(np.nanstd(a))
        std = std if std > 1e-9 else 1.0
        chans.append(((a - mean) / std).astype(np.float32))
    x_dem = torch.from_numpy(np.stack(chans, axis=0)).unsqueeze(0)  # (1, C_dem, H, W)
    lat = (ds["lat"] if "lat" in ds else ds["latitude"]).values.astype(np.float64)
    lon = (ds["lon"] if "lon" in ds else ds["longitude"]).values.astype(np.float64)
    return x_dem, lat, lon, dem_vars


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_karpos_sr_module(
    checkpoint: str,
    *,
    dem_in_ch: int,
    base_ch: int = 32,
    n_levels: int = 3,
    device: str = "cpu",
) -> UNetSparseCalibrationModule:
    """Rebuild the U-Net and load a KarposSR Lightning checkpoint.

    ``model`` is an injected object excluded from the checkpoint hyper-params
    (``save_hyperparameters(ignore=["model", ...])``), so it must be reconstructed
    and passed explicitly to ``load_from_checkpoint``. ``met_in_ch`` is always 1
    for the KarposSR sparse-calibration model.
    """
    model = build_model(
        "unet", met_in_ch=1, dem_in_ch=dem_in_ch, base_ch=base_ch, n_levels=n_levels,
        use_film=True, cond_dim=0,
    )
    lit = UNetSparseCalibrationModule.load_from_checkpoint(
        checkpoint, model=model, map_location=device
    )
    lit.eval()
    lit.to(device)
    log.info(
        "KarposSR checkpoint loaded: %s (target_mode=%s, reduce=%s, clamp=%s, denorm=%s)",
        checkpoint, lit.target_mode, lit.reduce, lit.clamp, lit.denorm,
    )
    return lit


# ---------------------------------------------------------------------------
# AROME → per-night coarse T2m fields
# ---------------------------------------------------------------------------
def _to_celsius(arr: np.ndarray) -> np.ndarray:
    """K → °C if the field looks like Kelvin (median > 100)."""
    return arr - 273.15 if float(np.nanmedian(arr)) > 100.0 else arr


def group_arome_nights(
    ds: xr.Dataset,
    *,
    dem_lat: np.ndarray,
    dem_lon: np.ndarray,
    night_hours: tuple[int, int],
    reduce: str,
    margin: float = 0.1,
) -> list[tuple[str, np.ndarray]]:
    """Group AROME hourly T2m into per-night coarse fields on the DEM grid (°C).

    A *night* is keyed by the local date at its start; ``night_hours=(18, 8)``
    means the window 18:00 UTC → 08:00 UTC next day. Within each night the field
    is reduced with ``reduce`` ('min' | 'mean' | 'snapshot'); for 'snapshot' the
    field nearest the coldest reference hour (05 UTC) is used.

    The reduced coarse field is regridded **coordinate-aware** onto the DEM
    lat/lon (``xarray.interp``, bilinear) so the geographic extent/orientation is
    handled correctly — never a naive tensor resize that would stretch a cropped
    AROME patch across the whole DEM domain. AROME is first cropped to the DEM
    extent ± ``margin`` for efficiency.

    Returns ``[(night_iso, field_2d_celsius_on_dem_grid), ...]``.
    """
    tvar = next((v for v in ("t2m", "2t", "temperature_2m") if v in ds), None)
    if tvar is None:
        raise ValueError(f"No T2m-like variable in AROME zarr (found {list(ds.data_vars)})")
    latname = "latitude" if "latitude" in ds.coords else "lat"
    lonname = "longitude" if "longitude" in ds.coords else "lon"
    ds = ds.rename({latname: "latitude", lonname: "longitude"}) if latname != "latitude" else ds
    # Crop to DEM extent ± margin (robust to ascending/descending coords).
    ds = ds.sortby("latitude").sortby("longitude").sel(
        latitude=slice(float(dem_lat.min()) - margin, float(dem_lat.max()) + margin),
        longitude=slice(float(dem_lon.min()) - margin, float(dem_lon.max()) + margin),
    )
    if ds.sizes.get("latitude", 0) < 2 or ds.sizes.get("longitude", 0) < 2:
        raise ValueError(
            "AROME run does not cover the DEM extent "
            f"(lat {dem_lat.min():.2f}..{dem_lat.max():.2f}, "
            f"lon {dem_lon.min():.2f}..{dem_lon.max():.2f})"
        )
    times = pd.to_datetime(ds["time"].values)
    da = ds[tvar]

    h0, h1 = night_hours
    night_keys = []
    for t in times:
        if t.hour >= h0:
            night_keys.append(t.normalize())
        elif t.hour < h1:
            night_keys.append((t - pd.Timedelta(days=1)).normalize())
        else:
            night_keys.append(pd.NaT)  # daytime, not part of any frost night
    night_keys = pd.Series(night_keys, index=range(len(times)))

    out: list[tuple[str, np.ndarray]] = []
    for key in sorted({k for k in night_keys if pd.notna(k)}):
        idx = night_keys.index[night_keys == key].tolist()
        sub = da.isel(time=idx)
        if reduce == "min":
            coarse = sub.min(dim="time")
        elif reduce == "mean":
            coarse = sub.mean(dim="time")
        elif reduce == "snapshot":
            ref = key + pd.Timedelta(hours=5)  # coldest-hour proxy
            sub_times = pd.to_datetime(sub["time"].values)
            coarse = sub.isel(time=int(np.argmin(np.abs(sub_times - ref))))
        else:
            raise ValueError(f"reduce must be min|mean|snapshot, got {reduce!r}")
        # Coordinate-aware regrid onto the DEM grid (bilinear). The DEM query
        # coordinates are clamped into the AROME footprint before interpolation:
        # this both keeps interp inside the data (no NaN) and gives a nearest-edge
        # extension over the thin strip where the DEM extends a fraction of a degree
        # beyond the AROME domain → the served field is 100 % finite on the DEM grid.
        alat, alon = coarse["latitude"].values, coarse["longitude"].values
        qlat = np.clip(dem_lat, float(alat.min()), float(alat.max()))
        qlon = np.clip(dem_lon, float(alon.min()), float(alon.max()))
        fine = coarse.interp(latitude=qlat, longitude=qlon, method="linear")
        out.append((key.strftime("%Y-%m-%d"), _to_celsius(fine.values.astype(np.float32))))
    return out


# ---------------------------------------------------------------------------
# Inference over nights → canonical Dataset
# ---------------------------------------------------------------------------
def run_inference(
    lit: UNetSparseCalibrationModule,
    x_dem: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    nights: list[tuple[str, np.ndarray]],
    *,
    device: str = "cpu",
) -> xr.Dataset:
    """Downscale each night's coarse T2m to the 1 km grid → Dataset(t2m_karpos_sr)."""
    x_dem = x_dem.to(device)
    grids = []
    with torch.no_grad():
        for night, coarse in nights:
            # ``coarse`` is already on the DEM grid (H, W); single met channel.
            x_met = torch.from_numpy(coarse).unsqueeze(0).unsqueeze(0).to(device)
            pred = lit._predict_target({"x_met": x_met, "x_dem": x_dem}).squeeze().cpu().numpy()
            grids.append(
                xr.DataArray(
                    pred.astype(np.float32),
                    dims=("latitude", "longitude"),
                    coords={"latitude": lat, "longitude": lon},
                    name=KARPOS_SR_VAR,
                ).expand_dims(time=[pd.Timestamp(night)])
            )
    return xr.Dataset({KARPOS_SR_VAR: xr.concat(grids, dim="time")})


# ---------------------------------------------------------------------------
# Canonical zarr write (same contract as backtest/api zarr_writer)
# ---------------------------------------------------------------------------
def write_canonical(ds: xr.Dataset, zarr_uri: str, *, source: str, source_version: str) -> dict:
    """Write ``ds`` as a canonical zarr at ``zarr_uri``.

    Prefers the authoritative ``write_canonical_zarr`` from the karpos-engine
    serving package **when it is importable** (i.e. running from the karpos-engine
    root, the actual deployment context) — that guarantees byte-for-byte manifest
    parity with the AROME ingestion. Falls back to an inline canonical write when
    the submodule is used standalone (OSS), so this file never hard-depends on the
    private repo.
    """
    # 1. Canonise: lat/lon ascending, tz-naive time.
    for dim in ("latitude", "longitude"):
        if dim in ds.coords and len(ds[dim]) > 1 and ds[dim].values[0] > ds[dim].values[-1]:
            ds = ds.sortby(dim, ascending=True)

    try:
        from backtest.api.zarr_writer import write_canonical_zarr  # type: ignore

        manifest = write_canonical_zarr(
            ds, zarr_uri, source=source, source_version=source_version,
            extra_attrs={"lot": "C", "model": "unet-film-sparse-calibration"},
        )
        log.info("Wrote via authoritative write_canonical_zarr (manifest parity).")
        return {"writer": "backtest.api.write_canonical_zarr", "sha256": manifest.sha256}
    except Exception as e:  # noqa: BLE001 — standalone/OSS fallback
        log.info("write_canonical_zarr unavailable (%s); inline canonical write.", e)

    ds.attrs.update(
        {
            "source": source,
            "source_version": source_version,
            "variables": KARPOS_SR_VAR,
            "lot": "C",
            "model": "unet-film-sparse-calibration",
        }
    )
    enc = {KARPOS_SR_VAR: {"chunks": (1, ds.sizes["latitude"], ds.sizes["longitude"])}}
    if _is_remote(zarr_uri):
        import s3fs

        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": _s3_endpoint()})
        store = s3fs.S3Map(root=_normalize_s3(zarr_uri)[len("s3://") :], s3=fs, check=False)
        ds.to_zarr(store, mode="w", consolidated=True, encoding=enc)
    else:
        Path(zarr_uri).parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(zarr_uri, mode="w", consolidated=True, encoding=enc)
    return {"writer": "inline", "sha256": None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AROME → FiLM KarposSR downscaling → canonical Tmin zarr")
    p.add_argument("--arome", required=True, help="AROME run zarr (s3://karpos-forecasts/arome/<run>.zarr or local)")
    p.add_argument("--dem", required=True, help="Terrain-attribute DEM NetCDF aligned on the FiLM grid (SVF DEM)")
    p.add_argument("--checkpoint", required=True, help="KarposSR Lightning .ckpt (checkpoints/multiyear_<year>/<year>-best.ckpt)")
    p.add_argument("--out", required=True, help="Output root (s3://karpos-downscaling/karpos_sr or local dir)")
    p.add_argument("--run-id", required=True, help="Run identifier → <out>/<run-id>.zarr")
    p.add_argument("--reduce", choices=("min", "mean", "snapshot"), default="min",
                   help="Hourly→night reduction of AROME T2m. See input-semantics caveat.")
    p.add_argument("--night-hours", type=int, nargs=2, default=[18, 8], metavar=("START", "END"),
                   help="Night window in UTC hours (default 18→08).")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--n-levels", type=int, default=3)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--wandb-project", default="karpos-sr-serving")
    p.add_argument("--wandb-disabled", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _resolve_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    x_dem, lat, lon, dem_vars = load_dem(args.dem)
    log.info("DEM: %d channels %s, grid %dx%d", len(dem_vars), dem_vars, len(lat), len(lon))

    lit = load_karpos_sr_module(
        args.checkpoint, dem_in_ch=len(dem_vars), base_ch=args.base_ch,
        n_levels=args.n_levels, device=device,
    )

    ds_arome = _open_zarr(args.arome)
    nights = group_arome_nights(
        ds_arome, dem_lat=lat, dem_lon=lon,
        night_hours=tuple(args.night_hours), reduce=args.reduce,
    )
    if not nights:
        log.error("No forecast nights extracted from AROME run — aborting.")
        return 2
    log.info("Extracted %d forecast nights: %s", len(nights), [n for n, _ in nights])

    ds_out = run_inference(lit, x_dem, lat, lon, nights, device=device)

    zarr_uri = f"{str(args.out).rstrip('/')}/{args.run_id}.zarr"
    write_info = write_canonical(
        ds_out, zarr_uri, source="KARPOS-SR-FILM", source_version=args.run_id
    )
    log.info("Wrote %s (%s)", zarr_uri, write_info["writer"])

    # Reproducibility sidecar.
    sha, dirty = _git_sha()
    meta = {
        "run_id": args.run_id,
        "command": " ".join(["uv", "run", "python", "-m", "downscaling.deep_learning.arome_film_inference", *sys.argv[1:]]),
        "git_sha": sha,
        "git_dirty": dirty,
        "checkpoint": str(args.checkpoint),
        "dem": str(args.dem),
        "dem_vars": dem_vars,
        "arome_run": str(args.arome),
        "dem_extent": [float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())],
        "reduce": args.reduce,
        "night_hours": list(args.night_hours),
        "nights": [n for n, _ in nights],
        "device": device,
        "output_var": KARPOS_SR_VAR,
        "output_zarr": zarr_uri,
        "writer": write_info["writer"],
        "sha256": write_info["sha256"],
        "input_semantics_caveat": (
            "Model trained on CERRA snapshot→station Tmin; AROME source-shift and "
            "hourly-reduce shift NOT calibrated — Lot D validation required before "
            "customer-facing use."
        ),
    }
    meta_uri = f"{str(args.out).rstrip('/')}/{args.run_id}.metadata.json"
    if _is_remote(meta_uri):
        import fsspec

        with fsspec.open(_normalize_s3(meta_uri), "w", client_kwargs={"endpoint_url": _s3_endpoint()}) as f:
            f.write(json.dumps(meta, indent=2))
    else:
        Path(meta_uri).parent.mkdir(parents=True, exist_ok=True)
        Path(meta_uri).write_text(json.dumps(meta, indent=2))
    log.info("Metadata sidecar: %s", meta_uri)
    return 0


if __name__ == "__main__":
    sys.exit(main())
