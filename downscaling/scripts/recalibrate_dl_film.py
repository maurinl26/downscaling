#!/usr/bin/env python3
"""DL FiLM recalibration with sparse Sencrop calibration.

Known internally as the "Lot C" deliverable of the Sencrop S23 campaign. Thin
orchestrator that wires:
- `build_model("unet", use_film=True)` from `downscaling.deep_learning.model`
- `UNetSparseCalibrationModule` from `downscaling.deep_learning.sparse_calibration`
- A bulk-aware Sencrop dataset (uses the patched `load_sencrop` from
  `downscaling.prtihvi_wxc.sencrop` which auto-detects bulk roots)

It does NOT re-implement training or losses — that lives in the existing
Lightning module.

Inputs
------

    --year YYYY                     # target year
    --cerra-atm   <NetCDF>           # CERRA atm for the year (T2m + extras)
    --dem         <NetCDF/GeoTIFF>   # high-res DEM (1 km)
    --sencrop     <bulk root>        # local path or s3://... (the patched
                                       loader auto-detects)
    --out         <dir>              # Zarr output dir for the year
    --epochs      <int>              # default 30
    --wandb-project KARPOS_LOT_C     # default; --wandb-disabled to skip
    --device      cuda|cpu           # default cuda

Output
------

    <out>/<year>.zarr                # 1 km recalibrated nightly Tmin grid
    <out>/<year>.metadata.json       # reproducibility envelope
    <out>/<year>.ckpt                # best checkpoint (Lightning)

Reproducibility envelope
------------------------

The JSON metadata records: `uv run` command, git SHA, W&B run URL, inputs
paths, n_nights trained, n_stations per night avg.

Caveats
-------

- Heavy: GPU recommended (cuda). On CPU only viable for smoke testing.
- The Dataset is built around the bulk Sencrop loader — the legacy
  `UNetStationDataset` (which expects `sencrop_<date>.csv` per-day files)
  is bypassed.
- W&B is only logged if not `--wandb-disabled` AND the `WANDB_API_KEY`
  env var is set on the host.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from downscaling.deep_learning.model import build_model
from downscaling.deep_learning.sparse_calibration import (
    UNetSparseCalibrationModule,
    UNetSparseDataModule,
    unet_sparse_collate,  # noqa: F401  (re-exported for users that want manual loops)
)
from downscaling.prtihvi_wxc.netatmo_qc import NetatmoNocturnalQC
from downscaling.prtihvi_wxc.sencrop import load_sencrop
from downscaling.prtihvi_wxc.stations import night_station_targets
from downscaling.utils.io import describe, is_remote, make_zarr_store, write_sidecar

log = logging.getLogger("recalibrate_dl_film")


# ---------------------------------------------------------------------------
# Bulk-aware Sencrop Dataset (replaces UNetStationDataset for bulk roots)
# ---------------------------------------------------------------------------
@dataclass
class _NightSample:
    date: str
    x_met: torch.Tensor
    x_dem: torch.Tensor
    obs_tmin: torch.Tensor
    obs_row: torch.Tensor
    obs_col: torch.Tensor
    obs_dz: torch.Tensor


_REGIME_LABELS = ["R1", "R2", "R3", "R4a", "R4b"]   # ordre canonique (R0 = unknown, ignoré)
_ERA5_FEATURES = ["wind_med", "tcc_med", "mslp_med", "dewpoint_dep_med"]


def _build_cond_table(regimes_csv: str | Path, cond_vars: set[str]) -> tuple[dict[str, np.ndarray], int]:
    """Construit ``{date_iso: cond_vec}`` à partir de ``regimes_<year>.csv``.

    ``cond_vars`` sous-ensemble de {'regime', 'era5', 'season'}.
    Retourne la table indexée par 'YYYY-MM-DD' et la dimension du vecteur.
    """
    cond_vars = set(cond_vars)
    df = pd.read_csv(regimes_csv)
    # Date column tolerant
    date_col = next((c for c in ("date", "night_date", "valid_date") if c in df.columns), df.columns[0])
    table: dict[str, np.ndarray] = {}
    # Normalisation ERA5 (z-score sur l'ensemble du fichier — stable cross-année)
    era5_means = {f: df[f].mean() for f in _ERA5_FEATURES if f in df.columns}
    era5_stds = {f: df[f].std() if df[f].std() > 1e-6 else 1.0 for f in _ERA5_FEATURES if f in df.columns}
    for _, row in df.iterrows():
        parts: list[float] = []
        if "regime" in cond_vars:
            reg = str(row.get("regime", "R0"))
            parts.extend([1.0 if reg == lbl else 0.0 for lbl in _REGIME_LABELS])
        if "era5" in cond_vars:
            for f in _ERA5_FEATURES:
                if f in df.columns:
                    val = (row[f] - era5_means[f]) / era5_stds[f] if pd.notna(row[f]) else 0.0
                    parts.append(float(val))
        if "season" in cond_vars:
            try:
                dt = pd.to_datetime(row[date_col])
                doy = dt.dayofyear
                parts.append(float(np.sin(2 * np.pi * doy / 365.25)))
                parts.append(float(np.cos(2 * np.pi * doy / 365.25)))
            except Exception:
                parts.extend([0.0, 0.0])
        d_iso = pd.to_datetime(row[date_col]).strftime("%Y-%m-%d")
        table[d_iso] = np.array(parts, dtype=np.float32)
    cond_dim = len(next(iter(table.values()))) if table else 0
    return table, cond_dim


class BulkSencropDataset(Dataset):
    """Sample = 1 nuit. Lit Sencrop depuis le bulk root (URI s3:// possible).

    Si ``cond_table`` est fourni (cf. ``_build_cond_table``), chaque sample
    expose un ``cond_vec`` (régime + ERA5 + saisonnalité) consommé par
    ``DownscalingUNet`` via FiLM.
    """

    def __init__(
        self,
        dates: list[str],
        coarse_provider,           # callable date -> (x_met, x_dem)
        sencrop_root: str | Path,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        elevation_grid: np.ndarray | None = None,
        min_stations: int = 5,
        lapse_rate: float = -6.5e-3,
        cond_table: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.coarse_provider = coarse_provider
        self.sencrop_root = sencrop_root
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.elevation_grid = elevation_grid
        self.min_stations = min_stations
        self.qc = NetatmoNocturnalQC(lapse_rate=lapse_rate)
        self.cond_table = cond_table

        # Pre-filter to dates where the bulk has enough stations.
        kept = []
        for d in dates:
            try:
                obs = load_sencrop(sencrop_root, d)
                if obs.station_id.size >= min_stations:
                    kept.append(d)
            except (ValueError, FileNotFoundError):
                continue
        self.dates = kept
        log.info("Dataset: %d/%d nights kept", len(kept), len(dates))
        if cond_table is not None:
            missing = [d for d in kept if d not in cond_table]
            if missing:
                log.warning("cond_table missing for %d/%d nights — will use zero vector", len(missing), len(kept))

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict:
        d = self.dates[idx]
        x_met, x_dem = self.coarse_provider(d)
        obs_qc = self.qc.run(load_sencrop(self.sencrop_root, d))
        tmin, row, col, dz = night_station_targets(
            obs_qc, self.lat_grid, self.lon_grid, self.elevation_grid
        )
        sample = {
            "x_met": x_met,
            "x_dem": x_dem,
            "obs_tmin": torch.as_tensor(tmin, dtype=torch.float32),
            "obs_row": torch.as_tensor(row, dtype=torch.long),
            "obs_col": torch.as_tensor(col, dtype=torch.long),
            "obs_dz": torch.as_tensor(dz, dtype=torch.float32),
            "date": d,
        }
        if self.cond_table is not None:
            cv = self.cond_table.get(d)
            if cv is None:
                # fallback : zero vector with the right dim
                cv = np.zeros(len(next(iter(self.cond_table.values()))), dtype=np.float32)
            sample["cond_vec"] = torch.as_tensor(cv, dtype=torch.float32)
        return sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def _build_coarse_provider(cerra_path: Path, dem_path: Path):
    """Loads CERRA atm + DEM once, returns a callable date -> (x_met, x_dem).

    Gère le mismatch CERRA NetCDF : variable temporelle 'valid_time' au lieu de
    'time', t2m en Kelvin. Convertit en Celsius côté provider pour cohérence
    avec le module de calibration (kelvin_to_celsius=False).
    """
    ds_cerra = xr.open_dataset(cerra_path)
    ds_dem = xr.open_dataset(dem_path)

    # Normalise time coord (CERRA utilise valid_time)
    if "valid_time" in ds_cerra.dims and "time" not in ds_cerra.dims:
        ds_cerra = ds_cerra.rename({"valid_time": "time"})
    elif "valid_time" in ds_cerra.coords and "time" not in ds_cerra.coords:
        ds_cerra = ds_cerra.rename({"valid_time": "time"})
    if ds_cerra["time"].dtype.kind != "M":
        ref = pd.Timestamp("1900-01-01")
        ds_cerra = ds_cerra.assign_coords(
            time=ref + pd.to_timedelta(ds_cerra["time"].values, unit="h")
        )

    # Identify temperature variable
    t_var = next(
        (v for v in ("t2m", "2t", "temperature_2m") if v in ds_cerra),
        None,
    )
    if t_var is None:
        raise ValueError(f"No T2m-like variable in {cerra_path}")

    # Détecte unités K et convertit en °C une seule fois.
    src_units = str(ds_cerra[t_var].attrs.get("units", "")).lower()
    sample = float(np.nanmedian(ds_cerra[t_var].values))
    is_kelvin = src_units in ("k", "kelvin") or sample > 100
    if is_kelvin:
        log.info("CERRA t2m en Kelvin (median=%.1f K), conversion → Celsius", sample)
        ds_cerra[t_var] = ds_cerra[t_var] - 273.15
        ds_cerra[t_var].attrs["units"] = "degC"

    # Provider retourne des tenseurs (C, H, W). Le DataLoader ajoute la dim batch
    # → (B, C, H, W) attendu par Conv2d. En inference directe (hors DataLoader),
    # il faut unsqueeze(0) manuellement (cf. boucle d'inférence ci-dessous).
    # CERRA est sur grille 5 km (~31×31 sur Drôme-Ardèche), DEM sur grille 1 km
    # (~167×118). On regridde x_met vers la résolution DEM par bilinéaire pour
    # que les deux channels partagent la même grille, indispensable au U-Net.
    import torch.nn.functional as F
    dem_arr = ds_dem[next(iter(ds_dem.data_vars))].values.astype(np.float32)
    H_fine, W_fine = dem_arr.shape
    x_dem = torch.from_numpy(dem_arr).unsqueeze(0)  # (1, H, W)
    log.info("Provider grids: DEM (%d, %d) | CERRA t2m yearly %s",
             H_fine, W_fine, tuple(ds_cerra[t_var].shape))

    def provider(d: str) -> tuple[torch.Tensor, torch.Tensor]:
        slab = ds_cerra[t_var].sel(time=d, method="nearest")
        arr = slab.values.astype(np.float32)
        # (H_coarse, W_coarse) → bilinear → (H_fine, W_fine)
        coarse = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, Hc, Wc)
        fine = F.interpolate(coarse, size=(H_fine, W_fine), mode="bilinear", align_corners=False)
        x_met = fine.squeeze(0)  # (1, H_fine, W_fine)
        return x_met, x_dem

    return provider, ds_cerra, ds_dem


def _dates_in_year(ds_cerra: xr.Dataset, year: int) -> list[str]:
    """Retourne les dates uniques de l'année (CERRA = 8 timesteps/jour, on dédup)."""
    # Le provider a déjà renommé valid_time → time.
    time_var = "time" if "time" in ds_cerra else "valid_time"
    times = pd.to_datetime(ds_cerra[time_var].values)
    return sorted({t.strftime("%Y-%m-%d") for t in times if t.year == year})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="DL FiLM recalibration with sparse Sencrop")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--cerra-atm", type=Path, required=True)
    p.add_argument("--dem", type=Path, required=True)
    p.add_argument("--sencrop", type=str, required=True, help="bulk root (local or s3://)")
    p.add_argument("--out", type=str, required=True,
                   help="Output target: local dir OR s3:// / scw:// URL "
                        "(zarr + sidecar metadata.json written there). "
                        "Lightning checkpoints stay local (--checkpoint-dir).")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Local dir for Lightning ModelCheckpoint. "
                        "Default: <out> if local, else ./checkpoints/<year>.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="auto",
                   choices=("auto", "cuda", "mps", "cpu"),
                   help="Accelerator. 'auto' = MPS si Apple Silicon, sinon CUDA, sinon CPU.")
    p.add_argument("--base-ch", type=int, default=32,
                   help="U-Net base channels (capacity). Issue #28: 64 ≈ 4.6M params.")
    p.add_argument("--n-levels", type=int, default=3,
                   help="U-Net depth. Issue #28: 4 augmente receptive field.")
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="If >0, EarlyStopping on val/rmse with this patience (issue #28).")
    p.add_argument("--loss-quantile", type=float, default=None,
                   help="If set in (0,1), use pinball/quantile loss instead of RMSE on station "
                        "observations. q<0.5 penalizes under-prediction of cold (missed frost) "
                        "more than over-prediction. Recommended for frost detection: q=0.1 "
                        "(misses cost ×9 false alarms). Issue #5.")
    p.add_argument("--cond-vars", type=str, default="",
                   help="Comma-separated FiLM conditioning vars: regime, era5, season. "
                        "Empty = DEM-only (baseline). Requires --regimes-csv. Issue #5 item 4.")
    p.add_argument("--regimes-csv", type=str, default=None,
                   help="Path to regimes_all.csv (output of flag_regimes.py). "
                        "Required when --cond-vars is non-empty.")
    p.add_argument("--wandb-project", default="karpos-recalibrate-dl-film")
    p.add_argument("--wandb-disabled", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="run 1 epoch on a tiny subset (CPU OK)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Resolve checkpoint dir : local default if --out is remote
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    elif is_remote(args.out):
        ckpt_dir = Path("./checkpoints") / str(args.year)
    else:
        ckpt_dir = Path(args.out)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Coarse provider + dates --------------------------------------------
    provider, ds_cerra, ds_dem = _build_coarse_provider(args.cerra_atm, args.dem)
    dates = _dates_in_year(ds_cerra, args.year)
    log.info("Year=%d, %d candidate dates", args.year, len(dates))

    lat_grid = ds_dem["lat"].values if "lat" in ds_dem else ds_dem["latitude"].values
    lon_grid = ds_dem["lon"].values if "lon" in ds_dem else ds_dem["longitude"].values
    dem_2d = ds_dem[next(iter(ds_dem.data_vars))].values.astype(np.float32)

    # ---- FiLM conditioning vars (régime + ERA5 + saisonnalité) --------------
    cond_vars = {v.strip() for v in args.cond_vars.split(",") if v.strip()}
    cond_table = None
    cond_dim = 0
    if cond_vars:
        if not args.regimes_csv:
            raise ValueError("--cond-vars requires --regimes-csv")
        cond_table, cond_dim = _build_cond_table(args.regimes_csv, cond_vars)
        log.info("FiLM conditioning : vars=%s, dim=%d", sorted(cond_vars), cond_dim)
    else:
        log.info("FiLM conditioning : DEM-only (baseline)")

    # ---- Dataset + DataModule -----------------------------------------------
    dataset = BulkSencropDataset(
        dates=dates,
        coarse_provider=provider,
        sencrop_root=args.sencrop,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        elevation_grid=dem_2d,
        min_stations=5,
        cond_table=cond_table,
    )
    if len(dataset) == 0:
        log.error("No valid nights found — aborting")
        return 2

    if args.smoke_test:
        log.info("Smoke test: subsampling dataset to 8 nights, epochs=1")
        dataset.dates = dataset.dates[:8]
        args.epochs = 1

    datamodule = UNetSparseDataModule(dataset, num_workers=0)

    # ---- Model + LightningModule --------------------------------------------
    model = build_model(
        "unet", met_in_ch=1, dem_in_ch=1,
        base_ch=args.base_ch, n_levels=args.n_levels, use_film=True,
        cond_dim=cond_dim,
    )
    log.info("U-Net: base_ch=%d, n_levels=%d, cond_dim=%d", args.base_ch, args.n_levels, cond_dim)
    lit = UNetSparseCalibrationModule(
        model=model,
        target_channel=0,
        lr=1e-4,
        warmup_epochs=2,
        max_epochs=args.epochs,
        loss_quantile=args.loss_quantile,
        kelvin_to_celsius=False,  # CERRA atm is already °C in our pipeline
        elevation_aware=True,
        hourly=False,
        reduce="min",
    )
    if args.loss_quantile is not None:
        log.info("Loss : pinball quantile q=%.2f (tail-aware, issue #5)", args.loss_quantile)
    else:
        log.info("Loss : RMSE (symmetric, baseline)")

    # ---- Trainer ------------------------------------------------------------
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir), filename=f"{args.year}-best",
            monitor="val/rmse", mode="min", save_top_k=1,
        )
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            monitor="val/rmse", mode="min",
            patience=args.early_stopping_patience, min_delta=1e-3,
        ))
        log.info("EarlyStopping enabled: patience=%d", args.early_stopping_patience)

    logger = False
    wandb_run_url = None
    if not args.wandb_disabled and os.environ.get("WANDB_API_KEY"):
        try:
            from lightning.pytorch.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                name=f"recalibrate_dl_film_{args.year}",
                tags=["recalibrate", "dl-film", "sencrop", f"year-{args.year}"],
            )
            logger = wandb_logger
            wandb_run_url = getattr(wandb_logger.experiment, "url", None)
        except ImportError:
            log.warning("wandb not installed; running without it")

    # Resolve accelerator (auto = MPS > CUDA > CPU).
    def _resolve_accelerator(requested: str) -> str:
        has_cuda = torch.cuda.is_available()
        has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        if requested == "auto":
            return "mps" if has_mps else ("gpu" if has_cuda else "cpu")
        if requested == "mps":
            if not has_mps:
                log.warning("MPS demandé mais indisponible, fallback CPU")
                return "cpu"
            return "mps"
        if requested == "cuda":
            if not has_cuda:
                log.warning("CUDA demandé mais indisponible, fallback CPU")
                return "cpu"
            return "gpu"
        return "cpu"

    accelerator = _resolve_accelerator(args.device)
    log.info("Lightning accelerator=%s (--device %s)", accelerator, args.device)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    log.info("Training: %d epochs on %d nights", args.epochs, len(dataset))
    trainer.fit(lit, datamodule=datamodule)

    # ---- Inference loop: 1 km grid Tmin per night → Zarr --------------------
    log.info("Inference: producing 1 km grid for %d nights", len(dataset))
    lit.eval()
    inference_device = next(lit.parameters()).device
    out_grids = []
    with torch.no_grad():
        for d in dataset.dates:
            x_met, x_dem = provider(d)
            # provider retourne (C, H, W) ; le modèle attend (B, C, H, W)
            batch = {
                "x_met": x_met.unsqueeze(0).to(inference_device),
                "x_dem": x_dem.unsqueeze(0).to(inference_device),
            }
            if cond_table is not None:
                cv = cond_table.get(d)
                if cv is None:
                    cv = np.zeros(cond_dim, dtype=np.float32)
                batch["cond_vec"] = torch.as_tensor(cv, dtype=torch.float32).unsqueeze(0).to(inference_device)
            pred = lit._predict_target(batch).squeeze().cpu().numpy()
            slab = xr.DataArray(
                pred, dims=("latitude", "longitude"),
                coords={"latitude": lat_grid, "longitude": lon_grid},
            ).expand_dims(time=[pd.Timestamp(d)])
            out_grids.append(slab)

    out_ds = xr.concat(out_grids, dim="time")
    zarr_store = make_zarr_store(args.out, args.year)
    out_ds.to_zarr(zarr_store, mode="w")
    log.info("Wrote %s (%d nights)", describe(args.out, args.year, ".zarr"), len(out_grids))

    # ---- Reproducibility metadata -------------------------------------------
    metadata = {
        "year": args.year,
        "command": " ".join(["uv", "run", "python", *sys.argv]),
        "git_sha": _git_sha(),
        "cerra_atm": str(args.cerra_atm),
        "dem": str(args.dem),
        "sencrop_root": str(args.sencrop),
        "epochs": args.epochs,
        "device": args.device,
        "base_ch": args.base_ch,
        "n_levels": args.n_levels,
        "loss": "pinball" if args.loss_quantile is not None else "rmse",
        "loss_quantile": args.loss_quantile,
        "early_stopping_patience": args.early_stopping_patience,
        "cond_vars": sorted(cond_vars) if cond_vars else [],
        "cond_dim": cond_dim,
        "n_nights": len(dataset),
        "wandb_run_url": wandb_run_url,
    }
    metadata_path = write_sidecar(args.out, args.year, ".metadata.json", json.dumps(metadata, indent=2))
    log.info("Done. Metadata: %s", metadata_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
