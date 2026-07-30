#!/usr/bin/env python3
"""DL FiLM recalibration with sparse Sencrop calibration.

Known internally as the "KarposSR" deliverable of the Sencrop S23 campaign. Thin
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
from downscaling.prtihvi_wxc.stations import (
    assign_to_grid,
    elevation_offset,
    night_station_targets,
)
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


_REGIME_LABELS = ["R1", "R2", "R3", "R4a", "R4b"]  # ordre canonique (R0 = unknown, ignoré)
# Synoptique large (vent / nuages / pression / température)
_ERA5_FEATURES = ["wind_med", "tcc_med", "mslp_med", "t2m_med"]
# Hygrométrie (déficit Td, point de rosée, RH, extrêmes)
_HYGRO_FEATURES = ["dewpoint_dep_med", "d2m_med", "rh_med", "dewpoint_dep_min", "rh_min"]


def _build_cond_table(
    regimes_csv: str | Path, cond_vars: set[str]
) -> tuple[dict[str, np.ndarray], int]:
    """Construit ``{date_iso: cond_vec}`` à partir de ``regimes_<year>.csv``.

    ``cond_vars`` sous-ensemble de {'regime', 'era5', 'hygro', 'season'}.

    - 'regime' : one-hot 5 dim (R1, R2, R3, R4a, R4b) — R0 = zéros
    - 'era5'   : 4 features synoptiques (wind, tcc, mslp, t2m médianes) z-scorées
    - 'hygro'  : 5 features hygrométrie (dewpoint dep med/min, d2m med, rh med/min) z-scorées
    - 'season' : sin/cos day-of-year

    Retourne la table indexée par 'YYYY-MM-DD' et la dimension du vecteur.
    """
    cond_vars = set(cond_vars)
    df = pd.read_csv(regimes_csv)
    # Date column tolerant
    date_col = next(
        (c for c in ("date", "night_date", "valid_date") if c in df.columns), df.columns[0]
    )
    table: dict[str, np.ndarray] = {}
    # Normalisation continues (z-score sur l'ensemble du fichier — stable cross-année)
    continuous = (_ERA5_FEATURES if "era5" in cond_vars else []) + (
        _HYGRO_FEATURES if "hygro" in cond_vars else []
    )
    means = {f: df[f].mean() for f in continuous if f in df.columns}
    stds = {f: df[f].std() if df[f].std() > 1e-6 else 1.0 for f in continuous if f in df.columns}
    for _, row in df.iterrows():
        parts: list[float] = []
        if "regime" in cond_vars:
            reg = str(row.get("regime", "R0"))
            parts.extend([1.0 if reg == lbl else 0.0 for lbl in _REGIME_LABELS])
        if "era5" in cond_vars:
            for f in _ERA5_FEATURES:
                if f in df.columns:
                    val = (row[f] - means[f]) / stds[f] if pd.notna(row[f]) else 0.0
                    parts.append(float(val))
        if "hygro" in cond_vars:
            for f in _HYGRO_FEATURES:
                if f in df.columns:
                    val = (row[f] - means[f]) / stds[f] if pd.notna(row[f]) else 0.0
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
        coarse_provider,  # callable date -> (x_met, x_dem)
        sencrop_root: str | Path,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        elevation_grid: np.ndarray | None = None,
        min_stations: int = 5,
        lapse_rate: float = -6.5e-3,
        cond_table: dict[str, np.ndarray] | None = None,
        holdout_bbox: tuple[float, float, float, float] | None = None,
        role: str = "all",
        radome_map: dict[str, list[tuple[float, float, float, float]]] | None = None,
        surfex_provider=None,  # #81 : callable date -> (H, W) champ SURFEX °C (enveloppe)
    ) -> None:
        self.coarse_provider = coarse_provider
        self.surfex_provider = surfex_provider
        self.sencrop_root = sencrop_root
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.elevation_grid = elevation_grid
        self.min_stations = min_stations
        self.qc = NetatmoNocturnalQC(lapse_rate=lapse_rate)
        self.cond_table = cond_table
        self.holdout_bbox = holdout_bbox
        self.role = role
        # RADOME (obs quotidiennes MF) : cibles de supervision ADDITIONNELLES,
        # {date 'YYYY-MM-DD': [(lat, lon, alt_m, tmin_C), …]}. cf. radome_loader.
        self.radome_map = radome_map

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
                log.warning(
                    "cond_table missing for %d/%d nights — will use zero vector",
                    len(missing),
                    len(kept),
                )

    def with_role(self, role: str) -> BulkSencropDataset:
        """Clone superficiel avec un rôle leave-station-out (#33)."""
        import copy

        clone = copy.copy(self)
        clone.role = role
        return clone

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict:
        d = self.dates[idx]
        x_met, x_dem = self.coarse_provider(d)
        sample_env = None
        if self.surfex_provider is not None:
            surfex = self.surfex_provider(d)  # (H, W) °C
            sample_env = torch.stack(
                [x_met[0], torch.as_tensor(surfex, dtype=torch.float32)], dim=0
            )  # (2, H, W) : [CERRA 1km, SURFEX 1km]
        obs_qc = self.qc.run(load_sencrop(self.sencrop_root, d))
        tmin, row, col, dz = night_station_targets(
            obs_qc,
            self.lat_grid,
            self.lon_grid,
            self.elevation_grid,
            holdout_bbox=self.holdout_bbox,
            role=self.role,
        )
        # Merge des cibles RADOME de la nuit (postes d'altitude). RADOME sert au
        # TRAINING : role="train" exclut les RADOME dans la holdout-bbox (pas de
        # fuite Baronnies) ; role="val" n'ajoute AUCUN RADOME (l'éval reste Sencrop
        # dans la bbox, comparable au baseline LOO Lot B) ; role="all" prend tout.
        if self.radome_map is not None and self.role != "val":
            rad = self.radome_map.get(d)
            if rad:
                arr = np.asarray(rad, dtype=np.float64)
                rlat, rlon, ralt, rtn = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
                if self.holdout_bbox is not None and self.role == "train":
                    la0, la1, lo0, lo1 = self.holdout_bbox
                    outside = ~(
                        (rlat >= la0) & (rlat <= la1) & (rlon >= lo0) & (rlon <= lo1)
                    )
                    rlat, rlon, ralt, rtn = rlat[outside], rlon[outside], ralt[outside], rtn[outside]
                if rtn.size:
                    rrow, rcol = assign_to_grid(rlat, rlon, self.lat_grid, self.lon_grid)
                    rdz = elevation_offset(ralt, rrow, rcol, self.elevation_grid)
                    tmin = np.concatenate([tmin, rtn.astype(np.float32)])
                    row = np.concatenate([row, rrow])
                    col = np.concatenate([col, rcol])
                    dz = np.concatenate([dz, rdz.astype(np.float32)])
        sample = {
            "x_met": x_met,
            "x_dem": x_dem,
            "obs_tmin": torch.as_tensor(tmin, dtype=torch.float32),
            "obs_row": torch.as_tensor(row, dtype=torch.long),
            "obs_col": torch.as_tensor(col, dtype=torch.long),
            "obs_dz": torch.as_tensor(dz, dtype=torch.float32),
            "date": d,
        }
        if sample_env is not None:
            sample["x_env"] = sample_env
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


def _resolve_to_local(uri: str, *, label: str) -> Path:
    """Résout un input local OU ``s3://`` vers un chemin local (même logique que
    recalibrate_statistical). s3:// → download /tmp via s3fs (endpoint Scaleway)."""
    if uri.startswith("s3://"):
        import tempfile

        import s3fs

        local = Path(tempfile.gettempdir()) / f"{label}_{Path(uri).name}"
        log.info("Téléchargement %s depuis %s → %s", label, uri, local)
        fs = s3fs.S3FileSystem(
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_S3_ENDPOINT"),
        )
        fs.get(uri.replace("s3://", "", 1), str(local))
        return local
    return Path(uri)


def _load_coarse_orography(orog_uri: str) -> np.ndarray:
    """Orographie coarse CERRA → altitude 2D (m). Même détection que recalibrate_statistical
    (orography/orog/z/surface_geopotential ; géopotentiel m²/s² → m via /9.80665)."""
    local = _resolve_to_local(orog_uri, label="cerra_orography")
    ds = xr.open_dataset(local)
    for name in ("orography", "orog", "z", "surface_geopotential"):
        if name in ds:
            da = ds[name]
            if name in ("z", "surface_geopotential"):
                da = da / 9.80665  # géopotentiel (m²/s²) → altitude (m)
            for tdim in ("valid_time", "time"):
                if tdim in da.dims:
                    da = da.isel({tdim: 0}, drop=True)
            return da.values.astype(np.float32)
    raise ValueError(
        f"Aucune variable orographie connue dans {orog_uri} "
        "(cherché : orography, orog, z, surface_geopotential)"
    )


def _build_coarse_provider(cerra_path: Path, dem_path: Path, cerra_orog_path: str | None = None):
    """Loads CERRA atm + DEM once, returns a callable date -> (x_met, x_dem).

    Gère le mismatch CERRA NetCDF : variable temporelle 'valid_time' au lieu de
    'time', t2m en Kelvin. Convertit en Celsius côté provider pour cohérence
    avec le module de calibration (kelvin_to_celsius=False).
    """
    ds_cerra = xr.open_dataset(cerra_path)
    ds_dem = xr.open_dataset(dem_path)

    # Normalise time coord (CERRA utilise valid_time)
    if (
        "valid_time" in ds_cerra.dims
        and "time" not in ds_cerra.dims
        or "valid_time" in ds_cerra.coords
        and "time" not in ds_cerra.coords
    ):
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

    # Stack all DEM vars (elevation, slope, aspect, curvature, svf, ...) en (C_dem, H, W).
    # Chaque canal est z-scoré pour stabiliser l'entraînement (la curvature
    # ~1e-3 sinon écrase tout si on garde l'élévation brute en mètres).
    dem_vars = list(ds_dem.data_vars)
    dem_channels = []
    for v in dem_vars:
        arr = ds_dem[v].values.astype(np.float32)
        mean, std = float(np.nanmean(arr)), float(np.nanstd(arr))
        if std < 1e-9:
            std = 1.0
        dem_channels.append(((arr - mean) / std).astype(np.float32))
    dem_arr = np.stack(dem_channels, axis=0)  # (C_dem, H, W)
    H_fine, W_fine = dem_arr.shape[1:]
    x_dem = torch.from_numpy(dem_arr)  # (C_dem, H, W)
    log.info(
        "Provider grids: DEM (%d ch: %s) (%d, %d) | CERRA t2m yearly %s",
        len(dem_vars),
        dem_vars,
        H_fine,
        W_fine,
        tuple(ds_cerra[t_var].shape),
    )

    def provider(d: str) -> tuple[torch.Tensor, torch.Tensor]:
        slab = ds_cerra[t_var].sel(time=d, method="nearest")
        arr = slab.values.astype(np.float32)
        # (H_coarse, W_coarse) → bilinear → (H_fine, W_fine)
        coarse = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, Hc, Wc)
        fine = F.interpolate(coarse, size=(H_fine, W_fine), mode="bilinear", align_corners=False)
        x_met = fine.squeeze(0)  # (1, H_fine, W_fine)
        return x_met, x_dem

    # First-guess lapse (v2b) : dz = z_DEM_1km − z_orog_coarse_1km (m), statique.
    # Le first-guess lapse = T_CERRA_1km + lapse_rate·dz reproduit le plancher Lot B
    # (~1,6°C) au lieu du CERRA bilinéaire brut (~2,8°C).
    fg_dz = None
    if cerra_orog_path is not None:
        z_coarse = _load_coarse_orography(cerra_orog_path)  # (Hc, Wc), m
        z_coarse_1km = (
            F.interpolate(
                torch.from_numpy(z_coarse).unsqueeze(0).unsqueeze(0),
                size=(H_fine, W_fine),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )  # (H_fine, W_fine)
        elev_var = "elevation" if "elevation" in ds_dem.data_vars else list(ds_dem.data_vars)[0]
        z_fine = ds_dem[elev_var].values.astype(np.float32)  # (H_fine, W_fine), m brut
        fg_dz = np.nan_to_num(z_fine - z_coarse_1km, nan=0.0).astype(np.float32)
        log.info(
            "First-guess lapse : dz = z_DEM − z_orog_coarse : mean=%.1f m std=%.1f m (min %.0f, max %.0f)",
            float(np.nanmean(fg_dz)),
            float(np.nanstd(fg_dz)),
            float(np.nanmin(fg_dz)),
            float(np.nanmax(fg_dz)),
        )

    return provider, ds_cerra, ds_dem, fg_dz


def _build_surfex_provider(
    surfex_path: str, lat_grid: np.ndarray, lon_grid: np.ndarray, var: str = "T2M"
):
    """Charge l'artefact SURFEX (LAT/LON + var K) → callable date -> (H, W) °C (#81).

    Le champ SURFEX 1 km (grille Lambert propre) est régridé au plus proche voisin
    sur la grille DEM fine (lat/lon), et réduit au **Tmin journalier** (min de la
    variable sur les heures de la date) → borne froide assimilée de l'enveloppe.

    ``var`` : 'T2M' (air) ou 'TSRAD' (T_skin, ~1,6°C plus froid = gel radiatif,
    plancher physiquement correct — #82).
    """
    from scipy.spatial import cKDTree

    local = _resolve_to_local(surfex_path, label="surfex_envelope")
    ds = xr.open_dataset(local)
    if var not in ds.variables:
        raise ValueError(f"--surfex-var {var!r} absent de l'artefact ({list(ds.data_vars)})")
    T = np.asarray(ds[var].values, dtype=np.float32)  # (time, yy, xx) Kelvin
    LAT = np.asarray(ds["LAT"].values, dtype=np.float64)  # (yy, xx)
    LON = np.asarray(ds["LON"].values, dtype=np.float64)
    times = pd.to_datetime(ds["time"].values)
    day_of = np.array([t.date() for t in times])

    la = np.asarray(lat_grid, dtype=np.float64)
    lo = np.asarray(lon_grid, dtype=np.float64)
    LO2, LA2 = np.meshgrid(lo, la) if la.ndim == 1 else (lo, la)
    H_fine, W_fine = LA2.shape
    tree = cKDTree(np.column_stack([LON.ravel(), LAT.ravel()]))
    _, idx = tree.query(np.column_stack([LO2.ravel(), LA2.ravel()]))
    log.info(
        "SURFEX envelope: %s var=%s (%d pas), régrid NN %s → grille fine (%d, %d)",
        Path(local).name, var, T.shape[0], tuple(LAT.shape), H_fine, W_fine,
    )

    def surfex_provider(d: str) -> np.ndarray:
        dd = pd.Timestamp(d).date()
        sel = day_of == dd
        if not sel.any():
            sel = np.zeros(len(times), bool)
            sel[int(np.argmin(np.abs(times - pd.Timestamp(d))))] = True
        tmin_c = np.nanmin(T[sel], axis=0) - 273.15  # (yy, xx) °C
        return tmin_c.ravel()[idx].reshape(H_fine, W_fine).astype(np.float32)

    return surfex_provider


def _dl_output_dataset(out_da: xr.DataArray) -> xr.Dataset:
    """Champ DL (backbone 1 km) → Dataset ``{t2m, t2m_prerbf}``.

    Le DL n'applique aucune correction RBF Sencrop lui-même : ``t2m`` (produit)
    et ``t2m_prerbf`` (backbone pour le RBF-LOO de ``analyze --loo``) sont donc
    identiques. Émettre ``t2m_prerbf`` permet de brancher le RBF Sencrop + LOO
    par-dessus le DL → comparaison fair DL+RBF vs Lot B (#33).
    """
    return xr.Dataset({"t2m": out_da, "t2m_prerbf": out_da})


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
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output target: local dir OR s3:// / scw:// URL "
        "(zarr + sidecar metadata.json written there). "
        "Lightning checkpoints stay local (--checkpoint-dir).",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Local dir for Lightning ModelCheckpoint. "
        "Default: <out> if local, else ./checkpoints/<year>.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="Accelerator. 'auto' = MPS si Apple Silicon, sinon CUDA, sinon CPU.",
    )
    p.add_argument(
        "--base-ch",
        type=int,
        default=32,
        help="U-Net base channels (capacity). Issue #28: 64 ≈ 4.6M params.",
    )
    p.add_argument(
        "--n-levels",
        type=int,
        default=3,
        help="U-Net depth. Issue #28: 4 augmente receptive field.",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="If >0, EarlyStopping on val/rmse with this patience (issue #28).",
    )
    p.add_argument(
        "--loss-quantile",
        type=float,
        default=None,
        help="If set in (0,1), use pinball/quantile loss instead of RMSE on station "
        "observations. q<0.5 penalizes under-prediction of cold (missed frost) "
        "more than over-prediction. Recommended for frost detection: q=0.1 "
        "(misses cost ×9 false alarms). Issue #5.",
    )
    p.add_argument(
        "--target-mode",
        choices=("raw", "residual"),
        default="raw",
        help="'raw' (défaut) = le U-Net prédit Tmin directement (baseline, replique "
        "CERRA + biais chaud). 'residual' (v2) = first-guess physique (CERRA 1 km) + "
        "résidu appris → plancher = first-guess, capacité focalisée sur la structure "
        "fine (cuvette). Fixe le déficit de fit in-sample.",
    )
    p.add_argument(
        "--first-guess",
        choices=("bilinear", "lapse"),
        default="bilinear",
        help="First-guess du mode résidu (v2b). 'bilinear' (défaut) = CERRA bilinéaire à "
        "1 km (plancher ~2,8°C). 'lapse' = + correction lapse-rate maille↔coarse "
        "(plancher ≈ Lot B ~1,6°C) — requiert --cerra-orog. N'a d'effet qu'avec "
        "--target-mode residual.",
    )
    p.add_argument(
        "--cerra-orog",
        type=str,
        default=None,
        help="Orographie coarse CERRA (cerra_orography.nc), locale OU s3://. Requise "
        "pour --first-guess lapse. Ex. s3://karpos-backtest-data/recalibrated/cerra_orography.nc.",
    )
    p.add_argument(
        "--cond-vars",
        type=str,
        default="",
        help="Comma-separated FiLM conditioning vars: regime, era5, hygro, season. "
        "Empty = DEM-only (baseline). Requires --regimes-csv. Issue #5 item 4.",
    )
    p.add_argument(
        "--regimes-csv",
        type=str,
        default=None,
        help="Path to regimes_all.csv (output of flag_regimes.py). "
        "Required when --cond-vars is non-empty.",
    )
    p.add_argument(
        "--holdout-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Leave-station-out : tient les stations de cette bbox HORS du fit "
        "(train) et les utilise comme val out-of-station. Rend val/rmse comparable "
        "au LOO Lot B (#33). Ex. Baronnies cold-pool : 44.15 44.45 5.15 5.40.",
    )
    p.add_argument(
        "--es-min-delta",
        type=float,
        default=1e-3,
        help="EarlyStopping min_delta sur val/rmse (avec --early-stopping-patience>0). "
        "Recommandé 0.05 pour durcir (évite de traîner sur du bruit).",
    )
    p.add_argument(
        "--radome-obs",
        type=str,
        default=None,
        help="Racine S3/locale des obs RADOME quotidiennes (station=<id>/*.csv), ex. "
        "s3://karpos-backtest-data/observations/radome-oauth/quotidienne/2023. Ajoute les "
        "postes d'altitude comme cibles de supervision (le régime R4a cold-pool). "
        "Requiert --radome-catalogue.",
    )
    p.add_argument(
        "--radome-catalogue",
        type=str,
        default=None,
        help="CSV catalogue RADOME (station_id, lat, lon, alt_m). "
        "Ex. s3://karpos-backtest-data/observations/radome-oauth/catalogue_2023.csv.",
    )
    p.add_argument(
        "--surfex",
        type=str,
        default=None,
        help="Artefact SURFEX (T2M/LAT/LON), local OU s3://. Fournit la borne froide "
        "assimilée de l'enveloppe du clamp (#81). Ex. "
        "s3://karpos-backtest-data/surfex/drome_1km/2023-04/surfex_drome_2023-04.nc.",
    )
    p.add_argument(
        "--surfex-var",
        type=str,
        default="TSRAD",
        help="Variable SURFEX pour la borne froide de l'enveloppe : 'TSRAD' (T_skin, "
        "gel radiatif, défaut, #82) ou 'T2M' (air).",
    )
    p.add_argument(
        "--clamp",
        action="store_true",
        help="#81 : borne la sortie DL dans l'enveloppe [min,max](CERRA, SURFEX) via "
        "tête tanh différentiable → composant assurable. Requiert --surfex.",
    )
    p.add_argument(
        "--clamp-margin",
        type=float,
        default=0.0,
        help="Marge °C ajoutée à la demi-largeur d'enveloppe (0 = enveloppe stricte).",
    )
    p.add_argument("--wandb-project", default="karpos-recalibrate-dl-film")
    p.add_argument("--wandb-disabled", action="store_true")
    p.add_argument(
        "--smoke-test", action="store_true", help="run 1 epoch on a tiny subset (CPU OK)"
    )
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

    if args.first_guess == "lapse" and not args.cerra_orog:
        raise ValueError("--first-guess lapse requiert --cerra-orog (orographie coarse CERRA)")

    # ---- Coarse provider + dates --------------------------------------------
    # dz (first-guess lapse) chargé seulement en mode lapse.
    orog_for_fg = args.cerra_orog if args.first_guess == "lapse" else None
    provider, ds_cerra, ds_dem, fg_dz = _build_coarse_provider(
        args.cerra_atm, args.dem, cerra_orog_path=orog_for_fg
    )
    dates = _dates_in_year(ds_cerra, args.year)
    log.info("Year=%d, %d candidate dates", args.year, len(dates))

    lat_grid = ds_dem["lat"].values if "lat" in ds_dem else ds_dem["latitude"].values
    lon_grid = ds_dem["lon"].values if "lon" in ds_dem else ds_dem["longitude"].values
    # elevation_grid sert juste à calculer dz station↔maille (lookup, pas le model input).
    elev_var = "elevation" if "elevation" in ds_dem.data_vars else next(iter(ds_dem.data_vars))
    dem_2d = ds_dem[elev_var].values.astype(np.float32)
    dem_in_ch = len(list(ds_dem.data_vars))

    # ---- SURFEX envelope provider (#81 clamp) -------------------------------
    if args.clamp and not args.surfex:
        raise ValueError("--clamp requiert --surfex (borne froide assimilée de l'enveloppe)")
    surfex_provider = None
    if args.surfex:
        surfex_provider = _build_surfex_provider(
            args.surfex, lat_grid, lon_grid, var=args.surfex_var
        )

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

    # ---- RADOME (cibles de supervision d'altitude additionnelles) -----------
    radome_map = None
    if args.radome_obs:
        if not args.radome_catalogue:
            raise ValueError("--radome-obs requiert --radome-catalogue")
        from downscaling.deep_learning.radome_loader import load_radome_targets

        radome_map = load_radome_targets(args.radome_obs, args.radome_catalogue)
        log.info(
            "RADOME branché en supervision (train). Holdout appliqué aussi aux RADOME "
            "(pas de fuite Baronnies) ; éval reste Sencrop dans la bbox."
        )

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
        holdout_bbox=tuple(args.holdout_bbox) if args.holdout_bbox else None,
        radome_map=radome_map,
        surfex_provider=surfex_provider,
    )
    if args.holdout_bbox:
        log.info(
            "Leave-station-out actif : val = stations dans bbox lat[%.2f,%.2f] lon[%.2f,%.2f] "
            "(hors du fit) — val/rmse comparable au LOO Lot B #33",
            *args.holdout_bbox,
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
        "unet",
        met_in_ch=1,
        dem_in_ch=dem_in_ch,
        base_ch=args.base_ch,
        n_levels=args.n_levels,
        use_film=True,
        cond_dim=cond_dim,
    )
    log.info(
        "U-Net: base_ch=%d, n_levels=%d, dem_in_ch=%d, cond_dim=%d",
        args.base_ch,
        args.n_levels,
        dem_in_ch,
        cond_dim,
    )
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
        target_mode=args.target_mode,
        first_guess=args.first_guess,
        first_guess_dz=torch.from_numpy(fg_dz) if fg_dz is not None else None,
        clamp=args.clamp,
        clamp_margin=args.clamp_margin,
    )
    if args.clamp:
        log.info(
            "Clamp #81 ACTIF : sortie bornée tanh dans [min,max](CERRA, SURFEX) + marge %.1f°C "
            "→ composant assurable, risque de modèle plafonné par |CERRA−SURFEX|",
            args.clamp_margin,
        )
    if args.loss_quantile is not None:
        log.info("Loss : pinball quantile q=%.2f (tail-aware, issue #5)", args.loss_quantile)
    else:
        log.info("Loss : RMSE (symmetric, baseline)")
    if args.target_mode == "residual":
        log.info(
            "Target : residual (first-guess=%s + résidu appris) — plancher %s",
            args.first_guess,
            "≈ Lot B lapse (~1,6°C)" if args.first_guess == "lapse" else "CERRA bilinéaire (~2,8°C)",
        )
    else:
        log.info("Target : raw Tmin (baseline)")

    # ---- Trainer ------------------------------------------------------------
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{args.year}-best",
            monitor="val/rmse",
            mode="min",
            save_top_k=1,
        )
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/rmse",
                mode="min",
                patience=args.early_stopping_patience,
                min_delta=args.es_min_delta,
            )
        )
        log.info(
            "EarlyStopping enabled: monitor=val/rmse patience=%d min_delta=%.3f",
            args.early_stopping_patience,
            args.es_min_delta,
        )

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
            if surfex_provider is not None:
                surfex = torch.as_tensor(surfex_provider(d), dtype=torch.float32)
                x_env = torch.stack([x_met[0], surfex], dim=0)  # (2, H, W)
                batch["x_env"] = x_env.unsqueeze(0).to(inference_device)
            if cond_table is not None:
                cv = cond_table.get(d)
                if cv is None:
                    cv = np.zeros(cond_dim, dtype=np.float32)
                batch["cond_vec"] = (
                    torch.as_tensor(cv, dtype=torch.float32).unsqueeze(0).to(inference_device)
                )
            pred = lit._predict_target(batch).squeeze().cpu().numpy()
            slab = xr.DataArray(
                pred,
                dims=("latitude", "longitude"),
                coords={"latitude": lat_grid, "longitude": lon_grid},
                name="t2m",
            ).expand_dims(time=[pd.Timestamp(d)])
            out_grids.append(slab)

    out_da = xr.concat(out_grids, dim="time")
    # Le champ DL est le BACKBONE 1 km (avant correction RBF Sencrop live). On l'émet
    # aussi sous `t2m_prerbf` pour que `analyze_recalibrated_statistical.py --loo`
    # applique le RBF Sencrop + leave-one-out par-dessus → comparaison fair DL+RBF
    # vs Lot B (lapse+QDM)+RBF (#33). Le DL n'applique aucun RBF lui-même, donc
    # t2m == t2m_prerbf ici.
    out_ds = _dl_output_dataset(out_da)
    zarr_store = make_zarr_store(args.out, args.year)
    out_ds.to_zarr(zarr_store, mode="w")
    log.info(
        "Wrote %s (%d nights, vars: t2m + t2m_prerbf pour RBF-LOO)",
        describe(args.out, args.year, ".zarr"),
        len(out_grids),
    )

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
        "target_mode": args.target_mode,
        "first_guess": args.first_guess,
        "clamp": bool(args.clamp),
        "clamp_margin": args.clamp_margin,
        "surfex": str(args.surfex) if args.surfex else None,
        "surfex_var": args.surfex_var if args.surfex else None,
        "cerra_orog": str(args.cerra_orog) if args.cerra_orog else None,
        "early_stopping_patience": args.early_stopping_patience,
        "es_min_delta": args.es_min_delta,
        "holdout_bbox": list(args.holdout_bbox) if args.holdout_bbox else None,
        "cond_vars": sorted(cond_vars) if cond_vars else [],
        "radome_obs": args.radome_obs,
        "radome_catalogue": args.radome_catalogue,
        "cond_dim": cond_dim,
        "n_nights": len(dataset),
        "wandb_run_url": wandb_run_url,
    }
    metadata_path = write_sidecar(
        args.out, args.year, ".metadata.json", json.dumps(metadata, indent=2)
    )
    log.info("Done. Metadata: %s", metadata_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
