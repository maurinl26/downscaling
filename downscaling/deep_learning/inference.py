"""
Inférence du modèle de descente d'échelle sur nouvelles données ERA5 / CERRA.

Gestion des grands domaines
----------------------------
Pour les domaines dépassant la taille des tuiles d'entraînement, on utilise
une inférence par tuiles avec chevauchement et recombinaison par moyenne
pondérée (fenêtre de Hann) pour éviter les artefacts aux jointures.

Usage CLI
---------
    python -m downscaling.deep_learning.inference \
        --config     config/drome_ardeche.yml \
        --checkpoint checkpoints/best_model.pt \
        --era5-sl    data/era5/era5_sl_20210427.nc \
        --dem-attrs  data/dem/dem_attributes.nc \
        --stats      checkpoints/normalization_stats.json \
        --out        output/dl_downscaled_20210427.nc
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("PyTorch requis : pip install torch") from e

from .dataset import prepare_inference_batch, DEFAULT_MET_VARS, DEFAULT_DEM_VARS
from .model import build_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inférence par tuiles avec recombinaison Hann
# ---------------------------------------------------------------------------

def hann_window_2d(size: int) -> np.ndarray:
    """Fenêtre de Hann 2D pour la recombinaison sans artefacts."""
    w1d = np.hanning(size).astype(np.float32)
    return np.outer(w1d, w1d)


def tiled_inference(
    model: torch.nn.Module,
    x_met: torch.Tensor,
    x_dem: torch.Tensor,
    tile_size: int = 64,
    overlap: int = 16,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Découpe x_met et x_dem en tuiles, applique le modèle, recombine.

    Parameters
    ----------
    x_met:
        Champs météo (1, C_met, H, W) — déjà normalisés.
    x_dem:
        Attributs MNT (1, C_dem, H, W) — déjà normalisés.
    tile_size:
        Taille des tuiles (pixels). Doit correspondre à la taille d'entraînement.
    overlap:
        Chevauchement entre tuiles (pixels).

    Returns
    -------
    torch.Tensor (1, C_out, H, W) — reconstruction complète.
    """
    _, C_met, H, W = x_met.shape
    _, C_out, _, _ = _infer_output_shape(model, C_met, tile_size, device)

    output = np.zeros((1, C_out, H, W), dtype=np.float32)
    weight = np.zeros((1, 1, H, W), dtype=np.float32)
    win = hann_window_2d(tile_size)[np.newaxis, np.newaxis, :, :]  # (1, 1, T, T)

    stride = tile_size - overlap
    model.eval()
    with torch.no_grad():
        for i0 in range(0, H - tile_size + 1, stride):
            for j0 in range(0, W - tile_size + 1, stride):
                i1, j1 = i0 + tile_size, j0 + tile_size
                tile_met = x_met[:, :, i0:i1, j0:j1].to(device)
                tile_dem = x_dem[:, :, i0:i1, j0:j1].to(device)
                pred = model(tile_met, tile_dem).cpu().numpy()
                output[:, :, i0:i1, j0:j1] += pred * win
                weight[:, :, i0:i1, j0:j1] += win

    # Normalise par le poids
    output /= np.where(weight > 0, weight, 1.0)
    return torch.from_numpy(output)


def _infer_output_shape(
    model: torch.nn.Module, c_in: int, tile_size: int, device: torch.device
) -> tuple:
    """Infère le nombre de canaux de sortie en faisant passer un batch factice."""
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, c_in, tile_size, tile_size, device=device)
        dem_ch = next(
            (m for m in model.modules() if hasattr(m, "encoders")), None
        )
        if dem_ch is not None:
            d_in = dem_ch.encoders[0][0].block[0].in_channels
        else:
            d_in = 4
        d = torch.zeros(1, d_in, tile_size, tile_size, device=device)
        out = model(x, d)
    return out.shape


# ---------------------------------------------------------------------------
# Pipeline d'inférence complet
# ---------------------------------------------------------------------------

class DLInferencePipeline:
    """
    Charge un checkpoint et applique le modèle sur un fichier ERA5/CERRA.

    Parameters
    ----------
    checkpoint_path:
        Fichier .pt sauvegardé par Trainer (contient model_state_dict).
    config:
        Dictionnaire de configuration (section 'deep_learning').
    stats_path:
        Fichier JSON des statistiques de normalisation.
    device:
        'cuda', 'mps' ou 'cpu'. Auto-détecté si None.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: dict,
        stats_path: str | Path,
        device: str | None = None,
    ):
        dl_cfg = config.get("deep_learning", config)
        self.met_vars = dl_cfg.get("met_vars", DEFAULT_MET_VARS)
        self.tile_size = dl_cfg.get("patch_size", 64)
        self.overlap = dl_cfg.get("overlap", 16)

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Statistiques de normalisation
        with open(stats_path) as f:
            raw = json.load(f)
        self.stats = {k: tuple(v) for k, v in raw.items()}

        # Modèle
        self.model = build_model(
            architecture=dl_cfg.get("architecture", "unet"),
            met_in_ch=len(self.met_vars),
            dem_in_ch=dl_cfg.get("dem_in_ch", 4),
            base_ch=dl_cfg.get("base_ch", 64),
            n_levels=dl_cfg.get("n_levels", 4),
            use_film=dl_cfg.get("use_film", True),
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        log.info(f"Checkpoint chargé : {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    def run(
        self,
        coarse_ds: xr.Dataset,
        dem_ds: xr.Dataset,
        output_vars: list[str] | None = None,
    ) -> xr.Dataset:
        """
        Applique le modèle sur toute la série temporelle.

        Parameters
        ----------
        coarse_ds:
            Dataset ERA5/CERRA basse résolution.
        dem_ds:
            Dataset attributs MNT (élévation, pente, aspect, courbure).
        output_vars:
            Variables à inclure dans le Dataset de sortie.
            Défaut = self.met_vars.

        Returns
        -------
        xr.Dataset haute résolution.
        """
        output_vars = output_vars or self.met_vars
        n_times = len(coarse_ds.time) if "time" in coarse_ds.dims else 1

        # Prépare le tenseur DEM une seule fois (constant dans le temps)
        _, x_dem = prepare_inference_batch(
            coarse_ds, dem_ds, self.met_vars, self.stats,
            time_idx=0, device=str(self.device)
        )

        H = x_dem.shape[-2]
        W = x_dem.shape[-1]
        C_out = len(self.met_vars)

        results = np.zeros((n_times, C_out, H, W), dtype=np.float32)

        log.info(f"Inférence sur {n_times} pas de temps…")
        for t in range(n_times):
            x_met, _ = prepare_inference_batch(
                coarse_ds, dem_ds, self.met_vars, self.stats,
                time_idx=t, device=str(self.device)
            )
            if H <= self.tile_size and W <= self.tile_size:
                with torch.no_grad():
                    pred = self.model(x_met, x_dem).cpu().numpy()
            else:
                pred = tiled_inference(
                    self.model, x_met, x_dem,
                    tile_size=self.tile_size,
                    overlap=self.overlap,
                    device=self.device,
                ).numpy()
            results[t] = pred[0]

        # Dénormalisation
        for ci, v in enumerate(self.met_vars):
            if v in self.stats:
                mu, sigma = self.stats[v]
                results[:, ci] = results[:, ci] * sigma + mu

        # Construction du Dataset xarray de sortie
        time_coord = coarse_ds.time if "time" in coarse_ds.dims else None
        lat = dem_ds.coords.get("lat", dem_ds.coords.get("latitude"))
        lon = dem_ds.coords.get("lon", dem_ds.coords.get("longitude"))

        data_vars = {}
        for ci, v in enumerate(self.met_vars):
            if v not in output_vars:
                continue
            if time_coord is not None:
                da = xr.DataArray(
                    results[:, ci],
                    dims=["time", "y", "x"],
                    coords={"time": time_coord},
                )
            else:
                da = xr.DataArray(results[0, ci], dims=["y", "x"])

            if lat is not None:
                da = da.assign_coords(lat=(["y", "x"], lat.values if lat.ndim == 2 else None))
            data_vars[v] = da

        ds_out = xr.Dataset(data_vars)
        ds_out.attrs["downscaling_method"] = "deep_learning (DEM-conditioned U-Net)"
        ds_out.attrs["model_checkpoint"] = str(self.checkpoint_path if hasattr(self, "checkpoint_path") else "")
        return ds_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inférence du modèle DL de descente d'échelle")
    p.add_argument("--config", required=True, help="Config YAML")
    p.add_argument("--checkpoint", required=True, help="Fichier checkpoint .pt")
    p.add_argument("--era5-sl", required=True, help="ERA5 single-level NetCDF")
    p.add_argument("--dem-attrs", required=True, help="Attributs MNT NetCDF")
    p.add_argument("--stats", required=True, help="JSON statistiques normalisation")
    p.add_argument("--out", required=True, help="Fichier NetCDF de sortie")
    p.add_argument("--device", default=None)
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=16)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.tile_size:
        cfg.setdefault("deep_learning", {})["patch_size"] = args.tile_size
    cfg.setdefault("deep_learning", {})["overlap"] = args.overlap

    pipeline = DLInferencePipeline(
        checkpoint_path=args.checkpoint,
        config=cfg,
        stats_path=args.stats,
        device=args.device,
    )

    coarse_ds = xr.open_dataset(args.era5_sl, engine="netcdf4")
    dem_ds = xr.open_dataset(args.dem_attrs, engine="netcdf4")

    ds_out = pipeline.run(coarse_ds, dem_ds)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(args.out)
    log.info(f"Sortie écrite dans {args.out}")


if __name__ == "__main__":
    main()
