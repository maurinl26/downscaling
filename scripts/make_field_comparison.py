#!/usr/bin/env python3
"""Figure de présentation — gain de résolution KarposSR vs modèle de fond.

Visuel spatial pour la réunion Sencrop : le champ T2m KarposSR fin (1 km) sur une
nuit de gel de référence, comparé au MÊME champ agrégé à ~5,5 km (la résolution
d'un modèle de fond type AROME/CERRA). Illustre le détail de structure (poches
froides) qu'un socle grossier ne peut pas résoudre.

HONNÊTETÉ : le panneau grossier est KarposSR AGRÉGÉ, pas un run AROME réel. Il
illustre l'effet de RÉSOLUTION, pas l'écart de modèle. La comparaison à un vrai
champ AROME 2025 demande le champ grossier stocké sur S3 (indisponible hors creds).

Sortie : docs/methodology/figures/F6_karpossr_resolution.{png,pdf}
Provenance loggée dans le pied de figure (source, git SHA, nuit, facteur d'agrégation).

    uv run python scripts/make_field_comparison.py \
        --zarr /Users/loicmaurin/kDrive/karpos_datasets/output/dl_film_multiyear/2025.zarr \
        --night 2025-02-15 --coarsen 6
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", type=str, required=True, help="Sortie KarposSR (var t2m)")
    p.add_argument("--night", type=str, default="2025-02-15", help="YYYY-MM-DD")
    p.add_argument("--coarse-nc", type=str, default=None,
                   help="NetCDF du socle grossier RÉEL (ex CERRA t2m). Si absent : agrégation.")
    p.add_argument("--coarsen", type=int, default=6,
                   help="Facteur d'agrégation 1 km -> socle si --coarse-nc absent (6 ~= 5,5 km)")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent
                   / "docs" / "methodology" / "figures")
    args = p.parse_args()

    ds = xr.open_zarr(args.zarr)
    tvar = "t2m" if "t2m" in ds.data_vars else list(ds.data_vars)[0]
    fine = ds[tvar].sel(time=args.night).load()
    ydim, xdim = fine.dims[-2], fine.dims[-1]
    latc = next((c for c in ("latitude", "lat") if c in ds.coords), ydim)
    lonc = next((c for c in ("longitude", "lon") if c in ds.coords), xdim)
    lat = ds[latc].values
    lon = ds[lonc].values
    extent = [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]

    if args.coarse_nc:
        # Socle RÉEL (CERRA) : min nocturne du jour cible, grille native ~5,5 km.
        cds = xr.open_dataset(args.coarse_nc)
        cvar = "t2m" if "t2m" in cds.data_vars else list(cds.data_vars)[0]
        tname = "valid_time" if "valid_time" in cds[cvar].dims else "time"
        day = cds[cvar].sel({tname: args.night}).min(dim=tname).load()
        if float(day.median()) > 100:  # Kelvin -> °C
            day = day - 273.15
        clatc = "latitude" if "latitude" in cds.coords else "lat"
        clonc = "longitude" if "longitude" in cds.coords else "lon"
        coarse = day
        coarse_extent = [float(cds[clonc].min()), float(cds[clonc].max()),
                         float(cds[clatc].min()), float(cds[clatc].max())]
        coarse_label = "Socle CERRA 5,5 km\n(réanalyse, analogue AROME opérationnel)"
    else:
        coarse_block = fine.coarsen({ydim: args.coarsen, xdim: args.coarsen},
                                    boundary="trim").mean()
        coarse = coarse_block.interp_like(fine, method="nearest")
        coarse_extent = extent
        coarse_label = f"Modele de fond ~5,5 km\n(KarposSR agrege x{args.coarsen}, illustration)"

    both = np.concatenate([coarse.values.ravel(), fine.values.ravel()])
    both = both[np.isfinite(both)]
    vmin, vmax = np.percentile(both, 2), np.percentile(both, 98)

    fig = plt.figure(figsize=(11.5, 5.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.16)

    def _map(ax, field, title, interp, ext):
        im = ax.imshow(field.values, origin="lower", extent=ext, aspect="auto",
                       cmap="RdYlBu_r", vmin=vmin, vmax=vmax, interpolation=interp)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("T2m (°C)", fontsize=9)
        return im

    # Socle : rendu « nearest » (blocs nets = maille grossière).
    # KarposSR : « bilinear » pour lisser l'aliasing d'affichage du champ fin.
    _map(fig.add_subplot(gs[0, 0]), coarse, coarse_label, "nearest", coarse_extent)
    _map(fig.add_subplot(gs[0, 1]), fine,
         "KarposSR 1 km\n(downscaling + calibration Sencrop)", "bilinear", extent)

    tmin = float(np.nanmin(fine.values))
    fig.suptitle(
        f"Gain de résolution KarposSR, nuit de gel du {args.night} "
        f"(Tmin champ {tmin:.1f} °C, Drôme/Baronnies)",
        fontsize=13.5, fontweight="bold", y=1.03,
    )
    src = f"socle {Path(args.coarse_nc).name}" if args.coarse_nc else \
          f"socle = KarposSR agrégé ×{args.coarsen} (illustration, pas AROME réel)"
    fig.text(
        0.5, -0.06,
        f"{src} · fin {Path(args.zarr).name} · git {_git_sha()} · "
        f"nuit = min nocturne · échelle T commune p2-p98.",
        ha="center", fontsize=8.5, color="#555",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"F6_karpossr_resolution.{ext}",
                    dpi=200, bbox_inches="tight")
    print("écrit:", args.out_dir / "F6_karpossr_resolution.png")
    print(f"contraste spatial : socle std={float(np.nanstd(coarse.values)):.2f} °C "
          f"vs KarposSR std={float(np.nanstd(fine.values)):.2f} °C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
