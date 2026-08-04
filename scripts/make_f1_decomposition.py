"""Fig. 1 GMD — décomposition des contributions (méthode de production).

2x2 : [① terrain/MNT | ② stations Sencrop seules] / [③ CERRA background | ④ CERRA-OI
production LISSÉE]. Montre la méthode EXTENSIVE de production :

    CERRA 5,5 km (background) -> downscaling U-Net léger (structure) ->
    assimilation OI terrain-aware Sencrop (décorrélation H x V) ->
    lissage post-traitement (#103, sigma ~ 1,5 km, cohérence sans perte de skill)

Détection = assimilation ; U-Net = structure fine (cf. §5.2). Écrit PNG (300 dpi) +
PDF dans docs/methodology/figures/ sous F1_decomposition ET F6_karpossr_resolution.

Données externes (kDrive) documentées ci-dessous ; nuit de référence gel 14->15 fév 2025.
"""

import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator, griddata

from downscaling.prtihvi_wxc.optimal_interpolation import smooth_analysis, terrain_aware_oi

DATA = Path("/Users/loicmaurin/kDrive/karpos_datasets/output/regen_cerra_2023")
SENC = "/Users/loicmaurin/kDrive/karpos_datasets/data/raw/sencrop"
KZ = "/Users/loicmaurin/kDrive/karpos_datasets/output/karpos_sr_multiyear/2025.zarr"
OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "methodology" / "figures"
A, B = pd.Timestamp("2025-02-14 20:00"), pd.Timestamp("2025-02-15 08:00")
SMOOTH_KM = 1.5

plt.rcParams.update({"savefig.dpi": 300, "font.size": 11})


def _load():
    kz = xr.open_zarr(KZ)
    klat, klon = kz["latitude"].values, kz["longitude"].values
    KLON, KLAT = np.meshgrid(klon, klat)
    elev = np.asarray(xr.open_dataset(DATA / "dem_attributes_svf.nc")["elevation"].values)
    if elev.shape != (klat.size, klon.size):
        elev = elev.T
    c = xr.open_dataset(DATA / "cerra_atm_2025.nc")
    ct = pd.DatetimeIndex(pd.to_datetime(c["valid_time"].values, utc=True)).tz_localize(None)
    clat, clon = c["latitude"].values, c["longitude"].values
    casc = clat[0] < clat[-1]
    CLON, CLAT = np.meshgrid(clon, clat)
    m = (ct >= A) & (ct <= B)
    cnm = np.nanmin(c["t2m"].values[m], axis=0) - 273.15
    bg = griddata(
        np.column_stack([CLON.ravel(), CLAT.ravel()]),
        (cnm if casc else cnm[::-1]).ravel(),
        (KLON, KLAT),
        method="linear",
    )
    bad = ~np.isfinite(bg)
    if bad.any():
        bg[bad] = griddata(
            np.column_stack([KLON[~bad], KLAT[~bad]]),
            bg[~bad],
            (KLON[bad], KLAT[bad]),
            method="nearest",
        )
    sdf = pd.read_csv(sorted(glob.glob(f"{SENC}/2025.csv/part-*.csv"))[0])
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], utc=True, errors="coerce").dt.tz_localize(
        None
    )
    sdf = sdf[sdf["temperature_source"] == "station"]
    cat = pd.read_csv(f"{SENC}/stations_integrated.csv")
    alt = "altitude_m" if "altitude_m" in cat.columns else "altitude"
    w = sdf[(sdf.timestamp >= A) & (sdf.timestamp <= B)]
    s = (
        w.groupby("station_id")["temperature"]
        .min()
        .reset_index()
        .merge(cat, left_on="station_id", right_on="bucket_id")
        .dropna(subset=["latitude", "longitude", "temperature"])
    )
    s = s[(s.latitude >= 44) & (s.latitude <= 45.5) & (s.longitude >= 4) & (s.longitude <= 5.5)]
    sla, slo, sv = s.latitude.values, s.longitude.values, s.temperature.values
    ei = RegularGridInterpolator((klat, klon), elev, bounds_error=False, fill_value=None)
    sa = np.where(np.isfinite(s[alt].values), s[alt].values, ei(np.column_stack([sla, slo])))
    oi = smooth_analysis(
        terrain_aware_oi(bg, klat, klon, elev, sla, slo, sa, sv), klat, klon, sigma_km=SMOOTH_KM
    )
    return klat, klon, elev, bg, oi, sla, slo, sv


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    klat, klon, elev, bg, oi, sla, slo, sv = _load()
    zl = (sla.min() - 0.10, sla.max() + 0.10)
    zo = (slo.min() - 0.10, slo.max() + 0.10)
    zi = (klat >= zl[0]) & (klat <= zl[1])
    zj = (klon >= zo[0]) & (klon <= zo[1])
    zext = [klon[zj].min(), klon[zj].max(), klat[zi].min(), klat[zi].max()]
    zLON, zLAT = np.meshgrid(klon[zj], klat[zi])
    zel = elev[np.ix_(zi, zj)]

    def crop(F):
        return F[np.ix_(zi, zj)]

    tn = TwoSlopeNorm(vmin=-9, vcenter=0, vmax=3)
    tm = "RdBu_r"

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 11), constrained_layout=True)
    imE = ax[0, 0].imshow(crop(elev), origin="lower", extent=zext, cmap="terrain", aspect="auto")
    ax[0, 0].scatter(slo, sla, c="k", s=14, zorder=5)
    ax[0, 0].set_title("① Terrain / MNT — OÙ le froid s'accumule", fontsize=11)
    fig.colorbar(imE, ax=ax[0, 0], shrink=0.8, label="altitude (m)")
    ax[0, 1].contour(
        zLON, zLAT, zel, levels=[400, 600, 800, 1000, 1200], colors="0.6", linewidths=0.4
    )
    imS = ax[0, 1].scatter(
        slo, sla, c=sv, cmap=tm, norm=tn, s=120, edgecolors="k", linewidths=0.7, zorder=5
    )
    ax[0, 1].set_xlim(zo)
    ax[0, 1].set_ylim(zl)
    ax[0, 1].set_title("② Stations Sencrop seules — COMBIEN il fait froid", fontsize=11)
    fig.colorbar(imS, ax=ax[0, 1], shrink=0.8, label="Tmin obs (°C)")
    imB = ax[1, 0].imshow(crop(bg), origin="lower", extent=zext, cmap=tm, norm=tn, aspect="auto")
    ax[1, 0].contour(
        zLON, zLAT, zel, levels=[400, 600, 800, 1000, 1200], colors="k", linewidths=0.3, alpha=0.3
    )
    ax[1, 0].scatter(
        slo, sla, c=sv, cmap=tm, norm=tn, s=55, edgecolors="k", linewidths=0.5, zorder=5
    )
    ax[1, 0].set_title("③ CERRA 5,5 km (background) — rate les poches froides", fontsize=11)
    imO = ax[1, 1].imshow(crop(oi), origin="lower", extent=zext, cmap=tm, norm=tn, aspect="auto")
    ax[1, 1].contour(
        zLON, zLAT, zel, levels=[400, 600, 800, 1000, 1200], colors="k", linewidths=0.3, alpha=0.3
    )
    ax[1, 1].scatter(
        slo, sla, c=sv, cmap=tm, norm=tn, s=55, edgecolors="k", linewidths=0.5, zorder=5
    )
    ax[1, 1].set_title("④ CERRA-OI PRODUCTION (assimilée + lissée) = ① × ②", fontsize=11)
    fig.colorbar(imO, ax=ax[1, :], shrink=0.6, label="Tmin nocturne (°C)", location="right")
    for a in (ax[0, 0], ax[1, 0], ax[1, 1]):
        a.set_xticks([])
        a.set_yticks([])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    fig.suptitle(
        "Décomposition des contributions — méthode de production (gel 14→15 fév 2025, Baronnies)\n"
        "le terrain dit OÙ, les stations DISENT COMBIEN, l'assimilation OI fusionne les deux",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Pipeline de production : CERRA 5,5 km (background) → downscaling U-Net léger (structure) → "
        "assimilation OI terrain-aware Sencrop (H×V) → lissage post-traitement (#103, σ≈1,5 km). "
        "Détection portée par l'assimilation ; U-Net = structure fine (§5.2).",
        ha="center",
        fontsize=8.5,
        style="italic",
        wrap=True,
    )
    for name in ("F1_decomposition", "F6_karpossr_resolution"):
        for ext in ("png", "pdf"):
            fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches="tight")
    print("wrote F1_decomposition + F6_karpossr_resolution (png+pdf) ->", OUTDIR)


if __name__ == "__main__":
    main()
