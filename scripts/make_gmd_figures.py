#!/usr/bin/env python
"""Génère les figures du manuscrit GMD (descente d'échelle du gel).

Reproductible : lit uniquement des données déjà persistées et écrit des figures
PNG (300 dpi) + PDF vectoriel dans ``docs/methodology/figures/``.

Sources de données (voir aussi la provenance annotée sur chaque figure) :
  * LOO KarposSLR / KarposSR : JSON persistés sous ``docs/methodology/figures/loo_json/``
    (copies des sorties de validation leave-one-station-out, seuil -2,2 °C,
    aussi disponibles sur S3 ``analyses/c5_karpos_slr/`` et ``analyses/c5_karpos_sr/``).
  * CERRA brut (hindcast) : CSI 0,17 — note vault « Métriques trackées — KarposSLR
    vs KarposSR » §6 (non recalculé ici, absent des JSON).
  * AROME : run live de ``downscaling/scripts/arome_forecast_skill.py``
    (2026-07-27), identique à la note vault §6.

Usage :
    .venv/bin/python scripts/make_gmd_figures.py [--data-dir DIR] [--out-dir DIR]

Aucun accès réseau requis.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Style publication — palette colorblind-safe (Okabe & Ito, 2008)
# --------------------------------------------------------------------------- #
OKABE_ITO = {
    "noir": "#000000",
    "orange": "#E69F00",
    "bleu_ciel": "#56B4E9",
    "vert": "#009E73",
    "jaune": "#F0E442",
    "bleu": "#0072B2",
    "vermillon": "#D55E00",
    "violet": "#CC79A7",
    "gris": "#999999",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.autolayout": False,
})

YEARS = [2022, 2023, 2024, 2025]
THR_KEY = "-2.2"  # seuil assurance -2,2 °C (LT10 abricot)

# AROME — run live arome_forecast_skill.py (2026-07-27), == note vault §6.
AROME = {
    "brut": {"label": "AROME brut\n(prévision J-1)", "POD": 0.05, "FAR": 0.12, "CSI": 0.05},
    "sencrop": {"label": "AROME\n+ calibration Sencrop", "POD": 0.50, "FAR": 0.53, "CSI": 0.32},
    "split_label": "Split temporel opérationnel : train 2024 → test 2025, CSI 0,34",
}

# CERRA brut hindcast — note vault « Métriques trackées » §6 (non recalculé).
CERRA_CSI = 0.17


# --------------------------------------------------------------------------- #
# Chargement des JSON LOO
# --------------------------------------------------------------------------- #
def load_year(data_dir: Path, lot: str, year: int) -> dict:
    p = data_dir / f"lot_{lot}" / f"{year}.loo.json"
    with open(p) as f:
        return json.load(f)


def pooled_csi(data_dir: Path, lot: str, mode: str = "station") -> dict:
    """CSI pooled = contingences sommées sur toutes les années (seuil -2,2 °C)."""
    tp = fp = fn = 0
    for y in YEARS:
        agg = load_year(data_dir, lot, y)["modes"][mode]["aggregate"]
        tp += agg["TP"]
        fp += agg["FP"]
        fn += agg["FN"]
    return {
        "POD": tp / (tp + fn),
        "FAR": fp / (tp + fp),
        "CSI": tp / (tp + fp + fn),
        "TP": tp, "FP": fp, "FN": fn,
    }


def per_station_bias(data_dir: Path, lot: str, mode: str = "station") -> dict:
    """Biais résiduel moyen (pondéré par n_pairs) par station, sur les 4 saisons."""
    acc: dict[int, list[tuple[float, int]]] = {}
    for y in YEARS:
        for r in load_year(data_dir, lot, y)["modes"][mode]["per_station"]:
            b = r.get("bias")
            if b is None or (isinstance(b, float) and math.isnan(b)):
                continue
            acc.setdefault(r["station_id"], []).append((b, r["n_pairs"]))
    out = {}
    for sid, vals in acc.items():
        w = sum(n for _, n in vals)
        out[sid] = sum(b * n for b, n in vals) / w
    return out


def prov(fig, text: str, y: float = 0.015):
    """Note de provenance en pied de figure (dans la marge basse réservée)."""
    fig.text(0.01, y, text, fontsize=6.5, color=OKABE_ITO["gris"],
             ha="left", va="bottom", wrap=True)


def save(fig, out_dir: Path, name: str, bottom: float = 0.10):
    """Réserve une bande basse (marge) pour la provenance puis écrit PNG+PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  écrit : {name}.png / {name}.pdf")


# --------------------------------------------------------------------------- #
# F5 — REV (résultat décisionnel) [PRIORITAIRE]
# --------------------------------------------------------------------------- #
def fig_f5_rev(data_dir: Path, out_dir: Path, year: int = 2023, mode: str = "station"):
    rev = load_year(data_dir, "c", year)["modes"][mode]["rev"]
    alphas = np.asarray(rev["alphas"])
    venv = np.asarray(rev["V_env"])
    a_lo, a_hi = 0.02, 0.10  # plage coût-perte réelle du vigneron (α = C/L)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xscale("log")

    # Zone V > 0 (le produit crée de la valeur)
    ax.fill_between(alphas, 0, venv, where=(venv > 0), color=OKABE_ITO["vert"],
                    alpha=0.15, label="V(α) > 0 : le produit crée de la valeur")
    ax.axhline(0, color=OKABE_ITO["noir"], lw=0.8)

    # Enveloppe V_env(α)
    ax.plot(alphas, venv, color=OKABE_ITO["bleu"], lw=2.2,
            label="Enveloppe de valeur V(α), KarposSR")

    # Plage coût-perte réelle α ∈ [0,02 ; 0,10]
    ax.axvspan(a_lo, a_hi, color=OKABE_ITO["orange"], alpha=0.12)
    for a in (a_lo, a_hi):
        ax.axvline(a, color=OKABE_ITO["orange"], lw=1.3, ls="--")
    ax.text(math.sqrt(a_lo * a_hi), 0.04, "plage coût-perte\nréelle  α ∈ [0,02 ; 0,10]",
            ha="center", va="bottom", fontsize=8.5, color=OKABE_ITO["vermillon"])

    # Pic de valeur
    a_peak, v_max = rev["alpha_at_peak"], rev["V_max"]
    ax.plot([a_peak], [v_max], "o", color=OKABE_ITO["vermillon"], ms=7, zorder=5)
    ax.annotate(f"V$_{{max}}$ = {v_max:.2f}\nà α = {a_peak:.3f}",
                xy=(a_peak, v_max), xytext=(a_peak * 1.5, v_max - 0.22),
                fontsize=8.5, color=OKABE_ITO["vermillon"],
                arrowprops=dict(arrowstyle="->", color=OKABE_ITO["vermillon"], lw=1))
    ax.axvline(rev["base_rate"], color=OKABE_ITO["gris"], lw=1, ls=":")
    ax.text(rev["base_rate"], 0.97, f"base rate s = {rev['base_rate']:.3f}",
            rotation=90, va="top", ha="right", fontsize=7.5, color=OKABE_ITO["gris"])

    # Annotations €/ha : points de la grille (C, L) tombant dans la plage réelle
    euros = [e for e in rev["euros_per_ha"] if a_lo <= e["alpha"] <= a_hi]
    euros.sort(key=lambda e: e["alpha"])
    ax.scatter([e["alpha"] for e in euros], [e["V"] for e in euros],
               color=OKABE_ITO["violet"], s=28, zorder=6, marker="D",
               label="€/ha évités (grille coût C / récolte L)")
    lo_e = min(e["euros_per_ha"] for e in euros)
    hi_e = max(e["euros_per_ha"] for e in euros)
    ax.text(0.98, 0.06,
            f"Sur la plage réelle : {lo_e:.0f}–{hi_e:.0f} €/ha·an évités\n"
            f"(V ≈ {min(e['V'] for e in euros):.2f}–{max(e['V'] for e in euros):.2f} "
            f"de la prévision parfaite)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=OKABE_ITO["violet"], alpha=0.9))

    ax.set_xlabel("Ratio coût-perte  α = C / L  (échelle log)")
    ax.set_ylabel("Valeur économique relative  V(α)")
    ax.set_title(f"F5 — Valeur décisionnelle (REV), KarposSR {year}, mode {mode}")
    ax.set_ylim(-0.35, 1.0)
    ax.set_xlim(alphas.min(), min(alphas.max(), 0.5))
    ax.legend(loc="upper left", fontsize=8.5)
    prov(fig, "Provenance : JSON LOO persisté (KarposSR, modes.station.rev). "
              "Modèle coût-perte Richardson 2000 / Wilks.")
    save(fig, out_dir, "F5_rev_karpos_sr")


# --------------------------------------------------------------------------- #
# F4 — AROME (résultat-phare) [PRIORITAIRE]
# --------------------------------------------------------------------------- #
def fig_f4_arome(out_dir: Path):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    groups = ["POD", "CSI"]
    x = np.arange(len(groups))
    w = 0.36
    brut = [AROME["brut"]["POD"], AROME["brut"]["CSI"]]
    sen = [AROME["sencrop"]["POD"], AROME["sencrop"]["CSI"]]

    b1 = ax.bar(x - w / 2, brut, w, label=AROME["brut"]["label"],
                color=OKABE_ITO["gris"])
    b2 = ax.bar(x + w / 2, sen, w, label=AROME["sencrop"]["label"],
                color=OKABE_ITO["bleu"])
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)

    # Annotation ×10 (POD) et gains
    for i, (lo, hi) in enumerate(zip(brut, sen)):
        factor = hi / lo if lo else float("inf")
        ax.annotate("", xy=(x[i] + w / 2, hi + 0.03), xytext=(x[i] - w / 2, lo + 0.03),
                    arrowprops=dict(arrowstyle="->", color=OKABE_ITO["vermillon"], lw=1.6))
        ax.text(x[i], hi + 0.06, f"×{factor:.0f}", ha="center", fontsize=12,
                fontweight="bold", color=OKABE_ITO["vermillon"])

    ax.set_xticks(x)
    ax.set_xticklabels(["POD (détection)", "CSI (skill global)"])
    ax.set_ylabel("Score (seuil −2,2 °C, LOO station-out)")
    ax.set_ylim(0, 0.75)
    ax.set_title("F4 — AROME à la parcelle : la calibration Sencrop débloque la détection")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.03, 0.97, AROME["split_label"], transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, style="italic", color=OKABE_ITO["vert"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=OKABE_ITO["vert"], alpha=0.9))
    prov(fig, "Provenance : run live arome_forecast_skill.py (2026-07-27, "
              "Open-Meteo Historical Forecast × Sencrop, 48 stations Drôme, "
              "2024-2025) ; == note vault §6.")
    save(fig, out_dir, "F4_arome_sencrop")


# --------------------------------------------------------------------------- #
# F2 — skill hindcast par étage
# --------------------------------------------------------------------------- #
def fig_f2_skill(data_dir: Path, out_dir: Path):
    b = pooled_csi(data_dir, "b", "station")
    c = pooled_csi(data_dir, "c", "station")
    stages = [
        ("CERRA brut\n(5,5 km)", CERRA_CSI, OKABE_ITO["gris"], "note §6"),
        ("Calibration\nstatistique (KarposSLR)", b["CSI"], OKABE_ITO["bleu_ciel"], "JSON pooled"),
        ("Deep learning\n+ supervision (KarposSR)", c["CSI"], OKABE_ITO["bleu"], "JSON pooled"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(stages))
    bars = ax.bar(x, [s[1] for s in stages], 0.6, color=[s[2] for s in stages])
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10, fontweight="bold")

    # flèches de gain entre étages
    vals = [s[1] for s in stages]
    for i in range(len(vals) - 1):
        ax.annotate("", xy=(x[i + 1], vals[i + 1] + 0.015),
                    xytext=(x[i], vals[i] + 0.015),
                    arrowprops=dict(arrowstyle="->", color=OKABE_ITO["vermillon"],
                                    lw=1.4, connectionstyle="arc3,rad=-0.25"))
    ax.text(0.5, vals[1] + 0.09, f"×{vals[1] / vals[0]:.1f}", ha="center",
            fontsize=10, color=OKABE_ITO["vermillon"], fontweight="bold")
    ax.text(1.5, vals[2] + 0.06, f"×{vals[2] / vals[0]:.1f} vs CERRA",
            ha="center", fontsize=10, color=OKABE_ITO["vermillon"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages])
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylabel("CSI (LOO station-out, seuil −2,2 °C, pooled 2022-2025)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("F2 — Skill de descente d'échelle par étage (hindcast)")
    prov(fig, "Provenance : CERRA = note vault §6 (0,17) ; KarposSLR / KarposSR = CSI "
              "pooled (contingences sommées) depuis JSON LOO persisté. "
              f"KarposSLR CSI={b['CSI']:.3f}, KarposSR CSI={c['CSI']:.3f}.")
    save(fig, out_dir, "F2_skill_hindcast")


# --------------------------------------------------------------------------- #
# F3 — biais résiduel par station, avant/après (KarposSLR → KarposSR)
# --------------------------------------------------------------------------- #
def fig_f3_bias(data_dir: Path, out_dir: Path):
    b = per_station_bias(data_dir, "b")
    c = per_station_bias(data_dir, "c")
    common = sorted(set(b) & set(c))
    bx = np.array([b[s] for s in common])
    cy = np.array([c[s] for s in common])
    mab_b = np.mean(np.abs(bx))
    mab_c = np.mean(np.abs(cy))

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    lim = max(np.abs(np.concatenate([bx, cy]))) * 1.1
    ax.axhline(0, color=OKABE_ITO["gris"], lw=0.8)
    ax.axvline(0, color=OKABE_ITO["gris"], lw=0.8)
    ax.plot([-lim, lim], [-lim, lim], ls=":", color=OKABE_ITO["noir"], lw=0.9,
            label="y = x (biais inchangé)")
    # bande |biais| < 0,5 °C
    ax.axhspan(-0.5, 0.5, color=OKABE_ITO["vert"], alpha=0.10)
    ax.text(-lim * 0.95, 0.0, "|biais KarposSR| < 0,5 °C", fontsize=8,
            va="center", color=OKABE_ITO["vert"])

    ax.scatter(bx, cy, color=OKABE_ITO["bleu"], s=42, alpha=0.8, edgecolor="white",
               linewidth=0.5, zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Biais station — KarposSLR (statistique)  [°C]")
    ax.set_ylabel("Biais station — KarposSR (deep learning)  [°C]")
    ax.set_title("F3 — Resserrement du biais par station (KarposSLR → KarposSR)")
    ax.text(0.03, 0.97,
            f"n = {len(common)} stations\n"
            f"biais absolu moyen :\n  KarposSLR = {mab_b:.2f} °C\n  KarposSR = {mab_c:.2f} °C "
            f"(−{100 * (1 - mab_c / mab_b):.0f} %)",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=OKABE_ITO["gris"], alpha=0.9))
    ax.legend(loc="lower right", fontsize=8.5)
    prov(fig, "Provenance : JSON LOO persisté (modes.station.per_station.bias), "
              "moyenne pondérée par n_pairs sur 2022-2025.")
    save(fig, out_dir, "F3_biais_stations", bottom=0.08)


def main():
    here = Path(__file__).resolve().parent
    default_out = here.parent / "docs" / "methodology" / "figures"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=default_out / "loo_json")
    ap.add_argument("--out-dir", type=Path, default=default_out)
    a = ap.parse_args()
    print(f"Données : {a.data_dir}\nSortie  : {a.out_dir}\n")
    fig_f5_rev(a.data_dir, a.out_dir)
    fig_f4_arome(a.out_dir)
    fig_f2_skill(a.data_dir, a.out_dir)
    fig_f3_bias(a.data_dir, a.out_dir)
    print("\nTerminé.")


if __name__ == "__main__":
    main()
