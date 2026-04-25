"""
Validation GABLS3 — comparaison pmap/PHYEX vs profils intercomparison.

Référence :
    Bosveld, F. C., et al. (2014). The third GABLS intercomparison case for
    evaluation studies of boundary-layer models. Part B: Results and process
    understanding. Boundary-Layer Meteorology, 152(1–2), 157–187.
    DOI: 10.1007/s10546-014-9919-1

Usage :
    python validate_gabls3.py
    python validate_gabls3.py --output-dir output/gabls3/ --plot

Critères d'acceptation :
    RMSE profil θ  < 0.5 K  à t = 9h (21:00 UTC)
    RMSE profil u  < 1.0 m/s
    Hauteur jet BL  100–300 m
    Amplitude jet   7–11 m/s
    Base inversion  80–200 m
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Enveloppe des modèles GABLS3 (Bosveld et al. 2014, Fig. 5 — t = 9h = 21:00 UTC)
# Valeurs extraites de la figure (percentiles 25–75 des ~30 modèles participants)
# z en m, θ en K, u en m/s
# ---------------------------------------------------------------------------
_GABLS3_THETA_Z = np.array([
    0, 10, 20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000,
])
_GABLS3_THETA_P25 = np.array([
    280.0, 280.2, 280.5, 281.0, 281.5, 282.0, 282.8, 284.5, 286.0,
    287.5, 288.8, 291.0, 292.8, 294.0, 296.5, 298.5,
])
_GABLS3_THETA_P75 = np.array([
    281.5, 281.8, 282.1, 282.5, 283.0, 283.5, 284.2, 285.5, 287.0,
    288.5, 289.8, 292.0, 293.8, 295.0, 297.5, 299.5,
])

_GABLS3_U_Z = np.array([
    0, 10, 20, 40, 80, 100, 150, 200, 300, 400, 600, 800, 1000,
])
_GABLS3_U_P25 = np.array([
    0.0, 2.0, 3.5, 5.5, 7.5, 8.5, 9.0, 8.5, 7.5, 7.0, 6.5, 6.5, 7.0,
])
_GABLS3_U_P75 = np.array([
    0.0, 3.0, 5.0, 7.5, 9.5, 10.5, 10.5, 10.0, 9.0, 8.5, 8.0, 8.0, 8.5,
])


def load_pmap_output(output_dir: Path, eval_time: str = "2006-07-01T21:00:00") -> xr.Dataset:
    """Charge la sortie PMAP et sélectionne le pas de temps d'évaluation (t = 9h)."""
    ds = xr.open_dataset(output_dir / "output.nc")
    return ds.sel(time=eval_time, method="nearest")


def horizontal_mean(ds: xr.Dataset, field: str) -> np.ndarray:
    """Moyenne horizontale (x, y) d'un champ 3D → profil vertical."""
    da = ds[field]
    dims = [d for d in da.dims if d in ("x", "y", "latitude", "longitude")]
    return da.mean(dim=dims).values


def compute_rmse(sim: np.ndarray, ref_p25: np.ndarray, ref_p75: np.ndarray) -> float:
    """RMSE par rapport au centre de l'enveloppe GABLS3 (médiane P25/P75)."""
    ref_median = 0.5 * (ref_p25 + ref_p75)
    return float(np.sqrt(np.mean((sim - ref_median) ** 2)))


def find_llj(u_profile: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """
    Détecte le jet basse couche (LLJ) : maximum de vent dans les 1000 m.
    Retourne (hauteur m, amplitude m/s).
    """
    mask = z <= 1000.0
    idx = int(np.argmax(u_profile[mask]))
    return float(z[mask][idx]), float(u_profile[mask][idx])


def find_inversion_base(theta_profile: np.ndarray, z: np.ndarray, dtheta_min: float = 0.01) -> float:
    """
    Base de l'inversion thermique : première altitude où dθ/dz > dtheta_min K/m.
    """
    dtheta_dz = np.gradient(theta_profile, z)
    above_surface = z > 5.0
    candidates = np.where(above_surface & (dtheta_dz > dtheta_min))[0]
    if len(candidates) == 0:
        return float("nan")
    return float(z[candidates[0]])


def validate(output_dir: Path, plot: bool = False) -> dict:
    """
    Valide le run GABLS3 pmap vs l'enveloppe d'intercomparison.
    Retourne un dict avec les métriques et un booléen `passed`.
    """
    ds_eval = load_pmap_output(output_dir)

    z = ds_eval["z"].values if "z" in ds_eval else ds_eval["altitude"].values
    theta = horizontal_mean(ds_eval, "theta_total")
    u = horizontal_mean(ds_eval, "uvelx_total")
    v = horizontal_mean(ds_eval, "uvely_total")
    wind_speed = np.sqrt(u ** 2 + v ** 2)

    # Interpoler les profils de référence sur la grille pmap
    theta_p25_interp = np.interp(z, _GABLS3_THETA_Z, _GABLS3_THETA_P25)
    theta_p75_interp = np.interp(z, _GABLS3_THETA_Z, _GABLS3_THETA_P75)
    u_p25_interp = np.interp(z, _GABLS3_U_Z, _GABLS3_U_P25)
    u_p75_interp = np.interp(z, _GABLS3_U_Z, _GABLS3_U_P75)

    # Métriques
    rmse_theta = compute_rmse(theta, theta_p25_interp, theta_p75_interp)
    rmse_u = compute_rmse(wind_speed, u_p25_interp, u_p75_interp)
    llj_height, llj_speed = find_llj(wind_speed, z)
    inv_base = find_inversion_base(theta, z)

    # Critères d'acceptation
    criteria = {
        "rmse_theta_ok":   rmse_theta < 0.5,
        "rmse_u_ok":       rmse_u < 1.0,
        "llj_height_ok":   100.0 <= llj_height <= 300.0,
        "llj_speed_ok":    7.0 <= llj_speed <= 11.0,
        "inv_base_ok":     80.0 <= inv_base <= 200.0,
    }
    passed = all(criteria.values())

    results = {
        "passed": passed,
        "rmse_theta_K":    round(rmse_theta, 3),
        "rmse_u_ms":       round(rmse_u, 3),
        "llj_height_m":    round(llj_height, 1),
        "llj_speed_ms":    round(llj_speed, 2),
        "inversion_base_m": round(inv_base, 1),
        "criteria":        criteria,
    }

    _print_report(results)

    if plot:
        _plot_profiles(z, theta, wind_speed, theta_p25_interp, theta_p75_interp,
                       u_p25_interp, u_p75_interp, output_dir)

    return results


def _print_report(r: dict) -> None:
    ok = "✓" if r["passed"] else "✗"
    print(f"\nGABLS3 validation — pmap/PHYEX  {ok}")
    print(f"  RMSE θ  : {r['rmse_theta_K']:.3f} K   (< 0.5 K)")
    print(f"  RMSE u  : {r['rmse_u_ms']:.3f} m/s  (< 1.0 m/s)")
    print(f"  LLJ z   : {r['llj_height_m']:.0f} m   (100–300 m)")
    print(f"  LLJ |V| : {r['llj_speed_ms']:.2f} m/s  (7–11 m/s)")
    print(f"  Inversion base : {r['inversion_base_m']:.0f} m  (80–200 m)")
    print()
    for k, v in r["criteria"].items():
        print(f"  {'✓' if v else '✗'} {k}")
    print()
    if r["passed"]:
        print("→ Run VALIDE — pmap/PHYEX apte pour la campagne Drôme-Ardèche.")
    else:
        print("→ Run INVALIDE — ajuster les paramètres de turbulence PHYEX avant la campagne.")


def _plot_profiles(z, theta, wind, theta_p25, theta_p75, u_p25, u_p75, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non installé — plot ignoré.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    z_plot = z[z <= 1500]

    ax1.fill_betweenx(z_plot, theta_p25[z <= 1500], theta_p75[z <= 1500],
                      alpha=0.3, color="gray", label="Enveloppe GABLS3 P25–P75")
    ax1.plot(theta[z <= 1500], z_plot, "b-", lw=2, label="pmap/PHYEX")
    ax1.set_xlabel("θ (K)")
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Température potentielle — t = 9h (21:00 UTC)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.fill_betweenx(z_plot, u_p25[z <= 1500], u_p75[z <= 1500],
                      alpha=0.3, color="gray", label="Enveloppe GABLS3 P25–P75")
    ax2.plot(wind[z <= 1500], z_plot, "r-", lw=2, label="pmap/PHYEX")
    ax2.set_xlabel("|V| (m/s)")
    ax2.set_title("Vitesse du vent — jet basse couche")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "GABLS3 — Bosveld et al. (2014) BLM 152, 157–187\n"
        "Cabauw, 1–2 juillet 2006, couche limite nocturne stable",
        fontsize=10,
    )
    plt.tight_layout()
    out = out_dir / "gabls3_validation.png"
    plt.savefig(out, dpi=150)
    print(f"Figure sauvegardée : {out}")


def main():
    parser = argparse.ArgumentParser(description="Validation GABLS3 pmap/PHYEX")
    parser.add_argument("--output-dir", default="output/gabls3/",
                        help="Dossier de sortie du run PMAP")
    parser.add_argument("--plot", action="store_true",
                        help="Générer les figures de profils")
    args = parser.parse_args()

    results = validate(Path(args.output_dir), plot=args.plot)
    raise SystemExit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
