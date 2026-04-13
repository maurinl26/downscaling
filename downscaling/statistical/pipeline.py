"""
Pipeline de descente d'échelle statistique ERA5 / CERRA → 1 km.

Enchaîne :
  1. Chargement ERA5 / CERRA + MNT
  2. Regrillage sur la grille fine (MNT)
  3. Correction lapse-rate (température, pression)
  4. Correction de biais QDM (température, précipitations, vent)
  5. Sauvegarde NetCDF

Usage CLI
---------
    python -m downscaling.statistical.pipeline \
        --era5-sl  data/era5/era5_sl_20210427.nc \
        --dem      data/dem/copdem_drome_100m.tif \
        --obs-ref  data/cerra/cerra_sl_ref.nc \
        --out      output/stat_downscaled_20210427.nc \
        --variables t2m tp u10 v10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ..shared.loaders import ERA5Loader, CERRALoader, DEMLoader, regrid_to_dem
from .lapse_rate import LapseRateCorrector, correct_surface_pressure, STANDARD_LAPSE_RATE
from .quantile_mapping import QuantileDeltaMapping, EmpiricalQuantileMapping

log = logging.getLogger(__name__)


class StatisticalDownscalingPipeline:
    """
    Pipeline modulaire de descente d'échelle statistique.

    Parameters
    ----------
    dem_path:
        Chemin vers le MNT haute résolution (GeoTIFF ou NetCDF).
    obs_ref_path:
        Chemin vers le champ de référence basse résolution pour la calibration
        QDM (CERRA coarse, SAFRAN regrillé…). Optionnel : si absent,
        seule la correction lapse-rate est appliquée.
    lapse_rate:
        Gradient thermique (K m⁻¹). Scalaire ou tableau (12,) mensuel.
        Défaut : -6.5e-3 K/m.
    use_qdm:
        Active la correction QDM (nécessite obs_ref_path).
    n_quantiles:
        Nombre de quantiles pour QDM/EQM.
    """

    def __init__(
        self,
        dem_path: str | Path,
        obs_ref_path: str | Path | None = None,
        lapse_rate: float | np.ndarray = STANDARD_LAPSE_RATE,
        use_qdm: bool = True,
        n_quantiles: int = 100,
    ):
        self.dem_loader = DEMLoader(dem_path)
        self.obs_ref_path = Path(obs_ref_path) if obs_ref_path else None
        self.lapse_corrector = LapseRateCorrector(lapse_rate=lapse_rate)
        self.use_qdm = use_qdm
        self.n_quantiles = n_quantiles

        # Chargés lors du premier appel à run()
        self._dem: xr.DataArray | None = None
        self._z_coarse_on_fine: xr.DataArray | None = None
        self._qdm_t: QuantileDeltaMapping | None = None
        self._qdm_tp: QuantileDeltaMapping | None = None
        self._qdm_wind: EmpiricalQuantileMapping | None = None

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def run(
        self,
        source: xr.Dataset | str | Path,
        variables: list[str] | None = None,
    ) -> xr.Dataset:
        """
        Applique la descente d'échelle statistique complète.

        Parameters
        ----------
        source:
            Dataset ERA5/CERRA (xr.Dataset) ou chemin vers un fichier NetCDF.
        variables:
            Variables à traiter. Défaut : ['t2m', 'tp', 'u10', 'v10'].

        Returns
        -------
        xr.Dataset haute résolution corrigé.
        """
        if variables is None:
            variables = ["t2m", "tp", "u10", "v10"]

        if isinstance(source, (str, Path)):
            source = xr.open_dataset(source, engine="netcdf4")

        dem = self._get_dem()
        z_coarse = self._get_coarse_orography(source)

        output_vars = {}

        # --- Température ---
        if "t2m" in variables and "t2m" in source:
            t_fine = self._regrid(source["t2m"], dem)
            z_coarse_fine = self._regrid(z_coarse, dem)
            t_fine = self.lapse_corrector.correct(t_fine, z_coarse_fine, dem)

            if self.use_qdm and self._qdm_t is not None:
                t_fine = self._qdm_t.transform(t_fine)

            output_vars["t2m"] = t_fine
            log.info("t2m : lapse-rate + QDM appliqués.")

        # --- Précipitations ---
        if "tp" in variables and "tp" in source:
            tp_fine = self._regrid(source["tp"], dem)

            if self.use_qdm and self._qdm_tp is not None:
                tp_fine = self._qdm_tp.transform(tp_fine)

            tp_fine = tp_fine.clip(min=0.0)
            output_vars["tp"] = tp_fine
            log.info("tp : QDM appliqué.")

        # --- Vent ---
        for comp in ("u10", "v10", "i10fg"):
            if comp in variables and comp in source:
                w_fine = self._regrid(source[comp], dem)
                if self.use_qdm and self._qdm_wind is not None:
                    w_fine = self._qdm_wind.transform(w_fine)
                output_vars[comp] = w_fine

        # --- Pression de surface ---
        if "sp" in source and "sp" in (variables or []):
            sp_fine = self._regrid(source["sp"], dem)
            z_coarse_fine = output_vars.get("_z_coarse_fine") or self._regrid(z_coarse, dem)
            sp_fine = correct_surface_pressure(sp_fine, z_coarse_fine, dem)
            output_vars["sp"] = sp_fine

        ds_out = xr.Dataset(output_vars)
        ds_out.attrs["downscaling_method"] = "statistical (lapse-rate + QDM)"
        ds_out.attrs["dem_source"] = str(self.dem_loader.dem_path)
        return ds_out

    def calibrate(
        self,
        modeled_ref: xr.Dataset,
        observed_ref: xr.Dataset,
    ) -> "StatisticalDownscalingPipeline":
        """
        Calibre les correcteurs QDM sur une période de référence.

        Parameters
        ----------
        modeled_ref:
            ERA5 / CERRA basse résolution sur la période de référence.
        observed_ref:
            CERRA haute résolution / SAFRAN — même période et même grille fine
            que le MNT (ou interpolé dessus).

        Returns
        -------
        self (pour chaînage).
        """
        dem = self._get_dem()

        if "t2m" in modeled_ref and "t2m" in observed_ref:
            log.info("Calibration QDM température…")
            t_mod_fine = self._regrid(modeled_ref["t2m"], dem)
            t_obs_fine = self._regrid(observed_ref["t2m"], dem)
            self._qdm_t = QuantileDeltaMapping(kind="delta", n_quantiles=self.n_quantiles)
            self._qdm_t.fit(t_mod_fine, t_obs_fine)

        if "tp" in modeled_ref and "tp" in observed_ref:
            log.info("Calibration QDM précipitations…")
            tp_mod_fine = self._regrid(modeled_ref["tp"], dem)
            tp_obs_fine = self._regrid(observed_ref["tp"], dem)
            self._qdm_tp = QuantileDeltaMapping(kind="ratio", n_quantiles=self.n_quantiles)
            self._qdm_tp.fit(tp_mod_fine, tp_obs_fine)

        for comp in ("u10", "v10"):
            if comp in modeled_ref and comp in observed_ref:
                log.info(f"Calibration EQM vent {comp}…")
                w_mod = self._regrid(modeled_ref[comp], dem)
                w_obs = self._regrid(observed_ref[comp], dem)
                self._qdm_wind = EmpiricalQuantileMapping(n_quantiles=self.n_quantiles)
                self._qdm_wind.fit(w_mod, w_obs)
                break  # Un seul correcteur vent

        return self

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _get_dem(self) -> xr.DataArray:
        if self._dem is None:
            self._dem = self.dem_loader.load()
        return self._dem

    def _get_coarse_orography(self, ds: xr.Dataset) -> xr.DataArray:
        """Extrait ou construit l'orographie de la source."""
        for name in ("z", "orog", "oro"):
            if name in ds:
                da = ds[name]
                if "time" in da.dims:
                    da = da.isel(time=0)
                if name == "z":
                    da = da / 9.80665
                return da.rename("orog_source")
        # Si absente : zéro (pas de correction d'altitude source)
        import warnings
        warnings.warn("Orographie source absente : on assume z_source=0 m.")
        lat = ds.coords.get("latitude", ds.coords.get("lat"))
        lon = ds.coords.get("longitude", ds.coords.get("lon"))
        return xr.DataArray(
            np.zeros((len(lat), len(lon))),
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )

    @staticmethod
    def _regrid(da: xr.DataArray, dem: xr.DataArray) -> xr.DataArray:
        return regrid_to_dem(da, dem, method="linear")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Descente d'échelle statistique ERA5/CERRA → 1 km"
    )
    p.add_argument("--era5-sl", required=True, help="Fichier ERA5 single-level")
    p.add_argument("--dem", required=True, help="Fichier MNT (GeoTIFF ou NetCDF)")
    p.add_argument("--obs-ref", default=None, help="Référence pour calibration QDM")
    p.add_argument("--mod-ref", default=None, help="Modèle de référence pour calibration QDM")
    p.add_argument("--out", required=True, help="Fichier NetCDF de sortie")
    p.add_argument(
        "--variables", nargs="+", default=["t2m", "tp", "u10", "v10"],
        help="Variables à traiter"
    )
    p.add_argument("--no-qdm", action="store_true", help="Désactive la correction QDM")
    p.add_argument("--n-quantiles", type=int, default=100)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    pipeline = StatisticalDownscalingPipeline(
        dem_path=args.dem,
        obs_ref_path=args.obs_ref,
        use_qdm=not args.no_qdm,
        n_quantiles=args.n_quantiles,
    )

    if args.obs_ref and args.mod_ref:
        log.info("Calibration QDM…")
        obs_ref = xr.open_dataset(args.obs_ref, engine="netcdf4")
        mod_ref = xr.open_dataset(args.mod_ref, engine="netcdf4")
        pipeline.calibrate(mod_ref, obs_ref)

    log.info(f"Traitement de {args.era5_sl}…")
    ds_out = pipeline.run(args.era5_sl, variables=args.variables)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(args.out)
    log.info(f"Sortie écrite dans {args.out}")


if __name__ == "__main__":
    main()
