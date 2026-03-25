"""
Loaders for ERA5, CERRA reanalyses and digital elevation models.

Variables retournées en convention xarray Dataset avec coordonnées
(time, lat, lon) ou (time, y, x) selon la source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# ERA5 loader
# ---------------------------------------------------------------------------

#: Variables single-level ERA5 utiles pour l'assurance paramétrique
ERA5_SL_VARS = {
    "2m_temperature": "t2m",
    "minimum_2m_temperature_since_previous_post_processing": "mn2t",
    "maximum_2m_temperature_since_previous_post_processing": "mx2t",
    "total_precipitation": "tp",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "instantaneous_10m_wind_gust": "i10fg",
    "surface_pressure": "sp",
    "2m_dewpoint_temperature": "d2m",
}


class ERA5Loader:
    """Chargement de fichiers NetCDF ERA5 (single levels)."""

    def __init__(self, sl_path: str | Path, pl_path: str | Path | None = None):
        self.sl_path = Path(sl_path)
        self.pl_path = Path(pl_path) if pl_path else None

    def load_sl(self, variables: list[str] | None = None) -> xr.Dataset:
        """
        Charge les champs de surface ERA5.

        Parameters
        ----------
        variables:
            Noms courts (clés de ERA5_SL_VARS) à charger. None = tous.

        Returns
        -------
        xr.Dataset avec coordonnées (time, latitude, longitude).
        """
        ds = xr.open_dataset(self.sl_path, engine="netcdf4")
        if variables is not None:
            short = set(variables)
            # Accepte aussi les noms longs ERA5
            keep = [v for v in ds.data_vars if v in short or ERA5_SL_VARS.get(v) in short]
            ds = ds[keep]
        return ds

    def load_pl(self, pressure_levels: list[int] | None = None) -> xr.Dataset:
        """Charge les niveaux de pression ERA5 (optionnel)."""
        if self.pl_path is None:
            raise FileNotFoundError("Aucun fichier ERA5 pression spécifié.")
        ds = xr.open_dataset(self.pl_path, engine="netcdf4")
        if pressure_levels is not None:
            ds = ds.sel(pressure_level=pressure_levels)
        return ds

    def orography(self) -> xr.DataArray:
        """
        Extrait l'orographie ERA5 (champ 'z' / geopotential au niveau z=0).

        Le champ ERA5 'orography' (z) est en m² s⁻² ; on divise par g = 9.80665.
        """
        ds = xr.open_dataset(self.sl_path, engine="netcdf4")
        if "z" in ds:
            return (ds["z"].isel(time=0) / 9.80665).rename("orog_era5")
        if "orog" in ds:
            return ds["orog"].isel(time=0).rename("orog_era5")
        raise KeyError("Champ orographie 'z' ou 'orog' absent du fichier ERA5.")


# ---------------------------------------------------------------------------
# CERRA loader
# ---------------------------------------------------------------------------

#: Variables single-level CERRA typiques
CERRA_SL_VARS = {
    "2m_temperature": "t2m",
    "total_precipitation": "tp",
    "10m_wind_speed": "si10",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "surface_pressure": "sp",
    "2m_relative_humidity": "r2",
}


class CERRALoader:
    """
    Chargement de fichiers NetCDF CERRA (Copernicus European Regional ReAnalysis).

    CERRA est fourni en projection Lambert (grille ~5.5 km).
    Les coordonnées peuvent être (y, x) ou (latitude, longitude) selon le
    niveau de traitement CDS.
    """

    def __init__(self, sl_path: str | Path, pl_path: str | Path | None = None):
        self.sl_path = Path(sl_path)
        self.pl_path = Path(pl_path) if pl_path else None

    def load_sl(self, variables: list[str] | None = None) -> xr.Dataset:
        ds = xr.open_dataset(self.sl_path, engine="netcdf4")
        if variables is not None:
            short = set(variables)
            keep = [v for v in ds.data_vars if v in short or CERRA_SL_VARS.get(v) in short]
            ds = ds[keep]
        return ds

    def load_pl(self, pressure_levels: list[int] | None = None) -> xr.Dataset:
        if self.pl_path is None:
            raise FileNotFoundError("Aucun fichier CERRA pression spécifié.")
        ds = xr.open_dataset(self.pl_path, engine="netcdf4")
        if pressure_levels is not None:
            ds = ds.sel(pressure_level=pressure_levels)
        return ds

    def orography(self) -> xr.DataArray:
        """Orographie CERRA (m)."""
        ds = xr.open_dataset(self.sl_path, engine="netcdf4")
        for name in ("z", "orog", "oro"):
            if name in ds:
                da = ds[name].isel(time=0) if "time" in ds[name].dims else ds[name]
                if name == "z":
                    da = da / 9.80665
                return da.rename("orog_cerra")
        raise KeyError("Orographie introuvable dans le fichier CERRA.")


# ---------------------------------------------------------------------------
# DEM loader
# ---------------------------------------------------------------------------

class DEMLoader:
    """
    Chargement du Modèle Numérique de Terrain (MNT).

    Supporte les formats GeoTIFF (rasterio) et NetCDF (xarray).
    Calcule automatiquement la pente et l'exposition (aspect).
    """

    def __init__(self, dem_path: str | Path):
        self.dem_path = Path(dem_path)

    def load(self) -> xr.DataArray:
        """
        Charge l'élévation.

        Returns
        -------
        xr.DataArray(y, x) ou (lat, lon) avec attributs CRS si disponibles.
        """
        suffix = self.dem_path.suffix.lower()
        if suffix in (".tif", ".tiff", ".geotiff"):
            return self._load_geotiff()
        else:
            return self._load_netcdf()

    def _load_geotiff(self) -> xr.DataArray:
        try:
            import rasterio
            from rasterio.transform import xy as rio_xy
        except ImportError as e:
            raise ImportError("rasterio requis pour les GeoTIFF : pip install rasterio") from e

        with rasterio.open(self.dem_path) as src:
            data = src.read(1).astype(np.float32)
            data[data == src.nodata] = np.nan
            rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing="ij")
            lons, lats = rasterio.transform.xy(src.transform, rows, cols)
            lons = np.array(lons)
            lats = np.array(lats)

        # Créer DataArray avec lat/lon 1D si grille régulière
        da = xr.DataArray(
            data,
            dims=["y", "x"],
            attrs={"units": "m", "long_name": "elevation", "source": str(self.dem_path)},
        )
        da = da.assign_coords(lat=(["y", "x"], lats), lon=(["y", "x"], lons))
        return da

    def _load_netcdf(self) -> xr.DataArray:
        ds = xr.open_dataset(self.dem_path, engine="netcdf4")
        for name in ("elevation", "orog", "z", "dem", "height", "topo"):
            if name in ds:
                return ds[name].rename("elevation")
        # Prend la première variable si nom inconnu
        varname = list(ds.data_vars)[0]
        return ds[varname].rename("elevation")

    def terrain_attributes(self, resolution_m: float = 100.0) -> xr.Dataset:
        """
        Calcule la pente (slope) et l'exposition (aspect) à partir de l'élévation.

        Parameters
        ----------
        resolution_m:
            Résolution en mètres du MNT (pour le calcul des gradients).

        Returns
        -------
        xr.Dataset avec variables : elevation, slope (degrés), aspect (degrés 0–360),
        curvature_plan, curvature_prof.
        """
        elev = self.load()
        z = elev.values

        # Gradients par différences finies centrées
        dz_dy, dz_dx = np.gradient(z, resolution_m, resolution_m)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)

        # Aspect : 0 = Nord, 90 = Est (convention météo)
        aspect_rad = np.arctan2(-dz_dx, dz_dy)
        aspect_deg = (np.degrees(aspect_rad) + 360) % 360

        # Courbure planiforme (influence sur ruissellement)
        dz_dxx = np.gradient(dz_dx, resolution_m, axis=1)
        dz_dyy = np.gradient(dz_dy, resolution_m, axis=0)
        curvature_plan = -(dz_dxx + dz_dyy)

        dims = elev.dims
        coords = elev.coords

        return xr.Dataset(
            {
                "elevation": xr.DataArray(z, dims=dims, coords=coords,
                                          attrs={"units": "m"}),
                "slope": xr.DataArray(slope_deg.astype(np.float32), dims=dims, coords=coords,
                                      attrs={"units": "degrees"}),
                "aspect": xr.DataArray(aspect_deg.astype(np.float32), dims=dims, coords=coords,
                                       attrs={"units": "degrees", "convention": "0=N 90=E"}),
                "curvature": xr.DataArray(curvature_plan.astype(np.float32), dims=dims,
                                          coords=coords, attrs={"units": "m-1"}),
            }
        )


# ---------------------------------------------------------------------------
# Helpers de regrillage
# ---------------------------------------------------------------------------

def regrid_to_dem(
    da: xr.DataArray,
    dem: xr.DataArray,
    method: str = "linear",
) -> xr.DataArray:
    """
    Regrille un champ basse résolution sur la grille du MNT.

    Parameters
    ----------
    da:
        Champ source ERA5 ou CERRA (dims lat, lon ou y, x).
    dem:
        DataArray du MNT cible.
    method:
        'linear' (bilinéaire) ou 'nearest'.

    Returns
    -------
    xr.DataArray sur la grille du MNT.
    """
    try:
        import xesmf as xe
    except ImportError:
        # Fallback : interp xarray (requiert lat/lon 1D)
        lat_name = "latitude" if "latitude" in da.coords else "lat"
        lon_name = "longitude" if "longitude" in da.coords else "lon"
        dem_lat = dem.coords.get("lat", dem.coords.get("latitude"))
        dem_lon = dem.coords.get("lon", dem.coords.get("longitude"))
        return da.interp(
            {lat_name: dem_lat, lon_name: dem_lon},
            method=method,
        )

    lat_src = "latitude" if "latitude" in da.coords else "lat"
    lon_src = "longitude" if "longitude" in da.coords else "lon"
    lat_dst = "lat" if "lat" in dem.coords else "latitude"
    lon_dst = "lon" if "lon" in dem.coords else "longitude"

    ds_in = da.rename({lat_src: "lat", lon_src: "lon"}).to_dataset(name="field")
    ds_out = xr.Dataset({"lat": dem[lat_dst], "lon": dem[lon_dst]})
    regridder = xe.Regridder(ds_in, ds_out, method=method)
    return regridder(ds_in)["field"]
