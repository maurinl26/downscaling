"""
Indices d'assurance paramétrique calculés sur des champs météorologiques.

Toutes les fonctions acceptent des xr.DataArray avec au minimum une dimension
'time' (fréquence horaire ou journalière). Elles renvoient des DataArray avec
la dimension temporelle agrégée ou filtrée selon l'indice.

Conventions d'unités
--------------------
- Températures : Kelvin (K) en entrée, Celsius dans les seuils par défaut
- Précipitations : m (ERA5/CERRA natif) ou mm selon `unit` argument
- Vent : m s⁻¹
"""

from __future__ import annotations

import numpy as np
import xarray as xr

_K0 = 273.15  # 0 °C en Kelvin


# ---------------------------------------------------------------------------
# Gel
# ---------------------------------------------------------------------------

def frost_days(
    tmin: xr.DataArray,
    threshold_c: float = 0.0,
    freq: str = "YS",
) -> xr.DataArray:
    """
    Nombre de jours avec Tmin < seuil.

    Parameters
    ----------
    tmin:
        Température minimale journalière (K).
    threshold_c:
        Seuil en °C (défaut 0 °C = gel).
    freq:
        Fréquence de résumé : 'YS' (annuel), 'MS' (mensuel), 'QS' (trimestriel).
    """
    tmin_c = tmin - _K0
    return (tmin_c < threshold_c).resample(time=freq).sum().rename("frost_days")


def frost_hours(
    t2m: xr.DataArray,
    threshold_c: float = 0.0,
    freq: str = "YS",
) -> xr.DataArray:
    """
    Nombre d'heures avec T2m < seuil (données horaires ERA5/CERRA).

    Pertinent pour le gel de la vigne et des cultures arboricoles.
    """
    t2m_c = t2m - _K0
    return (t2m_c < threshold_c).resample(time=freq).sum().rename("frost_hours")


def spring_frost_index(
    t2m: xr.DataArray,
    tmin: xr.DataArray,
    gdd_threshold: float = 50.0,
    frost_threshold_c: float = -2.0,
    base_temp_c: float = 5.0,
) -> xr.DataArray:
    """
    Indice de gel printanier : nombre d'heures de gel après le débourrement.

    Le débourrement est estimé par l'atteinte d'un cumul de GDD (degrés-jours de
    croissance) depuis le 1er janvier. Une fois le seuil GDD atteint, on compte
    les heures avec T2m < frost_threshold_c jusqu'à fin saison (31 juillet).

    Parameters
    ----------
    t2m:
        Température horaire 2 m (K).
    tmin:
        Tmin journalière (K) pour le calcul des GDD.
    gdd_threshold:
        GDD cumulés marquant le débourrement (défaut 50 °C·j, stade B Eichhorn-Lorenz).
    frost_threshold_c:
        Seuil de gel après débourrement (défaut -2 °C, dommages significatifs).
    base_temp_c:
        Température de base pour le calcul des GDD (défaut 5 °C).

    Returns
    -------
    xr.DataArray (spatial) — nb heures de gel post-débourrement par saison.
    """
    # GDD journaliers (approché : max(Tmin - base, 0))
    tmin_c = tmin - _K0
    gdd_daily = (tmin_c - base_temp_c).clip(min=0.0)

    # Cumul annuel : on travaille année par année
    results = []
    years = np.unique(t2m.time.dt.year.values)
    for yr in years:
        t2m_yr = t2m.sel(time=t2m.time.dt.year == yr)
        gdd_yr = gdd_daily.sel(time=gdd_daily.time.dt.year == yr)

        # Jour de débourrement : premier jour où cumul GDD dépasse le seuil
        cum_gdd = gdd_yr.cumsum("time")
        deb_mask = cum_gdd >= gdd_threshold  # shape (time, ...)

        # Crée un mask horaire post-débourrement
        if not deb_mask.any():
            results.append(xr.zeros_like(t2m_yr.isel(time=0)))
            continue

        deb_doy = int(deb_mask.argmax("time").min().values)  # conservative: earliest pixel
        t2m_post = t2m_yr.isel(time=slice(deb_doy * 24, None))  # approximation 24h/j

        frost_post = (t2m_post - _K0 < frost_threshold_c).sum("time")
        results.append(frost_post.assign_coords(time=np.datetime64(f"{yr}-01-01")))

    return xr.concat(results, dim="time").rename("spring_frost_index")


# ---------------------------------------------------------------------------
# Thermique / agronomie
# ---------------------------------------------------------------------------

def growing_degree_days(
    tmax: xr.DataArray,
    tmin: xr.DataArray,
    base_c: float = 10.0,
    cap_c: float | None = 30.0,
    freq: str = "YS",
) -> xr.DataArray:
    """
    Somme thermique (Growing Degree Days).

    GDD_j = max(0, (Tmax + Tmin) / 2 - base) plafonné à cap si défini.

    Parameters
    ----------
    tmax, tmin:
        Températures journalières max/min (K).
    base_c:
        Température de base (°C). Maïs : 10, vigne : 10, blé : 0.
    cap_c:
        Plafond (°C). None = pas de plafond.
    """
    tmean_c = (tmax + tmin) / 2.0 - _K0
    gdd = (tmean_c - base_c).clip(min=0.0)
    if cap_c is not None:
        gdd = gdd.clip(max=cap_c - base_c)
    return gdd.resample(time=freq).sum().rename("gdd")


def heat_stress_days(
    tmax: xr.DataArray,
    threshold_c: float = 35.0,
    freq: str = "YS",
) -> xr.DataArray:
    """Nombre de jours avec Tmax > seuil (stress thermique cultures)."""
    tmax_c = tmax - _K0
    return (tmax_c > threshold_c).resample(time=freq).sum().rename("heat_stress_days")


def heatwave_index(
    tmax: xr.DataArray,
    threshold_c: float = 35.0,
    min_consecutive_days: int = 3,
    freq: str = "YS",
) -> xr.DataArray:
    """
    Durée cumulée des vagues de chaleur (séquences de N jours consécutifs Tmax > seuil).

    Retourne le nombre de jours en vague de chaleur par période `freq`.
    """
    tmax_c = tmax - _K0
    hot = (tmax_c > threshold_c).astype(int)

    # Fenêtre glissante : au moins N jours consécutifs au-dessus du seuil
    # On utilise une convolution pour détecter les séquences
    from scipy.ndimage import uniform_filter1d

    hot_np = hot.values  # shape (time, ...)
    # Moyenne glissante sur N jours ≥ 1 si tous les jours du bloc sont chauds
    kernel = np.ones(min_consecutive_days) / min_consecutive_days
    # Appliquer sur l'axe temporel
    conv = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"),
        axis=0,
        arr=hot_np.astype(float),
    )
    in_wave = xr.DataArray(
        (conv >= 1.0 - 1e-6).astype(int),
        dims=hot.dims,
        coords=hot.coords,
    )
    return in_wave.resample(time=freq).sum().rename("heatwave_days")


# ---------------------------------------------------------------------------
# Précipitations
# ---------------------------------------------------------------------------

def _to_mm(tp: xr.DataArray, unit: str) -> xr.DataArray:
    """Convertit les précipitations en mm si elles sont en m."""
    if unit == "m":
        return tp * 1000.0
    return tp


def accumulated_precipitation(
    tp: xr.DataArray,
    unit: str = "m",
    freq: str = "YS",
) -> xr.DataArray:
    """Précipitations cumulées (mm) par période."""
    return _to_mm(tp, unit).resample(time=freq).sum().rename("accumulated_precip_mm")


def extreme_precip_days(
    tp_daily: xr.DataArray,
    threshold_mm: float = 20.0,
    unit: str = "m",
    freq: str = "YS",
) -> xr.DataArray:
    """
    Nombre de jours avec précipitations > seuil (inondation, coulées).

    Parameters
    ----------
    tp_daily:
        Précipitations journalières.
    threshold_mm:
        Seuil (mm/j). Défaut 20 mm.
    """
    tp_mm = _to_mm(tp_daily, unit)
    return (tp_mm > threshold_mm).resample(time=freq).sum().rename("extreme_precip_days")


def dry_spell_days(
    tp_daily: xr.DataArray,
    threshold_mm: float = 1.0,
    unit: str = "m",
    freq: str = "YS",
) -> xr.DataArray:
    """
    Nombre total de jours secs (précipitations < seuil) par période.

    Pour une sécheresse assurance, on complète avec dry_spell_max_length.
    """
    tp_mm = _to_mm(tp_daily, unit)
    return (tp_mm < threshold_mm).resample(time=freq).sum().rename("dry_spell_days")


def dry_spell_max_length(
    tp_daily: xr.DataArray,
    threshold_mm: float = 1.0,
    unit: str = "m",
    freq: str = "YS",
) -> xr.DataArray:
    """
    Durée maximale de la séquence sèche consécutive par période.

    Retourne un DataArray spatial avec la longueur max (jours) de chaque séquence sèche.
    """
    tp_mm = _to_mm(tp_daily, unit)
    dry = (tp_mm < threshold_mm).astype(int)

    groups = dry.resample(time=freq)
    chunks = []
    for label, block in groups:
        dry_np = block.values  # (time, ...)
        # Longueur max de séquence consécutive de 1 sur l'axe 0
        max_run = _max_consecutive_along_axis0(dry_np)
        da = xr.DataArray(
            max_run,
            dims=block.dims[1:],
            coords={k: v for k, v in block.coords.items() if k != "time"},
            attrs={"units": "days"},
        )
        da = da.expand_dims({"time": [label]})
        chunks.append(da)
    return xr.concat(chunks, dim="time").rename("dry_spell_max_length")


def _max_consecutive_along_axis0(arr: np.ndarray) -> np.ndarray:
    """Longueur maximale de 1 consécutifs le long de l'axe 0."""
    result = np.zeros(arr.shape[1:], dtype=np.int32)
    current = np.zeros(arr.shape[1:], dtype=np.int32)
    for t in range(arr.shape[0]):
        current = np.where(arr[t] == 1, current + 1, 0)
        result = np.maximum(result, current)
    return result


def r95p(
    tp_daily: xr.DataArray,
    unit: str = "m",
    freq: str = "YS",
) -> xr.DataArray:
    """
    Précipitations totales au-dessus du 95e percentile (indice ETCCDI).

    Le percentile est calculé sur l'ensemble de la période disponible (référence).
    """
    tp_mm = _to_mm(tp_daily, unit)
    p95 = tp_mm.quantile(0.95, dim="time")
    above_p95 = tp_mm.where(tp_mm > p95)
    return above_p95.resample(time=freq).sum().rename("r95p_mm")


# ---------------------------------------------------------------------------
# Vent
# ---------------------------------------------------------------------------

def wind_storm_hours(
    wind_speed: xr.DataArray,
    threshold_ms: float = 15.0,
    freq: str = "YS",
) -> xr.DataArray:
    """
    Nombre d'heures avec vitesse de vent > seuil (tempête).

    Parameters
    ----------
    wind_speed:
        Vitesse de vent (m s⁻¹). Peut être la vitesse instantanée ou les rafales.
    threshold_ms:
        Seuil tempête (m s⁻¹). Beaufort 7 ≈ 14 m/s, Beaufort 9 ≈ 21 m/s.
    """
    return (wind_speed > threshold_ms).resample(time=freq).sum().rename("wind_storm_hours")


def wind_speed_from_components(
    u: xr.DataArray,
    v: xr.DataArray,
) -> xr.DataArray:
    """Vitesse de vent scalaire depuis les composantes u, v."""
    return np.sqrt(u**2 + v**2).rename("wind_speed")


# ---------------------------------------------------------------------------
# Enneigement (proxy)
# ---------------------------------------------------------------------------

def snowfall_proxy_days(
    tp_daily: xr.DataArray,
    t2m_daily: xr.DataArray,
    temp_threshold_c: float = 2.0,
    precip_threshold_mm: float = 1.0,
    unit: str = "m",
    freq: str = "YS",
) -> xr.DataArray:
    """
    Proxy jours de neige : précipitations > seuil ET T2m < seuil température.

    Utile en l'absence de données de neige observées.
    """
    tp_mm = _to_mm(tp_daily, unit)
    t2m_c = t2m_daily - _K0
    snow_mask = (tp_mm > precip_threshold_mm) & (t2m_c < temp_threshold_c)
    return snow_mask.resample(time=freq).sum().rename("snowfall_proxy_days")


# ---------------------------------------------------------------------------
# Calcul en batch de tous les indices
# ---------------------------------------------------------------------------

def compute_all_indices(
    ds: xr.Dataset,
    unit_tp: str = "m",
    freq: str = "YS",
) -> xr.Dataset:
    """
    Calcule l'ensemble des indices à partir d'un Dataset standardisé.

    Variables attendues dans `ds`:
        t2m  : température 2 m horaire (K)
        tmin : Tmin journalière (K)  [calculée si absente]
        tmax : Tmax journalière (K)  [calculée si absente]
        tp   : précipitations horaires ou journalières (m ou mm selon unit_tp)
        u10, v10 : composantes du vent 10 m horaire (m s⁻¹)

    Returns
    -------
    xr.Dataset avec tous les indices.
    """
    indices = {}

    # Préparation des entrées journalières
    if "t2m" in ds:
        t2m = ds["t2m"]
        tmin = ds.get("tmin", t2m.resample(time="1D").min())
        tmax = ds.get("tmax", t2m.resample(time="1D").max())
        t2m_daily = t2m.resample(time="1D").mean()

        indices["frost_days"] = frost_days(tmin, freq=freq)
        indices["frost_hours"] = frost_hours(t2m, freq=freq)
        indices["heat_stress_days"] = heat_stress_days(tmax, freq=freq)
        indices["heatwave_days"] = heatwave_index(tmax, freq=freq)
        indices["gdd"] = growing_degree_days(tmax, tmin, freq=freq)

    if "tp" in ds:
        tp = ds["tp"]
        tp_daily = tp.resample(time="1D").sum()

        indices["accumulated_precip_mm"] = accumulated_precipitation(tp, unit=unit_tp, freq=freq)
        indices["extreme_precip_days"] = extreme_precip_days(tp_daily, unit=unit_tp, freq=freq)
        indices["dry_spell_days"] = dry_spell_days(tp_daily, unit=unit_tp, freq=freq)
        indices["r95p_mm"] = r95p(tp_daily, unit=unit_tp, freq=freq)

    if "u10" in ds and "v10" in ds:
        ws = wind_speed_from_components(ds["u10"], ds["v10"])
        indices["wind_storm_hours"] = wind_storm_hours(ws, freq=freq)

    if "tp" in ds and "t2m" in ds:
        indices["snowfall_proxy_days"] = snowfall_proxy_days(
            tp_daily, t2m_daily, unit=unit_tp, freq=freq
        )

    return xr.Dataset(indices)
