"""Skill de prévision AROME (Open-Meteo Historical Forecast API) vs Sencrop.

Vérification du gel au niveau parcelle : Tmin nocturne prévu (AROME
`meteofrance_arome_france`, ~J-1) vs observé (Sencrop), seuil -2,2 °C (LT10
abricot), stations Drôme, saisons gel dispo sur Open-Meteo (2024-2025).

Trois évaluations :
  1. AROME brut  — le forecast public, non corrigé.
  2. AROME + calibration Sencrop, cross-validation SPATIALE (leave-one-station-out
     quantile mapping) : correction apprise sur les autres stations, appliquée à
     une parcelle tenue à l'écart → teste le transfert vers parcelles sans capteur.
  3. AROME + calibration Sencrop, split TEMPOREL (train saison A → test saison B) :
     calibrer sur le passé, prévoir la saison suivante → cadre opérationnel.

Usage : uv run --with pandas --with numpy python .../arome_forecast_skill.py \
          [--sencrop <dir>] [--cache <pkl>] [--no-fetch]
Nécessite un accès réseau à open-meteo.com (sauf --no-fetch avec cache présent).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd

THR = -2.2
BBOX = dict(la0=44.0, la1=45.5, lo0=4.0, lo1=5.5)
API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
YEARS = [2024, 2025]  # profondeur d'archive AROME sur Open-Meteo


def _night_date(ts: pd.Timestamp):
    """Nuit rattachée au matin : heures <=8 -> ce jour ; >=18 -> lendemain ; jour -> NaT."""
    h = ts.hour
    if h <= 8:
        return ts.normalize()
    if h >= 18:
        return (ts + pd.Timedelta(days=1)).normalize()
    return pd.NaT


def _nightly_tmin(times, temps) -> pd.Series:
    s = pd.Series(temps, index=pd.to_datetime(times)).dropna()
    nd = s.index.map(_night_date)
    g = pd.Series(s.values, index=nd).dropna()
    return g.groupby(level=0).min()


def _fetch(chunk, start, end):
    lats = ",".join(f"{s.latitude:.4f}" for s in chunk)
    lons = ",".join(f"{s.longitude:.4f}" for s in chunk)
    q = urllib.parse.urlencode(
        dict(
            latitude=lats,
            longitude=lons,
            start_date=start,
            end_date=end,
            hourly="temperature_2m",
            models="meteofrance_arome_france",
            timezone="UTC",
        )
    )
    r = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=60))
    return r if isinstance(r, list) else [r]


def load_arome(stations, cache, no_fetch):
    if cache and os.path.exists(cache):
        d = pickle.load(open(cache, "rb"))["arome"]  # noqa: SIM115
        return {k: {pd.Timestamp(kk): vv for kk, vv in v.items()} for k, v in d.items()}
    if no_fetch:
        raise SystemExit("--no-fetch mais pas de cache")
    arome = {}
    for year in YEARS:
        for i in range(0, len(stations), 10):
            chunk = stations[i : i + 10]
            try:
                res = _fetch(chunk, f"{year}-02-01", f"{year}-04-30")
            except Exception as e:
                print(f"  fetch {year} chunk {i} FAIL {str(e)[:60]}")
                continue
            for s, r in zip(chunk, res):
                h = r.get("hourly", {})
                tm = _nightly_tmin(h.get("time", []), h.get("temperature_2m", []))
                arome.setdefault(s.bucket_id, {}).update(tm.to_dict())
            time.sleep(0.5)
    return arome


def load_sencrop(sencrop_dir):
    sen = {}
    for year in YEARS:
        p = f"{sencrop_dir}/{year}.csv"
        if not (p.endswith(".csv") and os.path.isfile(p)):
            parts = glob.glob(f"{p}/part-*.csv")
            p = parts[0] if parts else p
        d = pd.read_csv(p)
        if "temperature_source" in d:
            d = d[d.temperature_source == "station"]
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dt.tz_localize(
            None
        )
        d = d.dropna(subset=["timestamp", "temperature"])
        for sid, grp in d.groupby("station_id"):
            tm = _nightly_tmin(grp["timestamp"].values, grp["temperature"].values)
            sen.setdefault(sid, {}).update(tm.to_dict())
    return sen


def scores(m, o, thr=THR):
    m, o = np.asarray(m), np.asarray(o)
    mf, of = m < thr, o < thr
    TP = int((mf & of).sum())
    FP = int((mf & ~of).sum())
    FN = int((~mf & of).sum())
    TN = int((~mf & ~of).sum())
    N = len(m)
    r = lambda a, b: a / b if b else float("nan")  # noqa: E731
    hr = (TP + FP) * (TP + FN) / N if N else 0
    return dict(
        n=N,
        evts=TP + FN,
        POD=r(TP, TP + FN),
        FAR=r(FP, TP + FP),
        CSI=r(TP, TP + FP + FN),
        ETS=r(TP - hr, TP + FP + FN - hr),
        RMSE=float(np.sqrt(np.mean((m - o) ** 2))),
        bias=float(np.mean(m - o)),
    )


def _qmap(train_a, train_o, xs):
    tas = np.sort(np.asarray(train_a))
    to = np.asarray(train_o)
    out = []
    for a in xs:
        q = np.searchsorted(tas, a) / len(tas)
        out.append(float(np.quantile(to, min(max(q, 1e-4), 1 - 1e-4))))
    return np.array(out)


def _fmt(tag, c):
    return (
        f"  {tag:34} n={c['n']:5d} evts={c['evts']:3d}  POD={c['POD']:.2f} FAR={c['FAR']:.2f} "
        f"CSI={c['CSI']:.2f} ETS={c['ETS']:.2f} RMSE={c['RMSE']:.2f} biais={c['bias']:+.2f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sencrop", default="/Users/loicmaurin/kDrive/karpos_datasets/data/raw/sencrop"
    )
    ap.add_argument(
        "--catalog",
        default=None,
        help="stations_integrated.csv (défaut: <sencrop>/stations_integrated.csv)",
    )
    ap.add_argument("--cache", default=None, help="pkl cache AROME/Sencrop (skip fetch si présent)")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    cat = a.catalog or f"{a.sencrop}/stations_integrated.csv"

    df = pd.read_csv(cat)
    df = (
        df[
            (df.latitude >= BBOX["la0"])
            & (df.latitude <= BBOX["la1"])
            & (df.longitude >= BBOX["lo0"])
            & (df.longitude <= BBOX["lo1"])
        ]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates("bucket_id")
    )
    stations = list(df[["bucket_id", "latitude", "longitude"]].itertuples(index=False))
    print(f"{len(stations)} stations Drôme")

    arome = load_arome(stations, a.cache, a.no_fetch)
    sen = load_sencrop(a.sencrop)
    print(f"AROME {len(arome)} stations · Sencrop {len(sen)} stations")

    # paires (sid, year, arome, sencrop)
    pairs = defaultdict(list)  # sid -> [(year, a, o)]
    M, O = [], []  # noqa: E741
    for sid, an in arome.items():
        sn = sen.get(sid, {})
        for nd, at in an.items():
            ot = sn.get(pd.Timestamp(nd))
            if ot is not None and np.isfinite(at) and np.isfinite(ot):
                pairs[sid].append((pd.Timestamp(nd).year, float(at), float(ot)))
                M.append(float(at))
                O.append(float(ot))

    print("\n=== Skill prévision AROME vs Sencrop (Tmin nocturne, seuil -2,2°C, 2024+2025) ===")
    print(_fmt("1. AROME brut", scores(M, O)))

    # 2. LOSO quantile mapping (CV spatiale)
    sids = [s for s in pairs if len(pairs[s]) >= 3]
    Mc, Oc = [], []
    for held in sids:
        ta = [x[1] for s in sids if s != held for x in pairs[s]]
        to = [x[2] for s in sids if s != held for x in pairs[s]]
        xs = [x[1] for x in pairs[held]]
        Mc += list(_qmap(ta, to, xs))
        Oc += [x[2] for x in pairs[held]]
    print(_fmt("2. + Sencrop, LOSO (spatial)", scores(Mc, Oc)))

    # 3. Split temporel : train saison A -> test saison B
    for train_y, test_y in [(2024, 2025), (2025, 2024)]:
        ta = [x[1] for s in pairs for x in pairs[s] if x[0] == train_y]
        to = [x[2] for s in pairs for x in pairs[s] if x[0] == train_y]
        xs = [x[1] for s in pairs for x in pairs[s] if x[0] == test_y]
        ys = [x[2] for s in pairs for x in pairs[s] if x[0] == test_y]
        if not xs or not ta:
            continue
        mc = _qmap(ta, to, xs)
        print(_fmt(f"3. + Sencrop, train {train_y}->test {test_y}", scores(mc, ys)))

    # cache + export paires
    outdir = os.path.dirname(a.cache) if a.cache else "."
    cache_path = a.cache or os.path.join(outdir, "arome_sen_cache.pkl")
    pickle.dump(
        {
            "arome": {k: {str(kk): vv for kk, vv in v.items()} for k, v in arome.items()},
            "sen": {k: {str(kk): vv for kk, vv in v.items()} for k, v in sen.items()},
        },
        open(cache_path, "wb"),  # noqa: SIM115
    )  # noqa: SIM115
    rows = [(sid, y, av, ov) for sid in pairs for (y, av, ov) in pairs[sid]]
    pd.DataFrame(rows, columns=["station_id", "year", "arome_tmin", "sencrop_tmin"]).to_csv(
        os.path.join(outdir, "arome_sencrop_pairs.csv"), index=False
    )
    print(f"\ncache -> {cache_path} · paires -> {outdir}/arome_sencrop_pairs.csv")


if __name__ == "__main__":
    main()
