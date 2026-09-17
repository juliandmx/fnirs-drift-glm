"""Hemisphaeren-Kontrolle auf dem Khan-Datensatz (rechte Hand -> linke Hemisphaere).

Die SNIRF-Dateien haben keine Landmarken; die Zuordnung der 48 Kanaele zu den Hemisphaeren
kommt aus Khan et al. (2026), Tabelle 3 (`realdata.khan_hemisphere`). Je Konfiguration
(Familie, Konstellation, Schaetzer) werden aus der Gruppenkarte berechnet:

  lat_index     mittleres beta links minus rechts (positiv = kontralaterale Erwartung
                erfuellt, weil rechts getappt wird), je HbO/HbR
  n_sig_left/right   signifikante Kanaele je Hemisphaere (FDR ueber 96 Tests)
  max_hemisphere     Hemisphaere des |beta|-Maximums, dazu die 10-10-Positionen des Kanals
  tmax_hemisphere    Hemisphaere des |t|-Maximums

Die AR-IRLS-Karte (dct:0.02 / baseline) stammt aus results/realglm_group_map.nc (Cache
des Reports, 4. August 2026); OLS-Karten werden neu gerechnet (schnell). Ausgabe:
results/khan_lateralisation.csv, results/khan_group_maps.nc, jeweils mit Metadaten.

Aufruf:  conda run -n cedalion python -m drift_glm.analysis.khan_lateralisation
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from drift_glm.analysis import realglm as rg
from drift_glm.core.provenance import build_run_metadata, write_csv_atomic, write_metadata_atomic
from drift_glm.data import realdata as rd
from drift_glm import paths

RESULTS = paths.RESULTS
CACHED = ("dct:0.02", "baseline", "ar_irls")          # realglm_group_map.nc
OLS_CONFIGS = [("none", "baseline", "ols"), ("poly:3", "baseline", "ols"),
               ("dct:0.02", "baseline", "ols"), ("dct:0.02", "global", "ols"),
               ("butter:0.01", "baseline", "ols")]


def metrics(ds: xr.Dataset, family, constellation, noise_model) -> list[dict]:
    """Kennzahlen einer Gruppenkarte (beta, t, significant) je Chromophor."""
    chans = [str(c) for c in ds.channel.values]
    side = rd.khan_hemisphere(chans)                  # +1 links, -1 rechts
    table = rd.khan_channel_table(chans).set_index("channel")
    rows = []
    for ch in ("HbO", "HbR"):
        b = np.asarray(ds["beta"].sel(chromo=ch).values, float)
        t = np.asarray(ds["t"].sel(chromo=ch).values, float)
        sig = np.asarray(ds["significant"].sel(chromo=ch).values).astype(bool)
        j, k = int(np.nanargmax(np.abs(b))), int(np.nanargmax(np.abs(t)))
        rows.append(dict(
            family=family, constellation=constellation, noise_model=noise_model, chromo=ch,
            n_subjects=int(ds.attrs.get("n_subjects", 25)),
            lat_index=float(np.nanmean(b[side > 0]) - np.nanmean(b[side < 0])),
            mean_beta_left=float(np.nanmean(b[side > 0])),
            mean_beta_right=float(np.nanmean(b[side < 0])),
            n_sig_left=int(sig[side > 0].sum()), n_sig_right=int(sig[side < 0].sum()),
            max_channel=chans[j], max_hemisphere="left" if side[j] > 0 else "right",
            max_position=f"{table.loc[chans[j], 'source_1010']}-"
                         f"{table.loc[chans[j], 'detector_1010']}",
            max_beta=float(b[j]),
            tmax_channel=chans[k], tmax_hemisphere="left" if side[k] > 0 else "right",
            tmax_position=f"{table.loc[chans[k], 'source_1010']}-"
                          f"{table.loc[chans[k], 'detector_1010']}",
            tmax=float(t[k])))
    return rows


def group_map(prepped, files, family, constellation, noise_model) -> xr.Dataset:
    worker = rg.first_level
    betas = Parallel(n_jobs=min(rg.N_JOBS, len(files)), backend="loky")(
        delayed(worker)(prepped[f][0], prepped[f][1], family, constellation, noise_model,
                        30, 1) for f in files)
    by_sub = {}
    for f, b in zip(files, betas):
        by_sub.setdefault(rd.subject_of(f), []).append(b)
    by_sub = {s: xr.concat(v, dim="run").mean("run") for s, v in by_sub.items()}
    t, p_adj, rej, mean = rg.group_test(by_sub)
    ds = xr.Dataset(dict(beta=mean, t=t, p_fdr=p_adj, significant=rej.astype("int8")))
    ds.attrs.update(family=family, constellation=constellation, noise_model=noise_model,
                    n_subjects=len(by_sub))
    return ds


def main(configs=OLS_CONFIGS):
    paths.ensure()
    t0 = time.time()
    rows, maps = [], []
    cache = RESULTS / "realglm_group_map.nc"
    if cache.exists():
        ds = xr.load_dataset(cache)
        rows += metrics(ds, *CACHED)
        maps.append(ds.expand_dims(config=[" / ".join(CACHED)]))
        print(f"Cache: {cache.name} ({' / '.join(CACHED)})", flush=True)

    files = rd.find_files()
    prepped = {}
    for f in files:
        rec = rd.load(f)
        P, _ = rd.preprocess_recording(rec, motion_method=rg.MOTION_METHOD)
        prepped[f] = (P.conc, rd.stim_df(rec, rd.subject_of(f), "tapping"))
    print(f"Vorverarbeitung: {len(files)} Dateien in {time.time() - t0:.0f}s", flush=True)
    for fam, con, nm in configs:
        tc = time.time()
        ds = group_map(prepped, files, fam, con, nm)
        rows += metrics(ds, fam, con, nm)
        maps.append(ds.expand_dims(config=[f"{fam} / {con} / {nm}"]))
        r = rows[-2]
        print(f"  {fam:12s} {con:8s} {nm:7s}  HbO lat={r['lat_index']:+.4f} µM  "
              f"sig L/R={r['n_sig_left']}/{r['n_sig_right']}  |beta|max {r['max_hemisphere']} "
              f"({r['max_position']})  ({time.time() - tc:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / "khan_lateralisation.csv"
    meta = build_run_metadata(dict(analysis="khan_lateralisation", cached=list(CACHED),
                                   configs=[list(c) for c in configs],
                                   motion_method=rg.MOTION_METHOD), files)
    meta["elapsed_seconds"] = round(time.time() - t0, 1)
    write_csv_atomic(df, out)
    write_metadata_atomic(meta, out.with_suffix(".meta.json"))
    allmaps = xr.concat(maps, dim="config")
    for k in list(allmaps.attrs):
        del allmaps.attrs[k]
    allmaps.to_netcdf(RESULTS / "khan_group_maps.nc")
    print(f"\n[OK] -> {out}")
    print(df[["family", "constellation", "noise_model", "chromo", "lat_index",
              "n_sig_left", "n_sig_right", "max_hemisphere", "max_position",
              "tmax_hemisphere"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
