"""Vergleichs-Sweep der Driftregressoren auf den simulierten nn22-Daten.

Achsen: Driftfamilie (poly, DCT, Legendre, B-Spline, none, Butterworth-Hochpass als
Vorverarbeitungs-Alternative ohne Driftregressoren), Analysefenster, Konstellation
(baseline/motion/global/short_avg/short_maxcorr), Motion Correction, Rauschmodell, Seed.
In den Filter-Armen gibt es nur den Offset, in allen anderen Armen wird nicht gefiltert.
Metriken je HbO/HbR: Bias, Varianz, RMSE von beta_hat gegen die Ground Truth (ueber
Seeds), HbO/HbR-Plausibilitaet, Modellfit (R^2, adj. R^2, Residual-RMS, resid_err_corr).
Schreibt results/sweep_summary.csv, sweep_per_channel.nc und sweep_meta.json.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.sweep pilot   # kleiner Test + Timing
    conda run -n cedalion python -m drift_glm.analysis.sweep v4       # Hauptstudie
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import cedalion
import cedalion.data
import cedalion.models.glm as glm
from cedalion import units

from drift_glm.core import fitstats as fs
from drift_glm.core import pipeline as pl
from drift_glm.core import preprocess as prep
from drift_glm.core import shortchannel as sc
from drift_glm.core.filtering import apply_filter
from drift_glm import paths

RESULTS = paths.RESULTS
HRF_REG = "HRF Stim"


PILOT = dict(   # klein; deckt alle Familien-Typen und Konstellationen ab
    families=["poly:3", "dct:0.01", "legendre:3", "bspline:5", "none",
              "butter:0.01", "lowpass:0.5", "bandpass:0.01-0.5"],
    windows=[90.0],
    constellations=["baseline", "motion", "global", "short_avg"],
    seeds=[0],
    motion_methods=["wavelet", "tddr+wavelet"],
    n_channels=20,
    noise_models=["ar_irls", "ols"],
    ar_order=30,
)

BASE_FAMILIES = ["poly:1", "poly:2", "poly:3", "poly:4", "poly:5",
                 "dct:0.005", "dct:0.01", "dct:0.02",
                 "legendre:1", "legendre:3", "legendre:5",
                 "none", "butter:0.01"]

# Hauptstudie (~8-9 h). Motion Correction ist eine eigene Achse, weil TDDR das Driftband
# und die eingemischte HRF daempft, Wavelet nicht; gekreuzt mit der
# Konstellation "motion" zeigt sich, ob sich Korrektur und Motion-Regressoren doppeln.
# 15 Familien x 3 Fenster x 5 Konstellationen x 2 Motion x 4 Seeds = 1800 Fits.
V4 = dict(
    families=BASE_FAMILIES + ["bspline:5", "bspline:8"],
    windows=[90.0, 180.0, 368.0],
    constellations=["baseline", "motion", "global", "short_avg", "short_maxcorr"],
    seeds=list(range(4)),
    motion_methods=["wavelet", "tddr+wavelet"],
    n_channels=20,
    noise_models=["ar_irls"],
    ar_order=30,
)

# Additional methodological control only: 1 x 3 x 5 x 2 x 4 = 120 fits.
BUTTERXY = dict(V4, families=["butterxy:0.01"])


def drift_dm(family: str, conc):
    """(DesignMatrix inkl. Offset, filter|None) fuer eine Driftfamilie.

    `filter` ist None oder ein Paar `(fmin, fmax)` in Hz fuer `freq_filter` (Cedalion:
    `fmax=0` -> Hochpass, `fmin=0` -> Tiefpass, beide gesetzt -> Bandpass). Die
    Filter-Familien sind Vorverarbeitungs-Alternativen und bekommen nur den Offset;
    gefiltert wird im Konzentrationsraum.
    """
    fam, _, param = family.partition(":")
    dmx = glm.design_matrix
    offset_only = lambda: dmx.drift_regressors(conc, drift_order=0)   # noqa: E731
    if fam == "poly":
        return dmx.drift_regressors(conc, drift_order=int(param)), None
    if fam == "legendre":
        return dmx.drift_legendre_regressors(conc, order=int(param)), None
    if fam == "dct":
        dm = dmx.drift_cosine_regressors(conc, fmax=float(param) * units.Hz)
        # Native k=0 is already a constant. For floor(2*N*f/fs)=0 Cedalion
        # returns no columns, so retain an explicit offset in that case only.
        return (dm if dm.common.sizes["regressor"] else offset_only()), None
    if fam == "bspline":
        return _bspline_dm(conc, int(param)), None    # enthaelt Offset (Zerlegung d. Eins)
    if fam == "none":
        return offset_only(), None                                    # nur Offset
    if fam in ("butter", "butterxy"):   # old arm / consistent-filter control
        return offset_only(), (float(param), 0.0)
    if fam == "lowpass":             # Tiefpass allein, z.B. lowpass:0.5
        return offset_only(), (0.0, float(param))
    if fam == "bandpass":            # z.B. bandpass:0.01-0.5
        lo, _, hi = param.partition("-")
        return offset_only(), (float(lo), float(hi))
    raise ValueError(f"Unbekannte Driftfamilie: {family}")


def _bspline_dm(conc, n_splines, degree=3):
    """B-Spline-Drift-Regressoren als eigene Spalten (nicht nativ in Cedalion).

    Geklemmte kubische B-Spline-Basis ueber das Analysefenster; sie bildet eine Zerlegung
    der Eins und enthaelt damit den Offset. Regressornamen 'Drift BS i'.
    """
    from scipy.interpolate import BSpline
    nt = conc.sizes["time"]
    n_int = max(n_splines - degree - 1, 0)
    x = np.linspace(0.0, 1.0, nt)
    interior = list(np.linspace(0.0, 1.0, n_int + 2)[1:-1]) if n_int > 0 else []
    knots = np.concatenate([[0.0] * (degree + 1), interior, [1.0] * (degree + 1)])
    cols = [BSpline(knots, (np.arange(n_splines) == i).astype(float), degree)(x)
            for i in range(n_splines)]
    da = xr.DataArray(
        np.stack(cols, axis=1), dims=("time", "regressor"),
        coords={"time": conc.time, "regressor": [f"Drift BS {i}" for i in range(n_splines)]},
    ).expand_dims({"chromo": list(conc.chromo.values)}).transpose(
        "time", "regressor", "chromo")
    return glm.design_matrix.DesignMatrix(common=da)


def motion_dm(aux, conc):
    """Motion-Regressoren aus Accelerometer+Gyroscope (auf conc.time, z-normiert)."""
    cols, names = [], []
    for src in ("accelerometer", "gyroscope"):
        if src not in aux:
            continue
        a = aux[src]
        try:
            a = a.pint.dequantify()
        except Exception:
            pass
        a = a.interp(time=conc.time)
        for k in range(a.sizes["aux_channel"]):
            col = a.isel(aux_channel=k)
            col = (col - col.mean("time")) / (float(col.std("time")) + 1e-12)
            cols.append(np.asarray(col.values, dtype=float))
            names.append(f"Motion {src[:3]}{k}")
    if not cols:
        return None
    arr = np.nan_to_num(np.stack(cols, axis=1))  # (time, regressor)
    da = xr.DataArray(
        arr, dims=("time", "regressor"),
        coords={"time": conc.time, "regressor": names},
    ).expand_dims({"chromo": list(conc.chromo.values)}).transpose(
        "time", "regressor", "chromo")
    return glm.design_matrix.DesignMatrix(common=da)


def constellation_dm(name, parts: dict):
    """Zusatz-DesignMatrix fuer eine Konstellation (None fuer baseline).

    `name` ist ein mit "+" verbundener Ausdruck aus den Schluesseln von `parts`, z.B.
    "motion", "global", "short_avg", "short_maxcorr". "baseline" bedeutet: nichts dazu.
    Fehlende Bausteine (z.B. keine Short-Channels im Datensatz) werden uebersprungen.
    """
    dm = None
    for key in str(name).split("+"):
        if key == "baseline":
            continue
        part = parts.get(key)
        if part is None:
            continue
        dm = part if dm is None else dm & part
    return dm


def _write_progress(done, total, t0, timings, last):
    """Fortschrittsdatei in results/logs, nach jedem Fit aktualisiert."""
    el = time.time() - t0
    med = float(np.median(timings)) if timings else 0.0
    eta = med * (total - done)
    (paths.LOGS / "sweep_progress.txt").write_text(
        f"Fortschritt: {done}/{total} ({100 * done / total:.0f} %)\n"
        f"verstrichen : {el / 60:5.1f} min\n"
        f"ETA (Rest)  : {eta / 60:5.1f} min\n"
        f"median/Fit  : {med:4.1f} s\n"
        f"letzte Zelle: {last}\n"
    )


def run(cfg: dict, output_prefix="sweep", replace: bool = False):
    """Sweep rechnen. Vorhandene Ergebnisdateien werden nur mit `replace=True` ersetzt,
    und dann erst nach einer verifizierten Archivkopie (results/archive/)."""
    paths.ensure()
    output_files = _result_files(output_prefix)
    if any(p.exists() for p in output_files):
        if not replace:
            raise FileExistsError(f"Refusing to replace existing results: {output_files}. "
                                  "Archive them (--replace) or choose a separate output prefix.")
        print("Archiviere vorhandene Ergebnisse:",
              _archive_results(output_prefix, "before_rerun"), flush=True)
    t0 = time.time()
    rec = cedalion.data.get_nn22_resting_state()   # einmal laden
    # Erste Haelfte der Preprocessing-Kette (Rohamplitude -> OD) haengt weder vom
    # Fenster noch vom Seed ab -> einmal berechnen und durchreichen.
    stage = prep.to_od_stage(rec)
    raw = defaultdict(dict)     # (fam, win, con) -> {seed: bhat (channel, chromo)}
    beta_true = None            # Peak-Referenz (Blob-Max) je chromo
    beta_true_map = None        # per-Kanal Ground-Truth (channel, chromo) auf Fit-Subset
    timings = []
    n_cells = (len(cfg["families"]) * len(cfg["windows"])
               * len(cfg["constellations"]) * len(cfg["seeds"])
               * len(cfg["motion_methods"]) * len(cfg["noise_models"]))
    done = 0
    print(f"Sweep: {n_cells} Zellen | Schaetzer={'/'.join(cfg['noise_models'])} | "
          f"n_channels={cfg['n_channels']} | "
          f"Motion={'/'.join(cfg['motion_methods'])}", flush=True)

    ref_channels = None     # feste Fit-Kanaele, auf der ersten Zelle bestimmt
    for mc in cfg["motion_methods"]:
      for win in cfg["windows"]:
        for seed in cfg["seeds"]:
            P = pl.build(window_s=win, stage=stage, motion_method=mc, seed=seed)
            beta_true = P.beta_true
            # Ausgewertet werden nur die langen Kanaele; die kurzen dienen als Regressor.
            ts_long, ts_short = sc.split(P.conc_syn, P.geo3d)
            bt_long = P.beta_true_map.sel(channel=ts_long.channel)
            if ref_channels is None:
                # Fit-Subset: die N staerkst-aktivierten langen Kanaele (Blob-Kern), einmal
                # bestimmt und per Label festgehalten, weil die Kanalmasken nach der Motion
                # Correction berechnet werden und je Stufe leicht abweichen koennen.
                w_hbo = np.abs(bt_long.sel(chromo="HbO").values)
                order = np.argsort(-w_hbo)[:cfg["n_channels"]]
                ref_channels = [str(c) for c in bt_long.channel.values[np.sort(order)]]
            have = set(str(c) for c in ts_long.channel.values)
            use = [c for c in ref_channels if c in have]
            if len(use) < len(ref_channels):
                print(f"  ! {len(ref_channels) - len(use)} Referenzkanaele fehlen bei "
                      f"{mc}/win={win:g}/seed={seed} -- Schnittmenge bei der "
                      f"Aggregation", flush=True)
            ts_base = ts_long.sel(channel=use)
            if beta_true_map is None:   # Blob ist fix -> einmal auf Fit-Subset erfassen
                beta_true_map = bt_long.sel(channel=use)
            parts = {
                "motion": motion_dm(P.aux, P.conc),
                "global": glm.design_matrix.global_mean_regressor(P.conc_syn),
            }
            # Short-Channel-Regressoren: nur bauen, was die Konfiguration braucht.
            for v in sc.VARIANTS:
                if any(v in str(c) for c in cfg["constellations"]):
                    parts[v] = sc.short_dm(v, ts_base, ts_short, P.geo3d)
            for family in cfg["families"]:
                dm_drift, filt = drift_dm(family, P.conc)
                for con in cfg["constellations"]:
                    extra = constellation_dm(con, parts)
                    dm = P.dm_hrf & dm_drift
                    if extra is not None:
                        dm = dm & extra
                    ts_fam, dm = apply_filter(ts_base, dm, filt,
                                             filter_design=family.startswith("butterxy:"))
                    for nm in cfg["noise_models"]:
                        tc = time.time()
                        betas = glm.fit(ts_fam, dm, noise_model=nm,
                                        ar_order=cfg["ar_order"],
                                        max_jobs=-1).sm.params
                        # Modellfit am selben Fit: R^2 und Residual-RMS je Kanal (ein predict).
                        fitq = fs.fit_metrics(ts_fam, betas, dm)
                        dt = time.time() - tc
                        raw[(family, win, con, mc, nm)][seed] = dict(
                            bhat=betas.sel(regressor=HRF_REG),
                            r2=fitq.r2, r2_adj=fitq.r2_adj,
                            resid_rms=fitq.resid_rms)
                        timings.append(dt)
                        done += 1
                        print(f"[{done}/{n_cells}] {mc:13s} {nm:7s} win={win:g} "
                              f"seed={seed} {family:15s} {con:14s} {dt:5.1f}s",
                              flush=True)
                        _write_progress(done, n_cells, t0, timings,
                                        f"{family} {con} ({mc}/{nm}, win={win:g}, "
                                        f"seed={seed})")

    _aggregate_and_export(cfg, raw, beta_true, beta_true_map, timings, time.time() - t0,
                          output_prefix=output_prefix)


def _aggregate_and_export(cfg, raw, beta_true, beta_true_map, timings, wall,
                          output_prefix="sweep"):
    fams, wins = cfg["families"], cfg["windows"]
    cons, seeds = cfg["constellations"], cfg["seeds"]
    mcs, nms = cfg["motion_methods"], cfg["noise_models"]
    sample = next(iter(raw.values()))[seeds[0]]["bhat"]
    chrom = [str(c) for c in sample.chromo.values]

    # Kanal-Schnittmenge ueber alle Zellen: die Masken haengen von der Motion Correction
    # ab, verglichen wird nur, was ueberall vorhanden ist.
    common = None
    for by_seed in raw.values():
        for cell in by_seed.values():
            s = {str(c) for c in cell["bhat"].channel.values}
            common = s if common is None else (common & s)
    chans = [str(c) for c in sample.channel.values if str(c) in common]
    if len(chans) < sample.sizes["channel"]:
        print(f"     Hinweis: {sample.sizes['channel'] - len(chans)} Kanaele nicht in "
              f"allen Zellen vorhanden -> auf {len(chans)} gemeinsame beschraenkt")

    # 8D-Rohwerte je Groesse: (family, window, constellation, motion, noise_model,
    #                          seed, channel, chromo)
    dims8 = ("family", "window_s", "constellation", "motion", "noise_model",
             "seed", "channel", "chromo")
    coords8 = dict(family=fams, window_s=wins, constellation=cons, motion=mcs,
                   noise_model=nms, seed=seeds, channel=chans, chromo=chrom)
    shape8 = tuple(len(coords8[d]) for d in dims8)
    fields = ("bhat", "r2", "r2_adj", "resid_rms")
    arrs = {k: np.full(shape8, np.nan) for k in fields}
    for (f, w, c, m, n), by_seed in raw.items():
        idx = (fams.index(f), wins.index(w), cons.index(c), mcs.index(m), nms.index(n))
        for si, s in enumerate(seeds):
            for k in fields:
                arrs[k][idx + (si,)] = (by_seed[s][k].sel(channel=chans)
                                        .transpose("channel", "chromo").values)
    bhat, r2x, r2ax, residx = (
        xr.DataArray(arrs[k], dims=dims8, coords=coords8) for k in fields)

    # per-Kanal Ground-Truth (raeumlicher Blob), auf bhat-Koordinaten ausgerichtet
    bt = xr.DataArray(
        beta_true_map.sel(channel=chans).transpose("channel", "chromo").values,
        dims=("channel", "chromo"), coords={"channel": chans, "chromo": chrom})
    mean_seed = bhat.mean("seed")
    bias = mean_seed - bt
    var = bhat.var("seed")
    rmse = np.sqrt(((bhat - bt) ** 2).mean("seed"))

    ds = xr.Dataset(dict(bhat=bhat, bias=bias, var=var, rmse=rmse, beta_true_map=bt,
                         r2=r2x, r2_adj=r2ax, resid_rms=residx))
    ds.attrs["beta_true_peak_hbo"] = beta_true["HbO"]
    ds.attrs["beta_true_peak_hbr"] = beta_true["HbR"]
    ds.attrs["dct_constant_schema"] = "one_constant_v2"
    paths.ensure()

    # Tidy-Zusammenfassung (ueber Kanaele aggregiert) + Plausibilitaet
    recs = []
    for f in fams:
        for w in wins:
            for c in cons:
                for m in mcs:
                  for n in nms:
                    sel = dict(family=f, window_s=w, constellation=c,
                               motion=m, noise_model=n)
                    mo = mean_seed.sel(**sel, chromo="HbO")
                    mr = mean_seed.sel(**sel, chromo="HbR")
                    corr = float(np.corrcoef(mo.values, mr.values)[0, 1])
                    peak = abs(beta_true["HbO"])
                    m_act = np.abs(mo.values) > 0.1 * peak   # nur aktive Kanaele
                    ratio = (float(np.nanmedian(mr.values[m_act] / mo.values[m_act]))
                             if m_act.any() else float("nan"))
                    for ch in chrom:
                        b = bias.sel(**sel, chromo=ch)
                        v = var.sel(**sel, chromo=ch)
                        r = rmse.sel(**sel, chromo=ch)
                        # resid_err_corr: Residual-RMS gegen |beta_hat - GT|, gepoolt
                        # ueber (seed, Kanal) innerhalb der Zelle.
                        rr = residx.sel(**sel, chromo=ch)
                        ae = np.abs(bhat.sel(**sel, chromo=ch) - bt.sel(chromo=ch))
                        xv, yv = rr.values.ravel(), ae.values.ravel()
                        ok = np.isfinite(xv) & np.isfinite(yv)
                        re_corr = (float(np.corrcoef(xv[ok], yv[ok])[0, 1])
                                   if ok.sum() > 3 and xv[ok].std() > 0
                                   and yv[ok].std() > 0 else float("nan"))
                        recs.append(dict(
                            family=f, window_s=w, constellation=c, motion=m,
                            noise_model=n, chromo=ch,
                            n_seeds=len(seeds), n_channels=len(chans),
                            beta_true_peak=beta_true[ch],
                            bias_med=float(b.median()),
                            absbias_med=float(np.abs(b).median()),
                            var_med=float(v.median()),
                            rmse_med=float(r.median()),
                            rmse_mean=float(r.mean()),
                            r2_med=float(r2x.sel(**sel, chromo=ch).median()),
                            r2_adj_med=float(r2ax.sel(**sel, chromo=ch).median()),
                            resid_rms_med=float(rr.median()),
                            resid_err_corr=re_corr,
                            hbo_hbr_corr=corr, hbr_hbo_ratio_med=ratio,
                        ))
    df = pd.DataFrame(recs).sort_values(["chromo", "window_s", "rmse_med"])

    meta = dict(config=cfg, n_cells=len(timings),
                wall_s=round(wall, 1),
                cell_time_s=dict(min=round(float(np.min(timings)), 2),
                                 median=round(float(np.median(timings)), 2),
                                 max=round(float(np.max(timings)), 2)))
    _write_results(output_prefix, ds, df, meta)

    print(f"\n[OK] {len(timings)} Fits in {wall/60:.1f} min "
          f"(median {np.median(timings):.1f}s/Fit).")
    print(f"     -> {RESULTS / (output_prefix + '_summary.csv')}")
    print(f"     -> {RESULTS / (output_prefix + '_per_channel.nc')}")
    print("\nBeste Familie je (Fenster, chromo) nach RMSE_med:")
    best = df.loc[df.groupby(["chromo", "window_s"])["rmse_med"].idxmin()]
    print(best[["chromo", "window_s", "family", "constellation", "motion",
                "rmse_med", "absbias_med", "var_med"]].to_string(index=False))
    # Bias getrennt nach Chromophor: TDDR daempft die HRF, was am RMSE allein nicht
    # sichtbar ist.
    print("\nMotion-Achse (Mittel ueber Familien/Fenster/Konstellationen):")
    print(df.groupby(["chromo", "motion"])[["bias_med", "rmse_med"]].mean()
            .to_string(float_format=lambda x: f"{x:+.4f}"))


def _result_files(prefix):
    if Path(prefix).name != prefix:
        raise ValueError("output prefix must be a filename, not a path")
    return [RESULTS / f"{prefix}_{suffix}" for suffix in
            ("per_channel.nc", "summary.csv", "meta.json")]


def _write_results(prefix, ds, frame, meta):
    """Serialize every output first, then atomically replace individual files."""
    destinations = _result_files(prefix)
    temporary = [p.with_name(f".{p.name}.{os.getpid()}.tmp") for p in destinations]
    try:
        ds.to_netcdf(temporary[0])
        frame.to_csv(temporary[1], index=False)
        temporary[2].write_text(json.dumps(meta, indent=2) + "\n")
        for src, dst in zip(temporary, destinations):
            os.replace(src, dst)
    finally:
        for p in temporary:
            p.unlink(missing_ok=True)


def _archive_results(prefix, reason):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%f")
    archive = RESULTS / "archive"
    archive.mkdir(exist_ok=True)
    mapping = []
    for source in _result_files(prefix):
        if source.exists():
            target = archive / f"{source.stem}_{stamp}_{reason}{source.suffix}"
            shutil.copy2(source, target)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise IOError(f"Archive verification failed: {source}")
            mapping.append(dict(source=str(source), archive=str(target), sha256=digest))
    (archive / f"{prefix}_{stamp}_{reason}_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n")
    return mapping


def merge_results(addition_prefix, base_prefix="sweep"):
    """Append new families only after checking exact axes, truth and complete cells."""
    base_files, extra_files = _result_files(base_prefix), _result_files(addition_prefix)
    base, extra = (xr.load_dataset(files[0]) for files in (base_files, extra_files))
    if set(base.data_vars) != set(extra.data_vars):
        raise ValueError("Sweep variables differ")
    for dim in base.dims:
        if dim != "family" and not base[dim].equals(extra[dim]):
            raise ValueError(f"Sweep coordinate mismatch: {dim}")
    if set(base.family.values) & set(extra.family.values):
        raise ValueError("Family already present; refusing to replace an existing arm")
    xr.testing.assert_identical(base.beta_true_map, extra.beta_true_map)
    for attr in ("beta_true_peak_hbo", "beta_true_peak_hbr"):
        if base.attrs[attr] != extra.attrs[attr]:
            raise ValueError(f"Ground-truth peak mismatch: {attr}")
    key = ["family", "window_s", "constellation", "motion", "noise_model", "chromo"]
    frames = []
    for files, data in ((base_files, base), (extra_files, extra)):
        frame = pd.read_csv(files[1])
        expected = pd.MultiIndex.from_product([data[k].values for k in key], names=key)
        found = pd.MultiIndex.from_frame(frame[key])
        if found.has_duplicates or set(found) != set(expected):
            raise ValueError(f"Incomplete or duplicate summary cells: {files[1]}")
        if not ((frame.n_channels == data.sizes["channel"]).all()
                and (frame.n_seeds == data.sizes["seed"]).all()):
            raise ValueError(f"Summary sample counts differ: {files[1]}")
        frames.append(frame)
    merged = xr.concat([base, extra], dim="family", data_vars="minimal",
                       coords="minimal", compat="equals", join="exact")
    meta = json.loads(base_files[2].read_text())
    meta["config"]["families"] = merged.family.values.tolist()
    extra_meta = json.loads(extra_files[2].read_text())
    for name in ("n_cells", "wall_s"):
        meta[name] += extra_meta[name]
    meta.setdefault("additional_runs", []).append(dict(prefix=addition_prefix,
                                                       metadata=extra_meta))
    meta["archive_before_merge"] = _archive_results(base_prefix, "B1_before_merge")
    _write_results(base_prefix, merged, pd.concat(frames, ignore_index=True), meta)
    return meta


def migrate_dct_adjusted_r2(prefix="sweep"):
    """Correct only DCT adjusted R² from stored per-seed/channel R², without fits.

    Window lengths follow pipeline.build's actual raw time-axis slicing; native
    DCT counts follow Cedalion's mean sampling interval, including K=0. Validate
    every old per-seed/chromophore value before replacing any result file.
    """
    files = _result_files(prefix)
    ds, frame = xr.load_dataset(files[0]), pd.read_csv(files[1])
    if ds.attrs.get("dct_constant_schema") == "one_constant_v2":
        raise ValueError("DCT correction already applied")
    rec = cedalion.data.get_nn22_resting_state()
    time_coord = rec["amp"].time
    raw_time = np.asarray(time_coord.values)
    slice_fs = 1.0 / float(np.median(np.diff(raw_time)))
    motion_columns = sum(rec.aux_ts[k].sizes["aux_channel"]
                         for k in ("accelerometer", "gyroscope") if k in rec.aux_ts)
    counts = []
    for family in ds.family.values:
        if not str(family).startswith("dct:"):
            continue
        cutoff = float(str(family).partition(":")[2])
        for window in ds.window_s.values:
            times = raw_time[:int(round(float(window) * slice_fs))]
            n = len(times)
            fs_hz = 1.0 / float(np.diff(times).mean())
            native_k = int(np.floor(2 * n * cutoff / fs_hz))
            for constellation in ds.constellation.values:
                extra_p = 0
                for part in str(constellation).split("+"):
                    if part == "motion":
                        extra_p += motion_columns
                    elif part in ("global", "short_avg", "short_maxcorr", "short_closest"):
                        extra_p += 1
                    elif part != "baseline":
                        raise ValueError(f"Unsupported stored constellation: {part}")
                old_p = 1 + native_k + 1 + extra_p  # HRF, native DCT, old offset, extras
                new_p = 1 + max(native_k, 1) + extra_p
                sel = dict(family=family, window_s=window, constellation=constellation)
                stored_r2 = ds.r2.sel(**sel)
                expected_old = 1 - (1 - stored_r2) * (n - 1) / (n - old_p)
                np.testing.assert_allclose(ds.r2_adj.sel(**sel), expected_old,
                                           rtol=2e-12, atol=2e-12, equal_nan=True,
                                           err_msg=f"Stored column count inconsistent: {sel}")
                corrected = 1 - (1 - stored_r2) * (n - 1) / (n - new_p)
                ds.r2_adj.loc[sel] = corrected
                for motion in ds.motion.values:
                    for noise in ds.noise_model.values:
                        for chromo in ds.chromo.values:
                            row = ((frame.family == family) & (frame.window_s == window)
                                   & (frame.constellation == constellation)
                                   & (frame.motion == motion) & (frame.noise_model == noise)
                                   & (frame.chromo == chromo))
                            if int(row.sum()) != 1:
                                raise ValueError(f"Expected one summary row: {sel}")
                            value = corrected.sel(motion=motion, noise_model=noise,
                                                  chromo=chromo).median()
                            frame.loc[row, "r2_adj_med"] = float(value)
                counts.append(dict(family=str(family), window_s=float(window),
                                   constellation=str(constellation), n_samples=n,
                                   native_dct_columns=native_k,
                                   old_columns=old_p, corrected_columns=new_p))
    ds.attrs["dct_constant_schema"] = "one_constant_v2"
    meta = json.loads(files[2].read_text())
    meta["dct_adjusted_r2_migration"] = dict(counts=counts, refitted=False,
        archive=_archive_results(prefix, "R06_before_dct_correction"))
    _write_results(prefix, ds, frame, meta)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", default="pilot",
                        choices=("pilot", "v4", "butterxy", "merge", "migrate-dct"))
    parser.add_argument("--output-prefix")
    parser.add_argument("--addition-prefix", default="sweep_butterxy")
    parser.add_argument("--replace", action="store_true",
                        help="vorhandene Ergebnisse archivieren und ersetzen")
    args = parser.parse_args()
    if args.preset == "merge":
        merge_results(args.addition_prefix, args.output_prefix or "sweep")
    elif args.preset == "migrate-dct":
        migrate_dct_adjusted_r2(args.output_prefix or "sweep")
    else:
        run({"pilot": PILOT, "v4": V4, "butterxy": BUTTERXY}[args.preset],
            output_prefix=args.output_prefix or
            ("sweep_butterxy" if args.preset == "butterxy" else "sweep"),
            replace=args.replace)
