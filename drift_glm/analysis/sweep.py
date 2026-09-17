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

import json
import sys
import time
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
# und die eingemischte HRF daempft (auf ~70 %), Wavelet nicht; gekreuzt mit der
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
        return dm & offset_only(), None                               # + Offset
    if fam == "bspline":
        return _bspline_dm(conc, int(param)), None    # enthaelt Offset (Zerlegung d. Eins)
    if fam == "none":
        return offset_only(), None                                    # nur Offset
    if fam == "butter":              # Hochpass, z.B. butter:0.01
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


def run(cfg: dict):
    paths.ensure()
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
                ts_fam = ts_base
                if filt is not None:
                    # Filter-Alternative im Konzentrationsraum, nach der Augmentation.
                    ts_fam = ts_base.cd.freq_filter(
                        filt[0] * units.Hz, filt[1] * units.Hz, 4)
                for con in cfg["constellations"]:
                    extra = constellation_dm(con, parts)
                    dm = P.dm_hrf & dm_drift
                    if extra is not None:
                        dm = dm & extra
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

    _aggregate_and_export(cfg, raw, beta_true, beta_true_map, timings, time.time() - t0)


def _aggregate_and_export(cfg, raw, beta_true, beta_true_map, timings, wall):
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
    paths.ensure()
    ds.to_netcdf(RESULTS / "sweep_per_channel.nc")

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
    df.to_csv(RESULTS / "sweep_summary.csv", index=False)

    meta = dict(config=cfg, n_cells=len(timings),
                wall_s=round(wall, 1),
                cell_time_s=dict(min=round(float(np.min(timings)), 2),
                                 median=round(float(np.median(timings)), 2),
                                 max=round(float(np.max(timings)), 2)))
    (RESULTS / "sweep_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[OK] {len(timings)} Fits in {wall/60:.1f} min "
          f"(median {np.median(timings):.1f}s/Fit).")
    print(f"     -> {RESULTS/'sweep_summary.csv'}")
    print(f"     -> {RESULTS/'sweep_per_channel.nc'}")
    print("\nBeste Familie je (Fenster, chromo) nach RMSE_med:")
    best = df.loc[df.groupby(["chromo", "window_s"])["rmse_med"].idxmin()]
    print(best[["chromo", "window_s", "family", "constellation", "motion",
                "rmse_med", "absbias_med", "var_med"]].to_string(index=False))
    # Bias getrennt nach Chromophor: TDDR daempft die HRF, was am RMSE allein nicht
    # sichtbar ist.
    print("\nMotion-Achse (Mittel ueber Familien/Fenster/Konstellationen):")
    print(df.groupby(["chromo", "motion"])[["bias_med", "rmse_med"]].mean()
            .to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    run({"pilot": PILOT, "v4": V4}[preset])
