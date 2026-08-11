"""Systematischer Vergleichs-Sweep der Driftregressoren -- Kernstueck der BA.

Variiert (unabhaengige Variablen aus dem Expose):
  - Driftfamilie: poly n=1..5, DCT (mehrere Cutoffs), Legendre, "none" (nur Offset),
    Butterworth-Hochpass (Vorverarbeitungs-Alternative STATT Driftregressoren)
  - Analysefenster (window_s)
  - Regressor-Konstellation: baseline / +motion / +global / +motion+global
  - Seeds (Monte-Carlo ueber die Stimulus-Platzierung)

Metriken (getrennt HbO/HbR), aggregiert UEBER SEEDS (echte MC-Bias/Varianz),
danach ueber Kanaele zusammengefasst:
  - Bias, Varianz, RMSE von beta_hat vs. Ground Truth
  - HbO/HbR-Plausibilitaet (Korrelation der beta ueber Kanaele; rueckgew. Ratio)

Schaetzer: AR-IRLS (Default). Ergebnisse -> results/.

BETREUUNGSHINWEISE, die hier umgesetzt sind:
  * Butterworth-Hochpass ist die Alternative STATT Driftregressoren -> dort keine
    Drift-Regressoren (nur Offset). In allen anderen Armen wird NICHT gefiltert;
    der Drift wird ausschliesslich ueber Regressoren modelliert.
  * Short-Channel-Regression ist auf nn22 NICHT moeglich (keine Short-Separation-
    Kanaele, min. Distanz 15.6 mm) -> ersetzt durch global_mean_regressor als
    oberflaechliches/systemisches Surrogat. Echte SC-Regression erst mit realen
    DOT-Daten, deren Montage Short-Channels enthaelt.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.sweep pilot   # kleiner Test + Timing
    conda run -n cedalion python -m drift_glm.analysis.sweep full     # voller Sweep
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

from drift_glm.core import pipeline as pl
from drift_glm.core import preprocess as prep
from drift_glm.core import shortchannel as sc
from drift_glm import paths

RESULTS = paths.RESULTS
HRF_REG = "HRF Stim"


PILOT = dict(   # klein, aber testet JEDEN Code-Pfad (alle Familien-Typen + Konstellationen)
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

FULL = dict(
    families=["poly:1", "poly:2", "poly:3", "poly:4", "poly:5",
              "dct:0.005", "dct:0.01", "dct:0.02",
              "legendre:1", "legendre:3", "legendre:5",
              "none", "butter:0.01"],
    windows=[90.0, 180.0, 368.0],
    constellations=["baseline", "motion", "global", "motion+global"],
    seeds=list(range(8)),
    motion_methods=["wavelet"],
    n_channels=20,
    noise_models=["ar_irls"],
    ar_order=30,
)

# Schnelle Version (~3 h): volles Familien-/Konstellations-Raster, aber nur 2 Fenster
# und 6 Seeds auf 20-Kanal-Subset.
QUICK = dict(
    families=FULL["families"],
    windows=[90.0, 180.0],
    constellations=FULL["constellations"],
    seeds=list(range(6)),
    motion_methods=["wavelet"],
    n_channels=20,
    noise_models=["ar_irls"],
    ar_order=30,
)

# v3 (~4-5 h): volles Raster inkl. B-Splines, 3 Fenster {90,180,368 s}, 4 Seeds.
V3 = dict(
    families=FULL["families"] + ["bspline:5", "bspline:8"],
    windows=[90.0, 180.0, 368.0],
    constellations=["baseline", "motion", "global", "motion+global"],
    seeds=list(range(4)),
    motion_methods=["wavelet"],
    n_channels=20,
    noise_models=["ar_irls"],
    ar_order=30,
)

# v4 (Nachtlauf, ~8-9 h): wie v3, aber mit ECHTER Short-Channel-Regression statt nur dem
# Global-Mean-Surrogat, und mit der Motion Correction als eigener Achse.
#
# Warum die Motion-Achse: TDDR daempft das Driftband auf 55.6 % UND die eingemischte HRF
# auf 70 %, Wavelet laesst beides unangetastet. Weil beide Verfahren aus der
# Betreuungsvorgabe stammen, wird nicht stillschweigend eines gewaehlt, sondern der
# Unterschied beziffert. Gekreuzt mit der Konstellation "motion" (Motion-REGRESSOREN in
# der Designmatrix) beantwortet das zugleich die Frage, ob sich beides doppelt.
#
# Konstellationen: baseline / motion / global (Surrogat, fuer Anschluss an v3) /
# short_avg + short_maxcorr (echte Short-Channel-Regression, Betreuungsvorgabe).
# 15 Familien x 3 Fenster x 5 Konstellationen x 2 Motion x 4 Seeds = 1800 Fits.
V4 = dict(
    families=FULL["families"] + ["bspline:5", "bspline:8"],
    windows=[90.0, 180.0, 368.0],
    constellations=["baseline", "motion", "global", "short_avg", "short_maxcorr"],
    seeds=list(range(4)),
    motion_methods=["wavelet", "tddr+wavelet"],
    n_channels=20,
    noise_models=["ar_irls"],
    ar_order=30,
)

# v5 (Nachtlauf): v4 plus die beiden restlichen Betreuungspunkte --
#   * Filter-Arm vollstaendig: Hochpass 0.01 (butter), Tiefpass 0.5, und beides als
#     Bandpass. Alle drei OHNE Driftregressoren (Alternative, nicht Ergaenzung), im
#     Konzentrationsraum. Der Bandpass entspricht genau der Vorverarbeitung, die die
#     Autoren des realen Datensatzes angewandt haben -- damit direkt anschlussfaehig.
#   * OLS als zweites Rauschmodell neben AR-IRLS. Kostet wenig, weil OLS um ein
#     Vielfaches schneller ist als AR-IRLS.
# 17 Familien x 3 Fenster x 5 Konstellationen x 2 Motion x 2 Rauschmodelle x 4 Seeds
# = 4080 Fits; davon die Haelfte OLS (billig).
V5 = dict(
    families=FULL["families"] + ["bspline:5", "bspline:8",
                                 "lowpass:0.5", "bandpass:0.01-0.5"],
    windows=[90.0, 180.0, 368.0],
    constellations=["baseline", "motion", "global", "short_avg", "short_maxcorr"],
    seeds=list(range(4)),
    motion_methods=["wavelet", "tddr+wavelet"],
    n_channels=20,
    noise_models=["ar_irls", "ols"],
    ar_order=30,
)


def drift_dm(family: str, conc):
    """(DesignMatrix inkl. Offset, filter|None) fuer eine Driftfamilie.

    `filter` ist None oder ein Paar `(fmin, fmax)` in Hz fuer `freq_filter`. Cedalions
    Konvention dort: `fmax=0` -> Hochpass bei fmin, `fmin=0` -> Tiefpass bei fmax,
    beides gesetzt -> Bandpass.

    Die Filter-Familien sind Vorverarbeitungs-ALTERNATIVEN und bekommen deshalb KEINE
    Driftregressoren, nur den Offset (Betreuungsvorgabe: "entweder Driftregressor oder
    Highpassfilter"). Angewandt wird im Konzentrationsraum, ebenfalls laut Vorgabe.
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

    Kubische B-Spline-Basis ueber das Analysefenster. Eine geklemmte B-Spline-Basis
    bildet eine Zerlegung der Eins und enthaelt damit den konstanten Offset. Namen
    'Drift BS i'. (Optionale Expose-Erweiterung, ueber die xarray-Designmatrix
    ohne Eingriff in Cedalion-Kernroutinen ergaenzt.)
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
    "motion", "global", "short_avg", "motion+global". "baseline" bedeutet: nichts dazu.
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
    """Live-Fortschrittsdatei (nach jedem Fit aktualisiert) zum Mitverfolgen."""
    el = time.time() - t0
    med = float(np.median(timings)) if timings else 0.0
    eta = med * (total - done)
    (paths.LOGS / "sweep_progress.txt").write_text(
        f"Fortschritt: {done}/{total} ({100 * done / total:.0f} %)\n"
        f"verstrichen : {el / 60:5.1f} min\n"
        f"ETA (Rest)  : {eta / 60:5.1f} min\n"
        f"median/Fit  : {med:4.1f} s\n"
        f"letzte Zelle: {last}\n"
        f"(Datei wird nach JEDEM Fit aktualisiert)\n"
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
            # Analysiert wird NUR ueber die langen Kanaele (Betreuungsvorgabe): die
            # kurzen dienen als Regressor, nicht als Messgroesse. Sie haben ohnehin
            # keine injizierte Aktivierung (pipeline.inject_long_only).
            ts_long, ts_short = sc.split(P.conc_syn, P.geo3d)
            bt_long = P.beta_true_map.sel(channel=ts_long.channel)
            if ref_channels is None:
                # Fit-Subset = die N staerkst-aktivierten LANGEN Kanaele (Blob-Kern).
                # EINMAL bestimmt und danach per Label festgehalten: die Kanalmasken
                # werden nach der Motion Correction berechnet und koennen daher je
                # Motion-Stufe leicht abweichen. Ohne feste Referenz wuerden die
                # Achsen unterschiedliche Kanalmengen vergleichen.
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
                    # Filter-Alternative: im KONZENTRATIONSRAUM, nach der Augmentation
                    # (Betreuungsvorgabe). ts_base ist bereits Konzentration.
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
                        dt = time.time() - tc
                        raw[(family, win, con, mc, nm)][seed] = \
                            betas.sel(regressor=HRF_REG)
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
    sample = next(iter(raw.values()))[seeds[0]]
    chrom = [str(c) for c in sample.chromo.values]

    # Kanal-Schnittmenge ueber ALLE Zellen: die Masken werden nach der Motion Correction
    # bestimmt und koennen je Stufe minimal abweichen. Verglichen wird nur, was ueberall
    # vorhanden ist -- sonst mischt sich ein Kanalauswahl-Effekt in den Achsenvergleich.
    common = None
    for by_seed in raw.values():
        for da in by_seed.values():
            s = {str(c) for c in da.channel.values}
            common = s if common is None else (common & s)
    chans = [str(c) for c in sample.channel.values if str(c) in common]
    if len(chans) < sample.sizes["channel"]:
        print(f"     Hinweis: {sample.sizes['channel'] - len(chans)} Kanaele nicht in "
              f"allen Zellen vorhanden -> auf {len(chans)} gemeinsame beschraenkt")

    # 8D-Rohwerte beta_hat(family, window, constellation, motion, noise_model,
    #                      seed, channel, chromo)
    arr = np.full((len(fams), len(wins), len(cons), len(mcs), len(nms), len(seeds),
                   len(chans), len(chrom)), np.nan)
    for (f, w, c, m, n), by_seed in raw.items():
        idx = (fams.index(f), wins.index(w), cons.index(c), mcs.index(m), nms.index(n))
        for si, s in enumerate(seeds):
            arr[idx + (si,)] = (
                by_seed[s].sel(channel=chans).transpose("channel", "chromo").values)
    bhat = xr.DataArray(
        arr, dims=("family", "window_s", "constellation", "motion", "noise_model",
                   "seed", "channel", "chromo"),
        coords=dict(family=fams, window_s=wins, constellation=cons, motion=mcs,
                    noise_model=nms, seed=seeds, channel=chans, chromo=chrom))

    # per-Kanal Ground-Truth (raeumlicher Blob), auf bhat-Koordinaten ausgerichtet
    bt = xr.DataArray(
        beta_true_map.sel(channel=chans).transpose("channel", "chromo").values,
        dims=("channel", "chromo"), coords={"channel": chans, "chromo": chrom})
    mean_seed = bhat.mean("seed")
    bias = mean_seed - bt
    var = bhat.var("seed")
    rmse = np.sqrt(((bhat - bt) ** 2).mean("seed"))

    ds = xr.Dataset(dict(bhat=bhat, bias=bias, var=var, rmse=rmse, beta_true_map=bt))
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
    # Bias getrennt nach Chromophor: TDDR daempft die HRF und kann so eine
    # Ueberschaetzung zufaellig kompensieren -- am RMSE allein nicht erkennbar.
    print("\nMotion-Achse (Mittel ueber Familien/Fenster/Konstellationen):")
    print(df.groupby(["chromo", "motion"])[["bias_med", "rmse_med"]].mean()
            .to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    run({"pilot": PILOT, "quick": QUICK, "full": FULL,
         "v3": V3, "v4": V4, "v5": V5}[preset])
