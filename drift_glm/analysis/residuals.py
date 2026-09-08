"""Residual-Analyse auf den simulierten nn22-Daten: WAS konnte das Modell nicht fitten?

Der Sweep liefert die Residual-KENNZAHLEN (R^2, Residual-RMS) fuer das volle Raster;
dieses Modul zeigt die Residuen selbst -- als Zeitspuren und als Spektren -- und
beantwortet damit die Frage aus den Gespraechsnotizen 2026-09-08 inhaltlich: Bleibt
niederfrequenter Drift uebrig ("Modell zu schwach"), oder nur breitbandiges Rauschen?
Dazu die zweite Notiz-Frage: WIE unterscheiden sich die HRF-Schaetzungen der Familien?
Mit der FLEXIBLEN Recovery-Basis (GaussianKernels, wie flex_basis.py) wird die
rueckgewonnene HRF-Form je Familie sichtbar, onset-gelockt und gegen die eingemischte
Ground Truth gestellt.

Erzeugt:
  figures/27_residual_analysis.png     Beispiel-Residualspuren + Residual-Spektren je
                                       Familie (mit dem Rohsignal als Referenz)
  figures/28_hrf_family_comparison.png rueckgewonnene HRF je Familie vs. Ground Truth
                                       (flexible Basis, Mittel ueber die Blob-Kanaele)
  results/residuals_summary.csv        R^2 / Residual-RMS / Niederfrequenz-Anteil

Der Niederfrequenz-Anteil (`lowfreq_frac`) ist der Anteil der Residualleistung
unterhalb von DRIFT_FMAX (0,02 Hz): gross = das Driftmodell hat Trend uebrig gelassen.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.residuals         # voll (~20 min)
    conda run -n cedalion python -m drift_glm.analysis.residuals test    # Code-Pfade
"""

from __future__ import annotations

import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sps

import cedalion.models.glm as glm
from cedalion import units

from drift_glm import figstyle, paths
from drift_glm.analysis.sweep import drift_dm
from drift_glm.core import fitstats as fs
from drift_glm.core import pipeline as pl
from drift_glm.core import shortchannel as sc

RESULTS = paths.RESULTS
OUT = paths.FIGURES

FAMILIES = ["none", "poly:1", "poly:3", "poly:5", "dct:0.01", "dct:0.02",
            "legendre:3", "bspline:5", "butter:0.01"]
WINDOWS = [368.0, 90.0]
SEED = 0
NCH = 20
NOISE_MODEL = "ar_irls"
DRIFT_FMAX = 0.02        # Hz; Grenze des "Driftbands" fuer lowfreq_frac
HBO_COLOR, HBR_COLOR = "#c44e52", "#4c72b0"   # Konvention wie Abb. 19


def _subset(P):
    """Fit-Subset wie im Sweep: die NCH staerkst-aktivierten LANGEN Kanaele."""
    ts_long, _ = sc.split(P.conc_syn, P.geo3d)
    bt = P.beta_true_map.sel(channel=ts_long.channel)
    if "trial_type" in bt.dims:
        bt = bt.max("trial_type")
    w = np.abs(bt.sel(chromo="HbO").values)
    idx = np.sort(np.argsort(-w)[:NCH])
    chans = [str(c) for c in ts_long.channel.values[idx]]
    return ts_long.sel(channel=chans), bt.sel(channel=chans)


def _psd(resid_hbo, fsamp):
    """Median-Residualspektrum ueber Kanaele (HbO). Rueckgabe (f, Pxx_med)."""
    a = np.asarray(resid_hbo.transpose("channel", "time").values, float)
    f, pxx = sps.welch(a, fs=fsamp, nperseg=min(a.shape[1], 1024), axis=1)
    return f, np.median(pxx, axis=0)


def _lowfreq_frac(resid, fsamp):
    """Anteil der Residualleistung unterhalb DRIFT_FMAX, je (channel, chromo)."""
    a = np.asarray(resid.transpose("channel", "chromo", "time").values, float)
    f, pxx = sps.welch(a, fs=fsamp, nperseg=min(a.shape[-1], 1024), axis=-1)
    lo = pxx[..., f <= DRIFT_FMAX].sum(axis=-1)
    tot = pxx.sum(axis=-1)
    ref = resid.isel(time=0, drop=True).transpose("channel", "chromo")
    return ref.copy(data=lo / np.maximum(tot, 1e-30))


def main(mode: str = "full"):
    paths.ensure()
    families = ["poly:3", "none"] if mode == "test" else FAMILIES
    windows = [90.0] if mode == "test" else WINDOWS
    rows = []
    traces = {}      # (win, family) -> (t, resid_beispielkanal, y_beispielkanal)
    spectra = {}     # (win, family) -> (f, psd_med) ; (win, "__data__") als Referenz
    hrf_curves = {}  # family -> dict(reltime, est/gt je chromo, shape_r)
    t0 = time.time()

    for win in windows:
        P = pl.build(window_s=win, seed=SEED)
        ts, bt = _subset(P)
        fsamp = 1.0 / float(np.median(np.diff(ts.time.values)))
        ch0 = str(bt.channel.values[int(np.argmax(np.abs(bt.sel(chromo="HbO").values)))])
        y_all = fs.dequantify(ts)
        f_ref, p_ref = _psd(y_all.sel(chromo="HbO") - y_all.sel(chromo="HbO").mean("time"),
                            fsamp)
        spectra[(win, "__data__")] = (f_ref, p_ref)

        for fam in families:
            dm_drift, filt = drift_dm(fam, P.conc)
            ts_f = ts if filt is None else ts.cd.freq_filter(
                filt[0] * units.Hz, filt[1] * units.Hz, 4)
            dm = P.dm_hrf & dm_drift
            betas = glm.fit(ts_f, dm, noise_model=NOISE_MODEL, ar_order=30,
                            max_jobs=2).sm.params
            resid = fs.residuals(ts_f, betas, dm)
            q = fs.fit_metrics(ts_f, betas, dm)
            lf = _lowfreq_frac(resid, fsamp)
            for c in ("HbO", "HbR"):
                rows.append(dict(
                    family=fam, window_s=win, chromo=c, noise_model=NOISE_MODEL,
                    n_channels=int(resid.sizes["channel"]),
                    n_regressors=q.attrs["n_regressors"],
                    r2_med=float(q.r2.sel(chromo=c).median()),
                    r2_adj_med=float(q.r2_adj.sel(chromo=c).median()),
                    resid_rms_med=float(q.resid_rms.sel(chromo=c).median()),
                    lowfreq_frac_med=float(lf.sel(chromo=c).median())))
            traces[(win, fam)] = (
                ts.time.values,
                np.asarray(resid.sel(channel=ch0, chromo="HbO").values, float),
                np.asarray(y_all.sel(channel=ch0, chromo="HbO").values, float))
            spectra[(win, fam)] = _psd(resid.sel(chromo="HbO"), fsamp)
            print(f"  win={win:g} {fam:12s} r2_adj(HbO)="
                  f"{rows[-2]['r2_adj_med']:.3f} lowfreq="
                  f"{rows[-2]['lowfreq_frac_med']:.2f} "
                  f"({time.time() - t0:5.0f}s)", flush=True)

        # ---- HRF-Formvergleich mit flexibler Basis (nur laengstes Fenster) ----
        if win != max(windows):
            continue
        from drift_glm.analysis import imageglm as ig
        flex = glm.GaussianKernels(t_pre=0 * units.s, t_post=20 * units.s,
                                   t_delta=2 * units.s, t_std=1.5 * units.s)
        dm_flex = glm.design_matrix.hrf_regressors(P.conc, P.stim_df, flex)
        tts = [str(t) for t in pd.unique(P.stim_df.trial_type)]

        def _blocks(x):
            ep = x.cd.to_epochs(P.stim_df, tts, before=ig.EPOCH_BEFORE,
                                after=ig.EPOCH_AFTER)
            ep = ep - ep.sel(reltime=(ep.reltime < 0)).mean("reltime")
            ba = ep.groupby("trial_type").mean("epoch")
            return ba.mean("trial_type") if "trial_type" in ba.dims else ba

        gt_ba = _blocks(ig.with_time_coords(P.activation.sel(channel=ts.channel), ts))
        for fam in families:
            dm_drift, filt = drift_dm(fam, P.conc)
            ts_f = ts if filt is None else ts.cd.freq_filter(
                filt[0] * units.Hz, filt[1] * units.Hz, 4)
            betas = glm.fit(ts_f, dm_flex & dm_drift, noise_model=NOISE_MODEL,
                            ar_order=30, max_jobs=2).sm.params
            hsel = betas.regressor.str.startswith("HRF")
            rec = ig.with_time_coords(
                glm.predict(ts_f, betas.sel(regressor=hsel), dm_flex)
                .transpose(*ts.dims), ts)
            est_ba = _blocks(rec)
            entry = dict(reltime=np.asarray(gt_ba.reltime.values, float))
            for c in ("HbO", "HbR"):
                e = est_ba.sel(chromo=c).mean("channel").values
                g = gt_ba.sel(chromo=c).mean("channel").values
                entry[f"est_{c}"] = np.asarray(e, float)
                entry[f"gt_{c}"] = np.asarray(g, float)
            entry["shape_r"] = float(np.corrcoef(entry["est_HbO"],
                                                 entry["gt_HbO"])[0, 1])
            hrf_curves[fam] = entry
            print(f"  flex {fam:12s} Form-r(HbO)={entry['shape_r']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "residuals_summary.csv", index=False)
    print(f"\n-> {RESULTS / 'residuals_summary.csv'}")
    print(df[df.chromo == "HbO"]
          [["family", "window_s", "r2_adj_med", "resid_rms_med", "lowfreq_frac_med"]]
          .to_string(index=False))

    _fig27(families, windows, traces, spectra)
    if hrf_curves:
        _fig28(families, hrf_curves, max(windows))
    print(f"\nFertig in {(time.time() - t0) / 60:.1f} min.")


def _fig27(families, windows, traces, spectra):
    """Abb. 27: Residualspuren (Beispielkanal) + Residual-Spektren je Familie."""
    nrow = len(windows)
    fig, axes = plt.subplots(nrow, 2, figsize=(15, 5.2 * nrow), squeeze=False)
    for i, win in enumerate(sorted(windows, reverse=True)):
        ax = axes[i, 0]
        t, _, y0 = traces[(win, families[0])]
        # Versatz gross genug, dass auch ein Residuum mit Resttrend (poly bei kurzen
        # Fenstern laesst einen zurueck) nicht in die Nachbarspur laeuft.
        step = float(np.percentile(np.abs(y0 - y0.mean()), 99)) * 1.7
        ax.plot(t, (y0 - y0.mean()) + step, color="0.75", lw=0.7)
        ax.text(t[-1], step, " Daten", va="center", fontsize=8, color="0.4")
        for k, fam in enumerate(families):
            _, r, _ = traces[(win, fam)]
            off = -k * step
            ax.plot(t, r + off, color=figstyle.family_color(fam), lw=0.8)
            ax.text(t[-1], off, f" {fam}", va="center", fontsize=8,
                    color=figstyle.family_color(fam))
        ax.set_title(f"Residuum des staerksten Kanals (HbO) | Fenster {win:g} s\n"
                     "oben grau: gefittete Zeitreihe (mittelwertbereinigt)")
        ax.set_xlabel("Zeit [s]")
        ax.set_yticks([])
        ax.set_ylabel(f"Familien, versetzt um {step:.2f} µM")
        ax.margins(x=0.12)

        ax = axes[i, 1]
        f_ref, p_ref = spectra[(win, "__data__")]
        ax.loglog(f_ref[1:], p_ref[1:], color="0.75", lw=1.6, label="Daten (vor Fit)")
        for fam in families:
            f, p = spectra[(win, fam)]
            ax.loglog(f[1:], p[1:], color=figstyle.family_color(fam), lw=1.1,
                      label=fam)
        ax.axvspan(f_ref[1] * 0.5, DRIFT_FMAX, color="0.85", alpha=0.5, zorder=0)
        ax.text(DRIFT_FMAX, ax.get_ylim()[0], "Driftband ", ha="right", va="bottom",
                fontsize=8, color="0.35")
        ax.set_title(f"Residual-Leistungsspektrum (Median über Kanäle, HbO) | "
                     f"Fenster {win:g} s")
        ax.set_xlabel("Frequenz [Hz]")
        ax.set_ylabel("PSD [µM²/Hz]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Was konnte das Modell nicht fitten? Residuen im Zeit- und "
                 "Frequenzraum (AR-IRLS, baseline)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "27_residual_analysis.png", dpi=130)
    plt.close(fig)
    print(f"-> {OUT / '27_residual_analysis.png'}")


def _fig28(families, hrf_curves, win):
    """Abb. 28: rueckgewonnene HRF je Familie (flexible Basis) vs. Ground Truth."""
    ncol = 3
    nrow = int(np.ceil(len(families) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for k, fam in enumerate(families):
        ax = axes[k]
        e = hrf_curves[fam]
        for c, col in (("HbO", HBO_COLOR), ("HbR", HBR_COLOR)):
            ax.plot(e["reltime"], e[f"gt_{c}"], color=col, lw=2.6, alpha=0.4)
            ax.plot(e["reltime"], e[f"est_{c}"], color=col, lw=1.3, ls="--")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{fam}   Form-r = {e['shape_r']:+.3f}", fontsize=9,
                     color=figstyle.family_color(fam))
        ax.grid(alpha=0.25)
    for k in range(len(families), len(axes)):
        axes[k].set_axis_off()
    for r in range(nrow):
        axes[r * ncol].set_ylabel(r"$\Delta c$ [µM]")
    for k in range(len(families) - ncol, len(families)):
        axes[k].set_xlabel("Zeit nach Stimulus-Onset [s]")
    fig.suptitle("Wie unterscheiden sich die HRF-Schätzungen? Rückgewonnene HRF je "
                 "Driftfamilie (flexible Basis, gestrichelt)\ngegen die eingemischte "
                 f"Ground Truth (dick, blass) · rot HbO, blau HbR · Fenster {win:g} s, "
                 "Mittel über die Blob-Kanäle", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "28_hrf_family_comparison.png", dpi=130)
    plt.close(fig)
    print(f"-> {OUT / '28_hrf_family_comparison.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
