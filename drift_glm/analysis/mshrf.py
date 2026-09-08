"""Multisubject: geschaetzte HRF je Driftfamilie GEGENUEBERGESTELLT (echte Shorts).

Der Punkt aus den Gespraechsnotizen 2026-09-08: "bei realdaten hrf gegenuebergestellt
visualisieren bei versch. vorgaben". Die Vorgaben sind die Driftfamilien; der Datensatz
ist der Multisubject-Fingertapping (5 Probanden, 8 ECHTE Short-Channels bei 7-8 mm),
weil nur er eine echte Short-Channel-Regression erlaubt (Khan folgt separat).

Damit sich die HRF-FORM zwischen den Familien ueberhaupt unterscheiden kann, wird mit
der FLEXIBLEN Basis gefittet (GaussianKernels, wie flex_basis.py) -- mit fester
Gamma-Basis waere jede Kurve nur ein skaliertes Abbild derselben Form. Als
familienUNabhaengige Referenz dient der Block-Mittelwert der GEMESSENEN Zeitreihe:
er zeigt, was in den Daten steckt, bevor ein Modell sie zerlegt.

Aufbau je Zelle (Proband x Familie): GLM mit flexibler HRF-Basis + Driftfamilie +
short_avg-Regressor (Designmatrix-Variante), OLS. OLS statt AR-IRLS aus dem in
`msglm.py` dokumentierten Laufzeit-/Speichergrund (AR-IRLS kostet auf diesen
Aufnahmen das ~200-fache und sprengte auf der dct:0.02-Matrix dreimal den Speicher);
fuer den FORMvergleich zaehlt der Punktschaetzer, nicht dessen Standardfehler.

ROI: je Hand die 3 staerksten KONTRAlateralen Kanaele, bestimmt aus dem
Block-Mittel der DATEN (familienunabhaengig -- keine Familie wird bevorzugt).

Erzeugt:
  figures/29_ms_hrf_families.png   HRF je Familie vs. Datenreferenz, je Hand x Chromophor
  figures/30_ms_fit_quality.png    Modellfit (adj. R^2, Residual-RMS) je Familie
  results/mshrf_summary.csv        Fit-Metriken je (Proband, Familie, Chromophor)

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.mshrf         # voll (~20 min)
    conda run -n cedalion python -m drift_glm.analysis.mshrf test    # 2 Probanden, 3 Familien
"""

from __future__ import annotations

import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import cedalion.nirs
import cedalion.models.glm as glm
from cedalion import units

from drift_glm import figstyle, paths
from drift_glm.analysis import msglm
from drift_glm.analysis.sweep import drift_dm
from drift_glm.core import fitstats as fs
from drift_glm.core import shortchannel as sc
from drift_glm.data import multisubject as ms

RESULTS = paths.RESULTS
OUT = paths.FIGURES

FAMILIES = list(msglm.FAMILIES)
SYSTEMIC = "short_avg"          # Designmatrix-Variante mit den echten kurzen Kanaelen
NOISE_MODEL = "ols"
EPOCH_BEFORE = 5.0 * units.s
EPOCH_AFTER = 20.0 * units.s
ROI_N = 3                       # Kanaele je Hand (kontralateral, aus den Daten)
PEAK_WINDOW_S = (3.0, 10.0)     # Fenster fuer die ROI-Wahl (Peak der Datenreferenz)
HBO_COLOR, HBR_COLOR = "#c44e52", "#4c72b0"


def _flex():
    return glm.GaussianKernels(t_pre=0 * units.s, t_post=20 * units.s,
                               t_delta=2 * units.s, t_std=1.5 * units.s)


def _epoch_mean(ts, stim, trial_types):
    """Baseline-korrigiertes Block-Mittel je trial_type: (trial_type, reltime, ...)."""
    ep = ts.cd.to_epochs(stim, list(trial_types), before=EPOCH_BEFORE,
                         after=EPOCH_AFTER)
    ep = ep - ep.sel(reltime=(ep.reltime < 0)).mean("reltime")
    return ep.groupby("trial_type").mean("epoch")


def _with_time(ts, like):
    """glm.predict verliert die time-Koordinaten; von `like` uebernehmen."""
    return ts.assign_coords(
        {k: like[k] for k in ("time", "samples") if k in like.coords})


def _roi_from_data(group_ba, geo3d) -> dict[str, list[str]]:
    """Je Hand die ROI_N staerksten kontralateralen Kanaele -- aus den DATEN.

    Score = HbO-Peak des Block-Mittels im Fenster PEAK_WINDOW_S. Familienunabhaengig,
    also fuer den Familienvergleich unverzerrt.
    """
    side = msglm.hemisphere_of(group_ba, geo3d)      # +1 links, -1 rechts
    rois = {}
    for tt, expect in ms.EXPECTED_SIDE.items():
        contra = side > 0 if expect == "C3" else side < 0
        d = group_ba.sel(trial_type=tt, chromo="HbO")
        d = d.sel(reltime=(d.reltime >= PEAK_WINDOW_S[0])
                  & (d.reltime <= PEAK_WINDOW_S[1])).max("reltime")
        v = np.where(contra, np.asarray(d.values, float), -np.inf)
        top = np.argsort(-v)[:ROI_N]
        rois[tt] = [str(c) for c in group_ba.channel.values[top]]
    return rois


def main(mode: str = "full"):
    paths.ensure()
    files = ms.paths()
    families = FAMILIES
    if mode == "test":
        files = files[:2]
        families = ["none", "poly:3", "dct:0.02"]

    t0 = time.time()
    prepped = {}
    for f in files:
        rec = ms.load(f)
        P, _ = ms.preprocess_recording(rec)
        long, short = cedalion.nirs.split_long_short_channels(
            P.conc, P.geo3d, distance_threshold=ms.SHORT_THRESHOLD)
        prepped[f] = (long, short, P.geo3d, ms.stim_df(rec, "hands"))
    long0, _, geo0, _ = prepped[files[0]]
    print(f"Vorverarbeitung: {len(files)} Probanden in {time.time() - t0:.0f}s",
          flush=True)

    # Familienunabhaengige Datenreferenz + ROI-Wahl.
    data_ba = [
        _epoch_mean(fs.dequantify(prepped[f][0]), prepped[f][3], ms.TAPPING)
        for f in files]
    # Kanalmengen koennen sich durchs Pruning minimal unterscheiden -> Schnittmenge.
    common = sorted(set.intersection(*[set(map(str, d.channel.values))
                                       for d in data_ba]))
    data_ba = [d.sel(channel=common) for d in data_ba]
    group_ba = xr.concat(data_ba, dim="subject").mean("subject")
    rois = _roi_from_data(group_ba, geo0)
    for tt, chs in rois.items():
        print(f"ROI {tt} (kontralateral, aus den Daten): {', '.join(chs)}", flush=True)

    rows = []
    curves = {}          # family -> {(tt, chromo): (reltime, mean, std) ueber Probanden}
    for fam in families:
        per_sub = {}     # (tt, chromo) -> Liste der Probanden-Kurven
        for f in files:
            long, short, geo3d, stim = prepped[f]
            dm_drift, filt = drift_dm(fam, long)
            ts_f = long if filt is None else long.cd.freq_filter(
                filt[0] * units.Hz, filt[1] * units.Hz, 4)
            dm = (glm.design_matrix.hrf_regressors(ts_f, stim, _flex())
                  & dm_drift & sc.short_dm(SYSTEMIC, ts_f, short, geo3d))
            params = glm.fit(ts_f, dm, noise_model=NOISE_MODEL,
                             max_jobs=1).sm.params
            q = fs.fit_metrics(ts_f, params, dm)
            for c in ("HbO", "HbR"):
                rows.append(dict(
                    subject=ms.subject_of(f), family=fam, systemic=SYSTEMIC,
                    noise_model=NOISE_MODEL, chromo=c,
                    n_channels=int(ts_f.sizes["channel"]),
                    n_regressors=q.attrs["n_regressors"],
                    r2_med=float(q.r2.sel(chromo=c).median()),
                    r2_adj_med=float(q.r2_adj.sel(chromo=c).median()),
                    resid_rms_med=float(q.resid_rms.sel(chromo=c).median())))
            hsel = params.regressor.str.startswith("HRF")
            dm_hrf_only = glm.design_matrix.hrf_regressors(ts_f, stim, _flex())
            rec_hrf = _with_time(
                glm.predict(ts_f, params.sel(regressor=hsel), dm_hrf_only)
                .transpose(*ts_f.dims), ts_f)
            ba = _epoch_mean(rec_hrf, stim, ms.TAPPING)
            for tt in ms.TAPPING:
                roi = [c for c in rois[tt] if c in map(str, ba.channel.values)]
                for c in ("HbO", "HbR"):
                    per_sub.setdefault((tt, c), []).append(np.asarray(
                        ba.sel(trial_type=tt, chromo=c, channel=roi)
                        .mean("channel").values, float))
        reltime = np.asarray(ba.reltime.values, float)
        curves[fam] = {k: (reltime, np.mean(v, axis=0), np.std(v, axis=0))
                       for k, v in per_sub.items()}
        print(f"  {fam:14s} fertig ({time.time() - t0:5.0f}s)", flush=True)

    # Datenreferenz auf dieselben ROIs verdichten.
    ref = {}
    for tt in ms.TAPPING:
        for c in ("HbO", "HbR"):
            v = [np.asarray(d.sel(trial_type=tt, chromo=c, channel=rois[tt])
                            .mean("channel").values, float) for d in data_ba]
            ref[(tt, c)] = (np.asarray(data_ba[0].reltime.values, float),
                            np.mean(v, axis=0), np.std(v, axis=0))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "mshrf_summary.csv", index=False)
    print(f"-> {RESULTS / 'mshrf_summary.csv'}")

    _fig29(families, curves, ref, len(files))
    _fig30(families, df, len(files))
    print(f"Fertig in {(time.time() - t0) / 60:.1f} min.")


def _fig29(families, curves, ref, n_subjects):
    """Abb. 29: HRF je Familie (flexible Basis) gegen die Datenreferenz."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True, sharey="row")
    for j, tt in enumerate(ms.TAPPING):
        for i, c in enumerate(["HbO", "HbR"]):
            ax = axes[i, j]
            t, m, s = ref[(tt, c)]
            ax.fill_between(t, m - s, m + s, color="0.85")
            ax.plot(t, m, color="0.35", lw=2.6, label="Daten (Block-Mittel)")
            for fam in families:
                t, m, _ = curves[fam][(tt, c)]
                ax.plot(t, m, color=figstyle.family_color(fam), lw=1.2, label=fam)
            ax.axvspan(0, 5, color="C2", alpha=0.08)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0, color="k", lw=0.6, ls=":")
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(f"{tt} → kontralaterale ROI", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Δ{c} [µM]")
            if i == 1:
                ax.set_xlabel("Zeit nach Stimulus-Onset [s]")
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Multisubject-Fingertapping: geschätzte HRF je Driftfamilie "
                 "(flexible Basis, +short_avg, OLS)\nDatenreferenz grau (±1 SD über "
                 f"{n_subjects} Probanden) · grün schattiert: Stimulusdauer · ROI: die "
                 f"{ROI_N} stärksten kontralateralen Kanäle", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "29_ms_hrf_families.png", dpi=130)
    plt.close(fig)
    print(f"-> {OUT / '29_ms_hrf_families.png'}")


def _fig30(families, df, n_subjects):
    """Abb. 30: Modellfit auf den Realdaten -- adj. R^2 und Residual-RMS je Familie."""
    fams = figstyle.order(families)
    x = np.arange(len(fams))
    colors = [figstyle.family_color(f) for f in fams]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for i, c in enumerate(["HbO", "HbR"]):
        for j, (col, label) in enumerate([("r2_adj_med", "median adj. R²"),
                                          ("resid_rms_med", "Residual-RMS [µM]")]):
            ax = axes[i, j]
            g = df[df.chromo == c].groupby("family")[col]
            mean, std = g.mean().reindex(fams), g.std().reindex(fams)
            bars = ax.bar(x, mean.values, yerr=std.values, color=colors,
                          error_kw=dict(lw=1, capsize=2))
            for bar, f in zip(bars, fams):
                if f.startswith(("butter", "lowpass", "bandpass")):
                    bar.set_hatch("//")
            ax.set_title(f"{c}: {label}")
            ax.grid(axis="y", alpha=0.3)
            if col.startswith("r2"):
                ax.set_ylim(0, 1)
    for j in range(2):
        axes[-1, j].set_xticks(x)
        axes[-1, j].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    fig.suptitle(f"Modellfit auf den Realdaten ({n_subjects} Probanden, ±1 SD): "
                 "variance explained und Residual-RMS je Driftfamilie\nschraffiert: "
                 "Filter-Arme (R² auf der gefilterten Zeitreihe)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "30_ms_fit_quality.png", dpi=130)
    plt.close(fig)
    print(f"-> {OUT / '30_ms_fit_quality.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
