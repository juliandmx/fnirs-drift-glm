"""Signifikanz je Kanal mit FDR-Korrektur und Detektionsguete gegen die Ground Truth.

Fittet je Driftfamilie ueber alle Kanaele, bildet t = beta/SE des HRF-Regressors je Kanal
(AR-IRLS), korrigiert mit Benjamini-Hochberg (q = 0.05) und vergleicht die detektierten
Kanaele mit dem wahren Blob (aktiv: |beta_true| > 10 % des Peaks). Metriken je
Familie/Chromophor, Mittel ueber Seeds: Sensitivitaet, Spezifitaet, Praezision, Youden-J,
Zahl detektierter Kanaele. Fenster 180 s, Konstellation baseline.
Ausgabe: results/detection_summary.csv, Abb. 14.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.detection        # voll (alle Kanaele, langsam)
    conda run -n cedalion python -m drift_glm.analysis.detection test    # Mini-Test (Kanal-Subset)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import cedalion
import cedalion.data
import cedalion.models.glm as glm
from cedalion import units

from drift_glm.core import pipeline as pl
from drift_glm.analysis.sweep import drift_dm
from drift_glm import paths

RESULTS = paths.RESULTS
OUT = paths.FIGURES

FAMILIES = ["none", "poly:1", "poly:3", "dct:0.01", "legendre:3", "bspline:5"]
WINDOW = 180.0
SEEDS = [0, 1]
ALPHA = 0.05        # FDR-Niveau q
ACTIVE_THR = 0.1    # Ground-Truth-aktiv: |beta_true| > 10% des Peaks


def _progress(done, total, t0, last):
    """Live-Fortschrittsdatei (nach jedem Fit) -> results/logs/detection_progress.txt."""
    el = time.time() - t0
    eta = (el / done) * (total - done) if done else 0.0
    (paths.LOGS / "detection_progress.txt").write_text(
        f"detection: {done}/{total} ({100 * done / total:.0f} %)\n"
        f"verstrichen : {el / 60:5.1f} min\n"
        f"ETA (Rest)  : {eta / 60:5.1f} min\n"
        f"letzte Zelle: {last}\n")


def _hrf_tvalues(res):
    """t-Werte des HRF-Regressors je Kanal als beta/SE (SE aus regressor_variances()).

    Cedalions .sm.p_values ist fuer die rohen Fit-Ergebnisse nicht implementiert; p wird
    zweiseitig aus der Normalverteilung abgeleitet, bei ~1600 Zeitpunkten unkritisch.
    Ungueltige Inferenz (negative oder nichtendliche Varianz, SE = 0) wird NICHT still
    umgewandelt, sondern als NaN zurueckgegeben und in `invalid_inference` gezaehlt.
    """
    beta = res.sm.params.sel(regressor="HRF Stim")
    var = res.sm.regressor_variances().sel(regressor="HRF Stim")
    v = np.asarray(var.values, dtype=float)
    bad = ~np.isfinite(v) | (v <= 0)
    se = np.sqrt(np.where(bad, np.nan, v))
    t = beta / xr.DataArray(se, dims=var.dims, coords=var.coords)
    return t.where(np.isfinite(t))        # (channel, chromo), NaN = ungueltig


def invalid_inference(tvals) -> int:
    """Zahl der Kanaele ohne gueltigen t-Wert (NaN nach `_hrf_tvalues`)."""
    return int((~np.isfinite(np.asarray(tvals.values, dtype=float))).sum())


def _metrics(pvals, truth_active):
    """Detektionsguete; Kanaele mit ungueltigem p (NaN) gelten als nicht detektiert und
    werden separat gezaehlt (`n_invalid`), statt als p = 1 oder p = 0 einzugehen."""
    pvals = np.asarray(pvals, dtype=float)
    valid = np.isfinite(pvals)
    reject = np.zeros(pvals.shape, dtype=bool)
    if valid.any():
        reject[valid], _, _, _ = multipletests(pvals[valid], alpha=ALPHA, method="fdr_bh")
    tp = int((reject & truth_active).sum())
    fp = int((reject & ~truth_active).sum())
    fn = int((~reject & truth_active).sum())
    tn = int((~reject & ~truth_active).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    youden = (sens + spec - 1) if (sens == sens and spec == spec) else float("nan")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, sensitivity=sens, specificity=spec,
                precision=prec, youden_J=youden, n_detected=int(reject.sum()),
                n_invalid=int((~valid).sum()))


def main(mode="full"):
    paths.ensure()
    families = ["poly:3", "none"] if mode == "test" else FAMILIES
    seeds = [0] if mode == "test" else SEEDS
    rec = cedalion.data.get_nn22_resting_state()
    rows = []
    t0 = time.time()
    total = len(seeds) * len(families)
    done = 0
    for seed in seeds:
        P = pl.build(window_s=WINDOW, rec=rec, seed=seed)
        if mode == "test":
            idx = np.unique(np.linspace(0, P.conc_syn.sizes["channel"] - 1, 60, dtype=int))
            ts_all, btm = P.conc_syn.isel(channel=idx), P.beta_true_map.isel(channel=idx)
        else:
            ts_all, btm = P.conc_syn, P.beta_true_map
        peak = abs(P.beta_true["HbO"])
        for fam in families:
            dm_drift, filt = drift_dm(fam, P.conc)
            ts_f = ts_all if filt is None else ts_all.cd.freq_filter(
                filt[0] * units.Hz, filt[1] * units.Hz, 4)
            res = glm.fit(ts_f, P.dm_hrf & dm_drift, noise_model="ar_irls",
                          ar_order=30, max_jobs=-1)
            tvals = _hrf_tvalues(res)                        # (channel, chromo)
            n_bad = invalid_inference(tvals)
            if n_bad:
                print(f"  ! {n_bad} Kanal-t-Werte ungueltig (Varianz <= 0 oder nicht "
                      f"endlich) bei {fam}, seed={seed}", flush=True)
            for c in ("HbO", "HbR"):
                t = np.asarray(tvals.sel(chromo=c).values, float)
                pvals = 2.0 * norm.sf(np.abs(t))             # zweiseitig, grosses df; NaN bleibt NaN
                truth = np.abs(btm.sel(chromo=c).values) > ACTIVE_THR * peak
                m = _metrics(pvals, truth)
                m.update(family=fam, chromo=c, seed=seed,
                         n_truth_active=int(truth.sum()), n_channels=len(pvals))
                rows.append(m)
            done += 1
            print(f"  [{done}/{total}] seed={seed} {fam:12s} fertig", flush=True)
            _progress(done, total, t0, f"{fam} (seed={seed})")

    df = pd.DataFrame(rows)
    agg = (df.groupby(["family", "chromo"])
           .agg(sensitivity=("sensitivity", "mean"),
                specificity=("specificity", "mean"),
                precision=("precision", "mean"),
                youden_J=("youden_J", "mean"),
                n_detected=("n_detected", "mean"),
                n_truth_active=("n_truth_active", "mean"),
                n_invalid=("n_invalid", "sum"))
           .reset_index())
    agg.to_csv(RESULTS / "detection_summary.csv", index=False)
    print("\n=== Detektionsguete nach FDR (q=0.05), Mittel ueber Seeds ===")
    print(agg.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, c in enumerate(["HbO", "HbR"]):
        d = agg[agg.chromo == c].set_index("family").reindex(families)
        x = np.arange(len(families))
        ax[j].bar(x - 0.2, d.sensitivity.values, width=0.4,
                  label="Sensitivität (TPR)", color="C0")
        ax[j].bar(x + 0.2, d.specificity.values, width=0.4,
                  label="Spezifität (TNR)", color="C1")
        ax[j].set_xticks(x); ax[j].set_xticklabels(families, rotation=60, ha="right", fontsize=8)
        ax[j].set_ylim(0, 1.05); ax[j].set_title(f"{c}: Detektion nach FDR (q=0.05)")
        ax[j].grid(axis="y", alpha=0.3); ax[j].legend(fontsize=8)
    fig.suptitle("Aktivierungs-Detektion je Driftfamilie (Signifikanz + FDR vs. Ground Truth)")
    fig.tight_layout(); fig.savefig(OUT / "14_detection.png", dpi=130); plt.close(fig)
    print(f"\n-> {RESULTS/'detection_summary.csv'} | Abb. 14 in {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
