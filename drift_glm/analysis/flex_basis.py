"""Form der geschaetzten HRF je Driftfamilie, gefittet mit flexibler Basis.

Injiziert wird eine feste Gamma-HRF (Ground Truth), gefittet mit GaussianKernels, deren
Form frei ist; so wird eine Verzerrung der HRF-Form durch die Drift-Modellierung
unabhaengig von der Amplitude messbar. Metriken je aktiver Kanal (Median ueber Kanaele
und Seeds): Form-Treue (Pearson r zwischen rueckgewonnener und injizierter HRF,
skaleninvariant) und relativer Amplitudenfehler am Ort des wahren Peaks. Fenster 180 s,
Konstellation baseline. Ausgabe: results/flex_basis_summary.csv, Abb. 12/13 in figures/.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.flex_basis        # voll
    conda run -n cedalion python -m drift_glm.analysis.flex_basis test    # Mini-Test (Code-Pfade)
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

import cedalion
import cedalion.data
import cedalion.models.glm as glm
from cedalion import units

from drift_glm.core import pipeline as pl
from drift_glm.analysis.sweep import drift_dm   # Driftfamilien wiederverwenden
from drift_glm import paths

RESULTS = paths.RESULTS
OUT = paths.FIGURES

FAMILIES = ["none", "poly:1", "poly:3", "poly:5", "dct:0.01",
            "legendre:3", "bspline:5", "butter:0.01"]
WINDOW = 180.0
SEEDS = [0, 1, 2, 3]
NCH = 20


def _config(mode):
    if mode == "test":
        return ["poly:3", "none"], [0], 8
    return FAMILIES, SEEDS, NCH


def _progress(done, total, t0, last):
    """Live-Fortschrittsdatei (nach jedem Fit) -> results/logs/flex_basis_progress.txt."""
    el = time.time() - t0
    eta = (el / done) * (total - done) if done else 0.0
    (paths.LOGS / "flex_basis_progress.txt").write_text(
        f"flex_basis: {done}/{total} ({100 * done / total:.0f} %)\n"
        f"verstrichen : {el / 60:5.1f} min\n"
        f"ETA (Rest)  : {eta / 60:5.1f} min\n"
        f"letzte Zelle: {last}\n")


def main(mode="full"):
    paths.ensure()
    families, seeds, nch = _config(mode)
    rec = cedalion.data.get_nn22_resting_state()
    # Flexible HRF-Basis: Gaussfunktionen alle 2 s ueber 0..20 s nach Stimulus-Onset.
    flex = glm.GaussianKernels(t_pre=0 * units.s, t_post=20 * units.s,
                               t_delta=2 * units.s, t_std=1.5 * units.s)
    rows = []
    shape_curves = {}   # family -> (t, rec_curve, true_curve) fuer eine Beispielspur
    t0 = time.time()
    total = len(seeds) * len(families)
    done = 0

    for seed in seeds:
        P = pl.build(window_s=WINDOW, rec=rec, seed=seed)
        w = np.abs(P.beta_true_map.sel(chromo="HbO").values)
        idx = np.sort(np.argsort(-w)[:nch])
        ts = P.conc_syn.isel(channel=idx)
        act = P.activation.isel(channel=idx)                    # injizierte HRF (Ground Truth)
        dm_flex_hrf = glm.design_matrix.hrf_regressors(P.conc, P.stim_df, flex)
        tvec = P.conc.time.values
        for fam in families:
            dm_drift, filt = drift_dm(fam, P.conc)
            ts_f = ts if filt is None else ts.cd.freq_filter(
                filt[0] * units.Hz, filt[1] * units.Hz, 4)
            betas = glm.fit(ts_f, dm_flex_hrf & dm_drift, noise_model="ar_irls",
                            ar_order=30, max_jobs=-1).sm.params
            hrf_mask = betas.regressor.str.startswith("HRF")
            rec_hrf = glm.predict(ts_f, betas.sel(regressor=hrf_mask), dm_flex_hrf)
            for c in ("HbO", "HbR"):
                r = rec_hrf.sel(chromo=c)
                a = act.sel(chromo=c)
                for ch in r.channel.values:
                    rc = np.asarray(r.sel(channel=ch).values, float)
                    ac = np.asarray(a.sel(channel=ch).values, float)
                    if np.std(ac) < 1e-9 or np.std(rc) < 1e-12:
                        continue
                    corr = float(np.corrcoef(rc, ac)[0, 1])
                    ipk = int(np.argmax(np.abs(ac)))            # Ort des wahren Peaks
                    rel_amp = float((rc[ipk] - ac[ipk]) / ac[ipk])
                    rows.append(dict(family=fam, chromo=c, seed=seed, channel=str(ch),
                                     shape_corr=corr, amp_rel_err=rel_amp))
            if seed == seeds[0]:                                # Beispielspur (HbO)
                ch0 = rec_hrf.channel.values[0]
                shape_curves[fam] = (
                    tvec,
                    np.asarray(rec_hrf.sel(chromo="HbO", channel=ch0).values, float),
                    np.asarray(act.sel(chromo="HbO", channel=ch0).values, float))
            done += 1
            print(f"  [{done}/{total}] seed={seed} {fam:12s} fertig", flush=True)
            _progress(done, total, t0, f"{fam} (seed={seed})")

    df = pd.DataFrame(rows)
    agg = (df.groupby(["family", "chromo"])
           .agg(shape_corr_med=("shape_corr", "median"),
                amp_rel_err_med=("amp_rel_err", "median"),
                n=("shape_corr", "size"))
           .reset_index())
    agg.to_csv(RESULTS / "flex_basis_summary.csv", index=False)
    print("=== Flexible-Basis-Analyse (Median ueber aktive Kanaele/Seeds) ===")
    print(agg.to_string(index=False))

    # ---- Abb. 12: Form-Treue je Familie (HbO) ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, c in enumerate(["HbO", "HbR"]):
        d = agg[agg.chromo == c].set_index("family").reindex(families)
        ax[j].bar(range(len(families)), d.shape_corr_med.values, color="C0")
        ax[j].set_xticks(range(len(families)))
        ax[j].set_xticklabels(families, rotation=60, ha="right", fontsize=8)
        ax[j].set_ylim(0, 1); ax[j].axhline(1.0, color="k", ls=":", lw=1)
        ax[j].set_title(f"{c}: shape fidelity (Pearson r, recovered vs. true HRF)")
        ax[j].set_ylabel("median r"); ax[j].grid(axis="y", alpha=0.3)
    fig.suptitle("Shape fidelity of the recovered HRF per drift family (flexible basis, Gaussian kernels)")
    fig.tight_layout(); fig.savefig(OUT / "12_flex_shape_corr.png", dpi=130); plt.close(fig)

    # ---- Abb. 13: Beispiel-Formspuren rueckgewonnen vs injiziert ----
    fig, ax = plt.subplots(figsize=(12, 5))
    fam0 = families[0]
    t0, _, true0 = shape_curves[fam0]
    ax.plot(t0, true0, color="k", lw=2.5, label="injected HRF (ground truth)")
    for fam in families:
        t, rc, _ = shape_curves[fam]
        ax.plot(t, rc, lw=1.1, alpha=0.8, label=f"recovered ({fam})")
    ax.set_xlabel("time [s]"); ax.set_ylabel("Δ HbO [µM]")
    ax.set_title("Recovered HRF (flexible basis) per drift family vs. injected HRF "
                 "(one example channel)")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "13_flex_shape_curves.png", dpi=130); plt.close(fig)
    print(f"\n-> {RESULTS/'flex_basis_summary.csv'}  | Abb. 12/13 in {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
