"""Visualisierung der Demo-Pipeline (siehe pipeline.py / demo_recovery.py).

Erzeugt PNGs in figures/:
  1) Designmatrix (normierter HRF-Regressor + Polynom-Drift)
  2) Ein Kanal: Ruhesignal, eingespeiste Aktivierung, rueckgewonnene HRF
  3) rueckgewonnenes vs. wahres beta pro Kanal (Blob-Gradient), OLS
  4) Scalp-Plot der Abweichung (beta_hat - Ground Truth) pro Kanal, farbkodiert
  5) Scalp-Plot der RELATIVEN Abweichung: rel. beta-Peak-Fehler + rel. Formfehler

Aufruf:
    conda run -n cedalion python -m drift_glm.reports.demo_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cedalion
import cedalion.vis.anatomy
import cedalion.models.glm as glm
from drift_glm.core import pipeline as pl
from drift_glm import paths

OUTDIR = paths.FIGURES


def _sym_lim(values, pct=98.0):
    """Robuste, um 0 symmetrische Farbgrenzen (unempfindlich ggü. Ausreissern)."""
    r = float(np.nanpercentile(np.abs(np.asarray(values, dtype=float)), pct))
    return -r, r


def main():
    paths.ensure()
    P = pl.build()
    main_hrf = P.hrf_names[0]

    print("Fit OLS (alle Kanaele) ...")
    betas_ols = glm.fit(P.conc_syn, P.dm_full, noise_model="ols", max_jobs=-1).sm.params

    # anschaulicher AKTIVER Kanal (nahe Blob-Zentrum): dort medianer HbO-Fehler
    bt_hbo = P.beta_true_map.sel(chromo="HbO")
    active = np.abs(bt_hbo) > 0.3 * abs(P.beta_true["HbO"])
    err = np.abs(betas_ols.sel(regressor=main_hrf, chromo="HbO")
                 - bt_hbo).where(active, drop=True)
    demo_ch = str(err.channel.values[int(err.argsort()[len(err) // 2])])

    # ---- Abb. 1: Designmatrix ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    dmc = P.dm_full.common.sel(chromo="HbO")
    t = dmc.time.values
    axes[0].plot(t, dmc.sel(regressor=main_hrf).values, color="C3")
    axes[0].set_title(f"HRF-Regressor ('{main_hrf}', Gamma-Basis, auf Peak=1 normiert)")
    axes[0].set_ylabel("a.u.")
    for r in dmc.regressor.values:
        if str(r).startswith("Drift"):
            axes[1].plot(t, dmc.sel(regressor=r).values, label=str(r))
    axes[1].set_title("Polynom-Drift-Regressoren")
    axes[1].set_xlabel("Zeit [s]"); axes[1].set_ylabel("a.u."); axes[1].legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "01_designmatrix.png", dpi=130); plt.close(fig)

    # ---- Abb. 2: Ein Kanal, AR-IRLS-Fit ----
    print(f"Fit AR-IRLS (Demo-Kanal {demo_ch}) ...")
    betas_hat = glm.fit(P.conc_syn.sel(channel=[demo_ch]), P.dm_full,
                        noise_model="ar_irls", max_jobs=1).sm.params
    pred = glm.predict(P.conc_syn.sel(channel=[demo_ch]), betas_hat, P.dm_full)
    # Nur die rueckgewonnene HRF-Komponente (ohne Drift/Offset), damit sie mit der
    # eingespeisten Ground-Truth-HRF auf derselben Nulllinie vergleichbar ist.
    # dm_hrf enthaelt exakt den peak-normierten Regressor, der injiziert wurde.
    hrf_mask = betas_hat.regressor.str.startswith("HRF")
    pred_hrf = glm.predict(P.conc_syn.sel(channel=[demo_ch]),
                           betas_hat.sel(regressor=hrf_mask), P.dm_hrf)
    t = P.conc.time.values
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, c in enumerate(["HbO", "HbR"]):
        syn = P.conc_syn.sel(channel=demo_ch, chromo=c).values
        act = P.activation.sel(channel=demo_ch, chromo=c).values
        pr = pred.sel(channel=demo_ch, chromo=c).values
        prh = pred_hrf.sel(channel=demo_ch, chromo=c).values
        ax[i].plot(t, syn, color="0.55", lw=0.7, label="Ruhe + eingespeiste HRF")
        ax[i].plot(t, pr, color="0.75", lw=1.0, ls=":",
                   label="GLM-Fit gesamt (HRF+Drift)")
        ax[i].plot(t, act, color="C3", lw=1.8,
                   label="eingespeiste HRF (Ground Truth)")
        ax[i].plot(t, prh, color="C0", lw=1.5,
                   label="rueckgewonnene HRF (AR-IRLS)")
        for _, row in P.stim_df.iterrows():
            ax[i].axvspan(row["onset"], row["onset"] + row["duration"], color="C2", alpha=0.08)
        bt = float(P.beta_true_map.sel(channel=demo_ch, chromo=c))
        bh = float(betas_hat.sel(channel=demo_ch, regressor=main_hrf, chromo=c))
        ax[i].set_title(f"{c} | Kanal {demo_ch} | Peak_true={bt:+.3f}  beta_hat={bh:+.3f} µM")
        ax[i].set_ylabel(f"Δ{c} [µM]"); ax[i].legend(loc="upper right", fontsize=8)
    ax[1].set_xlabel("Zeit [s]")
    fig.tight_layout(); fig.savefig(OUTDIR / "02_kanal_fit.png", dpi=130); plt.close(fig)

    # ---- Abb. 3: rueckgewonnenes vs. wahres beta pro Kanal (Blob-Gradient) ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for j, c in enumerate(["HbO", "HbR"]):
        true = P.beta_true_map.sel(chromo=c).values
        est = betas_ols.sel(regressor=main_hrf, chromo=c).values
        ax[j].scatter(true, est, s=8, alpha=0.4, color="C0")
        lim = [min(true.min(), est.min()), max(true.max(), est.max())]
        ax[j].plot(lim, lim, "k--", lw=1, label="Identität (β̂ = GT)")
        ax[j].set_title(f"{c}: β̂ vs. Ground Truth über {len(true)} Kanäle (OLS)")
        ax[j].set_xlabel("wahres β [µM]"); ax[j].set_ylabel("β̂ [µM]")
        ax[j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTDIR / "03_beta_recovery.png", dpi=130); plt.close(fig)

    # ---- Abb. 4: Scalp-Plot der Abweichung (beta_hat - Ground Truth) pro Kanal ----
    # Analog zum Data-Augmentation-Tutorial (examples/tutorial/7) bzw. GLM-Notebook 32:
    # jeder Kanal auf der 2D-Kopfprojektion, Abweichung diverging um 0 farbkodiert.
    # Hier OLS ueber alle Kanaele (schnell); AR-IRLS-Betas liessen sich 1:1 einsetzen.
    print("Erzeuge Scalp-Plot der Abweichung ...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    for j, c in enumerate(["HbO", "HbR"]):
        dev = betas_ols.sel(regressor=main_hrf, chromo=c) - P.beta_true_map.sel(chromo=c)
        vlo, vhi = _sym_lim(dev.values)   # robuste, um 0 symmetrische Skala (Ausreisser)
        cedalion.vis.anatomy.scalp_plot(
            P.conc, P.geo3d, dev, ax[j],
            vmin=vlo, vmax=vhi, cmap="RdBu_r",
            min_dist=1.5 * cedalion.units.cm,
            title=f"{c}: Abweichung β̂ − Ground Truth (OLS)",
            cb_label="β̂ − GT [µM]",
        )
    fig.tight_layout(); fig.savefig(OUTDIR / "04_scalp_abweichung.png", dpi=130); plt.close(fig)

    # ---- Abb. 5: Abweichung RELATIV zur HRF, pro Kanal (farbkodiert) ----
    # Zwei gewuenschte Lesarten:
    #   (a) rel. beta-Peak-Fehler: (beta_hat - GT)/GT  -> Fehler der Peak-Amplitude
    #   (b) rel. Formfehler:       RMSE_t(rueckgewonnene HRF - GT-HRF) / HRF-Peak
    # HINWEIS: In diesem PoC nutzen Injektion UND Fit dieselbe (einzelne) Gamma-Basis,
    # d.h. die rueckgewonnene HRF hat per Konstruktion dieselbe FORM wie die Ground Truth
    # und unterscheidet sich nur in der Amplitude. Dadurch ist (b) hier ~proportional zu
    # (a) (Faktor = RMS des normierten Regressors). Unabhaengig informativ wird (b) erst
    # mit einer FLEXIBLEN Recovery-Basis (z.B. GaussianKernels/GammaDeriv) oder auf realen
    # Daten -- die Metrik-/Plot-Infrastruktur steht damit aber bereit.
    print("Erzeuge Scalp-Plots der relativen Abweichung ...")
    hrf_mask_all = betas_ols.regressor.str.startswith("HRF")
    recovered_hrf = glm.predict(P.conc_syn, betas_ols.sel(regressor=hrf_mask_all), P.dm_hrf)
    # relative Fehler nur auf AKTIVEN Kanaelen (sonst Division durch GT~0 am Blob-Rand)
    active = np.abs(P.beta_true_map.sel(chromo="HbO")) > 0.1 * abs(P.beta_true["HbO"])
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for j, c in enumerate(["HbO", "HbR"]):
        bhat = betas_ols.sel(regressor=main_hrf, chromo=c)
        gt = P.beta_true_map.sel(chromo=c)
        rel_beta = ((bhat - gt) / gt).where(active)               # (a) nur aktive Kanaele
        diff = recovered_hrf.sel(chromo=c) - P.activation.sel(chromo=c)
        rel_shape = (np.sqrt((diff ** 2).mean("time")) / np.abs(gt)).where(active)  # (b)
        rb, rs = rel_beta * 100.0, rel_shape * 100.0
        # Feste Farbgrenzen bei +/-100 % (Betreuungsvorgabe "bei rel scalp plot: limit auf
        # 100% setzen") statt perzentilbasiert. Zwei Gruende: 100 % ist eine
        # interpretierbare Marke -- der Fehler ist so gross wie die gesuchte Groesse
        # selbst --, und feste Grenzen machen die Teilbilder untereinander und ueber Laeufe
        # hinweg vergleichbar. Werte darueber werden gesaettigt dargestellt und im Titel
        # ausgewiesen, damit die Saettigung nicht unbemerkt bleibt.
        n_clip = int((np.abs(rb.values) > 100.0).sum())
        clip_note = f"  ({n_clip} Kanal/Kanaele > 100 %)" if n_clip else ""
        cedalion.vis.anatomy.scalp_plot(
            P.conc, P.geo3d, rb, ax[0, j],
            vmin=-100.0, vmax=100.0, cmap="RdBu_r", min_dist=1.5 * cedalion.units.cm,
            title=f"{c}: rel. β-Peak-Fehler{clip_note}", cb_label="(β̂−GT)/GT [%]",
        )
        cedalion.vis.anatomy.scalp_plot(
            P.conc, P.geo3d, rs, ax[1, j],
            vmin=0.0, vmax=100.0, cmap="YlOrRd", min_dist=1.5 * cedalion.units.cm,
            title=f"{c}: rel. Formfehler", cb_label="RMSE_t(HRF)/Peak [%]",
        )
        print(f"  {c}: median |rel. β-Fehler| = "
              f"{float(np.abs(rel_beta).median()) * 100:.1f} %"
              f"  | median rel. Formfehler = {float(rel_shape.median()) * 100:.1f} %")
    fig.tight_layout(); fig.savefig(OUTDIR / "05_scalp_rel_abweichung.png", dpi=130); plt.close(fig)

    print(f"\nFertig. Demo-Kanal: {demo_ch}  | Abbildungen in {OUTDIR}")
    for f in sorted(OUTDIR.glob("*.png")):
        print("  -", f.name)


if __name__ == "__main__":
    main()
