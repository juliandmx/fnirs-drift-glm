"""Einfluss der Vorverarbeitungskette auf die beta-Rueckgewinnung (Abb. 11).

Vergleicht die alte Kette (SNR > 10, keine Motion Correction) mit der neuen Kette und
darin die Stufen der Motion-Correction-Achse; jede Variante wird einmal ohne und einmal
mit Motion-Regressoren gefittet, um eine Doppelung mit der Korrektur zu pruefen. Die alte
Kette wird ueber die Parameter der neuen ausgedrueckt (`motion_method="none"`,
`snr_threshold=10`, offenes `amp_range`); ohne Motion Correction ist od2int(int2od(amp))
exakt amp, die SNR-Maske also dieselbe. Die synthetische HRF wird vor der Motion
Correction in die OD eingemischt, TDDR daempft sie dadurch messbar (~70 %, Wavelet 100 %).
Verglichen wird auf der Kanal-Schnittmenge aller Varianten.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.compare_preprocessing        # voll (~6 min)
    conda run -n cedalion python -m drift_glm.analysis.compare_preprocessing test   # 1 Seed, schneller
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cedalion.data
import cedalion.models.glm as glm

from drift_glm.core import pipeline as pl
from drift_glm.core import preprocess as prep
from drift_glm.analysis.sweep import motion_dm
from drift_glm import paths

RESULTS = paths.RESULTS
HRF_REG = "HRF Stim"
WINDOW_S = 180.0
N_CHANNELS = 20

# amp_range, das mean_amp faktisch abschaltet (alles ausser exakt 0 bleibt drin)
NO_AMP_GATE = (0.0, 1e9)

VARIANTS = {
    "alt: snr=10, ohne MC": dict(motion_method="none", snr_threshold=10.0,
                                 amp_range=NO_AMP_GATE),
    "neu: ohne MC":         dict(motion_method="none"),
    "neu: wavelet":         dict(motion_method="wavelet"),
    "neu: tddr+wavelet":    dict(motion_method="tddr+wavelet"),
}


RETENTION_METHODS = ("none", "wavelet", "tddr", "tddr+wavelet")


def hrf_retention(stage, methods=RETENTION_METHODS, seeds=(0, 1)):
    """Anteil der eingemischten HRF-Amplitude, der die Bewegungskorrektur ueberlebt.

    Braucht keinen GLM-Fit: die Aktivierung ist bekannt, gemessen wird am Ort des wahren
    Maximums je Chromophor. 1.00 = unveraendert; ohne Korrektur ist der Wert exakt 1, weil
    Konzentration -> OD -> Konzentration verlustfrei ist.
    """
    recs = []
    for m in methods:
        for seed in seeds:
            t = time.time()
            P = pl.build(window_s=WINDOW_S, stage=stage, motion_method=m, seed=seed)
            got = (P.conc_syn - P.conc).transpose(*P.activation.dims)
            row = dict(motion_method=m, seed=seed)
            for ch in ("HbO", "HbR"):
                w = np.asarray(P.activation.sel(chromo=ch).values, float)
                g = np.asarray(got.sel(chromo=ch).values, float)
                j = np.unravel_index(np.argmax(np.abs(w)), w.shape)
                row[ch] = float(g[j] / w[j])
            recs.append(row)
            print(f"  Erhalt {m:14s} seed={seed}: HbO {row['HbO']:.3f}  "
                  f"HbR {row['HbR']:.3f}  ({time.time() - t:5.1f}s)", flush=True)
    return pd.DataFrame(recs)


def figure(df, ret):
    """Abb. 11: Erhalt der HRF-Amplitude je Korrektur und Bias der Schaetzung je Variante."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # links: Erhalt der eingemischten Amplitude
    ax = axes[0]
    g = ret.groupby("motion_method")[["HbO", "HbR"]].mean().reindex(
        [m for m in RETENTION_METHODS if m in set(ret.motion_method)])
    x = np.arange(len(g))
    for k, ch in enumerate(["HbO", "HbR"]):
        ax.bar(x + (k - 0.5) * 0.38, g[ch].values * 100, width=0.36, label=ch,
               color="#c44e52" if k == 0 else "#4c72b0")
    ax.axhline(100, ls="--", color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=20, ha="right")
    ax.set_ylabel("erhaltene HRF-Amplitude [%]")
    ax.set_title("Erhalt der eingemischten HRF-Amplitude je Bewegungskorrektur\n"
                 "(am Ort des wahren Maximums)")
    ax.grid(axis="y", alpha=0.3); ax.legend()

    # rechts: Folge fuer die Schaetzung (Bias, aus dem Vergleichslauf)
    ax = axes[1]
    order = [v for v in VARIANTS if v in set(df.variante)]
    x = np.arange(len(order))
    for k, ch in enumerate(["HbO", "HbR"]):
        d = df[(df.chromo == ch) & (df.regressoren == "baseline")]
        v = [float(d[d.variante == o].bias.iloc[0]) if len(d[d.variante == o]) else np.nan
             for o in order]
        ax.bar(x + (k - 0.5) * 0.38, v, width=0.36, label=ch,
               color="#c44e52" if k == 0 else "#4c72b0")
        truth = float(d.beta_true_med.iloc[0]) if len(d) else np.nan
        ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Bias [µM]")
    ax.set_title("Bias der beta-Schätzung je Vorverarbeitungsvariante\n"
                 "(Konstellation baseline)")
    ax.grid(axis="y", alpha=0.3); ax.legend()

    fig.suptitle("Bewegungskorrektur: Erhalt der HRF-Amplitude und Bias der Schätzung")
    fig.tight_layout()
    fig.savefig(paths.FIGURES / "11_preprocessing_effect.png", dpi=130)
    plt.close(fig)


def metrics(bhat, truth):
    """Bias/RMSE/Streuung von beta_hat gegen die per-Kanal-Wahrheit.

    `bhat` hat Dims (seed, channel), `truth` (channel,). Der Bias wird ueber Seeds
    gemittelt (Monte-Carlo-Bias), RMSE ueber Seeds und Kanaele.
    """
    bias = np.nanmedian(np.nanmean(bhat, axis=0) - truth)
    rmse = float(np.sqrt(np.nanmean((bhat - truth[None, :]) ** 2)))
    std = float(np.nanmean(np.nanstd(bhat, axis=0)))
    return float(bias), rmse, std


def main(seeds):
    paths.ensure()
    t0 = time.time()
    rec = cedalion.data.get_nn22_resting_state()

    stage = prep.to_od_stage(rec)

    # Ein Build je Variante/Seed vorab, um die gemeinsame Kanalbasis zu bestimmen. Die
    # Einmischung liegt vor der Motion Correction, die Vorverarbeitung haengt also am Seed.
    print("Varianten aufbauen ...", flush=True)
    builds = {}
    for name, kw in VARIANTS.items():
        for seed in seeds:
            t = time.time()
            builds[(name, seed)] = pl.build(window_s=WINDOW_S, stage=stage,
                                            seed=seed, **kw)
        print(f"  {name:22s} {builds[(name, seeds[0])].conc.sizes['channel']:4d} "
              f"Kanaele ({time.time() - t:5.1f}s/Build)", flush=True)

    # Gemeinsame Kanalbasis: nur so misst der Vergleich die Vorverarbeitung und nicht
    # nebenbei eine andere Kanalauswahl.
    common = set(str(c) for c in next(iter(builds.values())).conc.channel.values)
    for P in builds.values():
        common &= set(str(c) for c in P.conc.channel.values)
    common = sorted(common)
    print(f"gemeinsame Kanalbasis: {len(common)}\n", flush=True)

    recs = []
    for name in VARIANTS:
        for with_motion in (False, True):
            per_seed = {}
            for seed in seeds:
                Pp = builds[(name, seed)]
                ts = Pp.conc_syn.sel(channel=common)
                bt = Pp.beta_true_map.sel(channel=common)
                # Fit-Subset: die N staerkst-aktivierten Kanaele (Blob-Kern)
                idx = np.sort(np.argsort(-np.abs(bt.sel(chromo="HbO").values))[:N_CHANNELS])
                ts = ts.isel(channel=idx)
                dm = Pp.dm_hrf & glm.design_matrix.drift_regressors(Pp.conc, drift_order=3)
                if with_motion:
                    md = motion_dm(Pp.aux, Pp.conc)
                    if md is not None:
                        dm = dm & md
                t = time.time()
                betas = glm.fit(ts, dm, noise_model="ar_irls", ar_order=30,
                                max_jobs=-1).sm.params
                per_seed[seed] = (betas.sel(regressor=HRF_REG), bt.isel(channel=idx))
                print(f"  {name:22s} {'+motion' if with_motion else 'baseline':9s} "
                      f"seed={seed} {time.time() - t:5.1f}s", flush=True)

            for chromo in ("HbO", "HbR"):
                bh = np.stack([per_seed[s][0].sel(chromo=chromo).values for s in seeds])
                tr = per_seed[seeds[0]][1].sel(chromo=chromo).values
                bias, rmse, std = metrics(bh, tr)
                recs.append(dict(variante=name,
                                 regressoren="motion" if with_motion else "baseline",
                                 chromo=chromo, n_seeds=len(seeds),
                                 n_channels=N_CHANNELS,
                                 beta_true_med=float(np.median(tr)),
                                 bias=round(bias, 4), rmse=round(rmse, 4),
                                 std=round(std, 4)))

    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "preprocessing_comparison.csv", index=False)

    # Erhalt der eingemischten Amplitude je Korrekturverfahren; braucht keinen Fit.
    print("\nErhalt der eingemischten HRF-Amplitude:", flush=True)
    ret = hrf_retention(stage, seeds=seeds[:2])
    ret.to_csv(RESULTS / "hrf_retention.csv", index=False)
    print(ret.groupby("motion_method")[["HbO", "HbR"]].mean()
             .to_string(float_format=lambda v: f"{100 * v:.1f} %"))
    figure(df, ret)
    print(f"-> {RESULTS / 'hrf_retention.csv'} + figures/11_preprocessing_effect.png")

    print(f"\n[OK] {time.time() - t0:.0f}s -> {RESULTS / 'preprocessing_comparison.csv'}\n")
    for chromo in ("HbO", "HbR"):
        d = df[df.chromo == chromo]
        piv = d.pivot_table(index="variante", columns="regressoren", values="rmse")
        piv = piv.reindex([v for v in VARIANTS if v in piv.index])
        print(f"RMSE {chromo} [µM] (Wahrheit {d.beta_true_med.iloc[0]:+.3f}):")
        print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
        print()


if __name__ == "__main__":
    main([0] if len(sys.argv) > 1 and sys.argv[1] == "test" else [0, 1, 2])
