"""Was macht die neue Preprocessing-Kette mit der beta-Rueckgewinnung?

Vergleicht die alte Kette (nur SNR>10, keine Motion Correction) gegen die neue Kette
nach Betreuungsvorgabe, und darin die beiden Stufen der Motion-Correction-Achse. Damit
sind zwei Fragen auf einmal beantwortet:

  1. Muessen die v3-Zahlen wirklich neu gerechnet werden -- also aendert die neue Kette
     die Schaetzung ueberhaupt?
  2. Doppelt sich die Motion Correction mit den Motion-Regressoren in der Designmatrix
     (Betreuungshinweis "evtl keine motion-correction ... sonst gedoppelt -> ueberpruefen")?
     Dafuer wird jede Preprocessing-Variante einmal OHNE und einmal MIT Motion-Regressoren
     gefittet.

Die ALTE Kette wird nicht als Code-Kopie nachgebaut, sondern ueber die Parameter der
neuen ausgedrueckt: `motion_method="none"`, `snr_threshold=10`, `amp_range` so weit, dass
mean_amp nichts verwirft. Das ist nachweislich aequivalent -- ohne Motion Correction gilt
od2int(int2od(amp)) == amp exakt (verifiziert: rel. Fehler 7.7e-16), also ist die
SNR-Maske auf der "korrigierten" Amplitude dieselbe wie auf der Rohamplitude.

WICHTIG (Umbau 2026-08-01): Die synthetische HRF wird inzwischen VOR der Motion
Correction in die OD eingemischt, nicht mehr danach in die Konzentration. Eine fruehere
Fassung dieses Vergleichs kam deshalb zum Ergebnis "tddr+wavelet ist am besten" -- ein
Artefakt: die Korrektur konnte die HRF gar nicht erreichen und durfte nur das Rauschen
putzen. Mit der realistischen Einmischung daempft TDDR die HRF-Amplitude gemessen auf
~70 % (Wavelet: 100 %), was als Unterschaetzung von beta durchschlaegt.

Verglichen wird auf IDENTISCHEN Kanaelen (Schnittmenge aller Varianten), sonst waere der
Unterschied teils nur eine andere Kanalauswahl.

Aufruf:
    conda run -n cedalion python compare_preprocessing.py        # voll (~6 min)
    conda run -n cedalion python compare_preprocessing.py test   # 1 Seed, schneller
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cedalion.data
import cedalion.models.glm as glm

import pipeline as pl
import preprocess as prep
from sweep import motion_dm

RESULTS = Path(__file__).parent / "results"
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


def metrics(bhat, truth):
    """Bias/RMSE/Streuung von beta_hat gegen die per-Kanal-Wahrheit.

    `bhat` hat Dims (seed, channel), `truth` (channel,). Der Bias wird ueber Seeds
    gemittelt (echter Monte-Carlo-Bias), RMSE ueber Seeds UND Kanaele.
    """
    bias = np.nanmedian(np.nanmean(bhat, axis=0) - truth)
    rmse = float(np.sqrt(np.nanmean((bhat - truth[None, :]) ** 2)))
    std = float(np.nanmean(np.nanstd(bhat, axis=0)))
    return float(bias), rmse, std


def main(seeds):
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    rec = cedalion.data.get_nn22_resting_state()

    stage = prep.to_od_stage(rec)

    # Ein Build je Variante/Seed vorab, um die gemeinsame Kanalbasis zu bestimmen.
    # (Die Einmischung passiert jetzt VOR der Motion Correction, deshalb haengt die
    # Vorverarbeitung am Seed und laesst sich nicht mehr einmal vorab berechnen.)
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
