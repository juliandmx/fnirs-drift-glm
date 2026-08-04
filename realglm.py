"""Stufe 2, Kern: GLM je Driftfamilie auf den realen Daten, Gruppenebene.

Zweistufig, wie in der fNIRS-Literatur ueblich:
  1. Ebene (je Datei = Proband x Durchgang): GLM mit HRF-Regressor "Tapping" + Drift.
     Der HRF-Regressor ist auf Peak 1 normiert, damit beta direkt die Peak-Aenderung
     in µM ist -- dieselbe Konvention wie in der Simulation.
  2. Ebene (Gruppe): beta je Proband ueber seine Durchgaenge mitteln, dann Einstichproben-
     t-Test ueber die 25 Probanden je Kanal, danach Benjamini-Hochberg-FDR ueber Kanaele.

DER ENTSCHEIDENDE UNTERSCHIED ZUR SIMULATION: hier gibt es KEINE Ground Truth. Ob eine
Driftfamilie "besser" ist, laesst sich nicht am Fehler gegen die Wahrheit messen. Deshalb
drei ersatzweise Kriterien, die ohne Wahrheit auskommen:

  * **Detektion**  -- wie viele Kanaele ueberstehen die FDR-Korrektur? Mehr ist nicht
    automatisch besser (koennten Falsch-Positive sein), aber im Zusammenspiel mit den
    beiden folgenden Kriterien aussagekraeftig.
  * **Reproduzierbarkeit** -- die Probanden haben 2-3 Durchgaenge. Die Korrelation der
    beta-Karten zwischen den Durchgaengen EINES Probanden misst, wie stabil die Schaetzung
    ist. Das ist das staerkste wahrheitsfreie Kriterium: eine Driftfamilie, die Rauschen
    als Aktivierung modelliert, ist zwischen Durchgaengen inkonsistent.
  * **Plausibilitaet** -- HbO und HbR muessen gegenlaeufig sein. Das Verhaeltnis HbR/HbO
    ueber die aktiven Kanaele sollte um -0.4 liegen; Werte nahe 0 deuten auf systemische
    Kontamination statt neuronaler Antwort.

Konstellationen: nur `baseline` und `global`. Der Datensatz hat weder Short-Separation-
Kanaele (kuerzester Abstand 25.9 mm) noch Bewegungs-Aux -- `short_*` und `motion` sind
hier nicht anwendbar und bleiben Simulationsbefunde.

Aufruf:
    conda run -n cedalion python realglm.py test    # 2 Probanden, 2 Familien (Timing)
    conda run -n cedalion python realglm.py         # voll (Nachtlauf)
"""

from __future__ import annotations

import importlib
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

import cedalion.models.glm as glm
from cedalion import units

import pipeline as pl
import realdata as rd
from sweep import drift_dm

RESULTS = Path(__file__).parent / "results"
HRF_REG = "HRF Tapping"
ALPHA = 0.05          # FDR-Niveau q

FAMILIES = ["none", "poly:1", "poly:2", "poly:3", "poly:5",
            "dct:0.005", "dct:0.01", "dct:0.02",
            "legendre:1", "legendre:3", "bspline:5", "bspline:8",
            "butter:0.01", "bandpass:0.01-0.5"]

#: Familien, die mit AR-IRLS NICHT auswertbar sind -- gemessen, nicht vermutet.
#:
#: Jeder TIEFPASS laesst beta auf ~1e-05 kollabieren (fuenf Groessenordnungen zu klein),
#: waehrend derselbe Datensatz mit OLS normale Werte liefert und ein reiner HOCHPASS
#: (butter:0.01) mit AR-IRLS problemlos laeuft. Ursache ist die Prewhitening-Stufe:
#: AR-IRLS schaetzt ein AR-Modell des Rauschens und wendet dessen Inverse an. Ein
#: Tiefpass bei 0.5 Hz entfernt bei fs=3.906 Hz rund drei Viertel des Spektrums
#: (0.5 Hz = 0.256 x Nyquist); das Residuum hat oberhalb davon praktisch keine Leistung
#: mehr, und der Whitening-Filter muesste dort unendlich verstaerken. Uebrig bleibt
#: numerisches Rauschen -- der HRF-Anteil verschwindet mit.
#:
#: Konsequenz fuer die Arbeit: **Tiefpassfilterung und AR-IRLS schliessen einander aus.**
#: Der Tiefpass aus der Betreuungsvorgabe ist daher nur mit OLS auswertbar. Die
#: betroffenen Zellen werden im Report ausgewiesen und nicht in Ranglisten gemischt.
AR_IRLS_INCOMPATIBLE = ("lowpass:", "bandpass:")


def is_degenerate(family: str, noise_model: str) -> bool:
    """True, wenn diese Kombination nicht auswertbar ist (s. AR_IRLS_INCOMPATIBLE)."""
    return noise_model == "ar_irls" and family.startswith(AR_IRLS_INCOMPATIBLE)
CONSTELLATIONS = ["baseline", "global"]
NOISE_MODELS = ["ar_irls", "ols"]
MOTION_METHOD = "wavelet"      # driftneutral, s. preprocess.DEFAULT_MOTION
#: Prozesse fuer die Parallelisierung ueber Dateien. 8 Kerne verfuegbar, aber jeder
#: Prozess braucht ~0.5 GB -- bei ~4 GB freiem RAM sind 5 die sichere Obergrenze.
N_JOBS = 5


def _progress(done, total, t0, last):
    el = time.time() - t0
    eta = (el / done) * (total - done) if done else 0.0
    (RESULTS / "realglm_progress.txt").write_text(
        f"realglm: {done}/{total} ({100 * done / total:.0f} %)\n"
        f"verstrichen : {el / 60:5.1f} min\n"
        f"ETA (Rest)  : {eta / 60:5.1f} min\n"
        f"letzte Zelle: {last}\n")


def first_level(conc, stim, family, constellation, noise_model, ar_order=30,
                max_jobs=-1):
    """GLM einer Datei -> beta des Tapping-Regressors, Dims (channel, chromo).

    `max_jobs=1`, wenn schon auf Datei-Ebene parallelisiert wird (s. `main`) -- sonst
    ueberzeichnen sich die Prozesse gegenseitig.
    """
    basis = glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=0 * units.s)
    dm_hrf = glm.design_matrix.hrf_regressors(conc, stim, basis)
    hrf_names = [r for r in dm_hrf.common.regressor.values if str(r).startswith("HRF")]
    dm_hrf, _ = pl._normalize_hrf_to_unit_peak(dm_hrf, hrf_names)

    dm_drift, filt = drift_dm(family, conc)
    ts = conc
    if filt is not None:
        # Filter-Alternative im Konzentrationsraum (Betreuungsvorgabe)
        ts = conc.cd.freq_filter(filt[0] * units.Hz, filt[1] * units.Hz, 4)

    dm = dm_hrf & dm_drift
    if constellation == "global":
        dm = dm & glm.design_matrix.global_mean_regressor(ts)

    res = glm.fit(ts, dm, noise_model=noise_model, ar_order=ar_order,
                  max_jobs=max_jobs)
    return res.sm.params.sel(regressor=hrf_names[0])


def group_test(beta_by_subject: dict[str, xr.DataArray]):
    """Zweite Ebene: Einstichproben-t-Test ueber Probanden + FDR ueber Kanaele.

    Rueckgabe (t, p_fdr, reject, mean_beta), je Dims (channel, chromo).
    """
    subs = sorted(beta_by_subject)
    stack = xr.concat([beta_by_subject[s] for s in subs], dim="subject")
    stack = stack.assign_coords(subject=subs)
    mean = stack.mean("subject")
    t, p = stats.ttest_1samp(stack.values, popmean=0.0, axis=0, nan_policy="omit")
    t = xr.DataArray(np.asarray(t), dims=mean.dims, coords=mean.coords)
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    ok = np.isfinite(flat)
    rej = np.zeros_like(flat, dtype=bool)
    p_adj = np.full_like(flat, np.nan)
    if ok.any():
        rej[ok], p_adj[ok], _, _ = multipletests(flat[ok], alpha=ALPHA,
                                                 method="fdr_bh")
    shape = p.shape
    return (t,
            xr.DataArray(p_adj.reshape(shape), dims=mean.dims, coords=mean.coords),
            xr.DataArray(rej.reshape(shape), dims=mean.dims, coords=mean.coords),
            mean)


def split_run_reliability(beta_by_run: dict[str, dict[str, xr.DataArray]], chromo="HbO"):
    """Reproduzierbarkeit: Korrelation der beta-Karten zwischen Durchgaengen.

    Je Proband mit mindestens zwei Durchgaengen die Pearson-Korrelation der
    Kanal-beta-Karten aller Durchgangspaare; anschliessend Median ueber Probanden.
    Wahrheitsfreies Guetemass -- eine Familie, die Rauschen als Aktivierung modelliert,
    ist zwischen Durchgaengen inkonsistent.
    """
    rs = []
    for runs in beta_by_run.values():
        keys = sorted(runs)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = np.asarray(runs[keys[i]].sel(chromo=chromo).values, float)
                b = np.asarray(runs[keys[j]].sel(chromo=chromo).values, float)
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() > 3 and a[m].std() > 0 and b[m].std() > 0:
                    rs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    return (float(np.median(rs)), len(rs)) if rs else (float("nan"), 0)


def main(mode="full"):
    RESULTS.mkdir(exist_ok=True)
    files = rd.find_files()
    if not files:
        print(f"Keine Daten in {rd.DATA_DIR}"); return

    families = FAMILIES if mode != "test" else ["poly:3", "none"]
    noise_models = NOISE_MODELS if mode != "test" else ["ols"]
    if mode == "test":
        subs = sorted({rd.subject_of(f) for f in files})[:2]
        files = [f for f in files if rd.subject_of(f) in subs]

    print(f"realglm [{mode}]: {len(files)} Dateien, {len(families)} Familien, "
          f"{len(CONSTELLATIONS)} Konstellationen, {len(noise_models)} Rauschmodelle",
          flush=True)

    # Vorverarbeitung einmal je Datei (unabhaengig von Familie/Konstellation/Modell)
    t0 = time.time()
    prepped = {}
    for f in files:
        rec = rd.load(f)
        P, amp_range = rd.preprocess_recording(rec, motion_method=MOTION_METHOD)
        prepped[f] = (P.conc, rd.stim_df(rec, rd.subject_of(f), "tapping"))
    print(f"Vorverarbeitung: {len(files)} Dateien in {time.time() - t0:.0f}s", flush=True)

    total = len(files) * len(families) * len(CONSTELLATIONS) * len(noise_models)
    done, recs = 0, []
    # Parallelisierung auf DATEI-Ebene. Cedalions glm.fit parallelisiert intern kaum:
    # alle 48 Kanaele teilen dieselbe Designmatrix und landen in EINER Rechengruppe,
    # gemessen ~1.5 von 8 Kernen. Die Dateien sind dagegen voneinander unabhaengig.
    # Innen daher max_jobs=1, sonst ueberzeichnen sich die Prozesse.
    #
    # Zwei Fallstricke, beide gemessen:
    #   * backend="threading" bringt NICHTS (gemessen Faktor 1.0-1.1): AR-IRLS ist
    #     GIL-gebunden,
    #     es sind Python-Schleifen in statsmodels, keine BLAS-Operationen, die den GIL
    #     freigeben wuerden. Es braucht echte Prozesse (loky).
    #   * loky serialisiert Funktionen aus __main__ per WERT (cloudpickle) und scheitert
    #     dabei an cedalion-Objekten. Deshalb wird der Worker ueber importlib aus dem
    #     MODUL geholt -- so wird er per Referenz gepickelt, und die Kinder importieren
    #     ihn selbst.
    worker = importlib.import_module(__spec__.name if __spec__ else "realglm").first_level
    n_jobs = min(N_JOBS, len(files))
    print(f"Parallel ueber Dateien: {n_jobs} Prozesse (loky)", flush=True)
    for fam in families:
        for con in CONSTELLATIONS:
            for nm in noise_models:
                tc = time.time()
                betas = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(worker)(prepped[f][0], prepped[f][1], fam, con, nm,
                                         30, 1)
                    for f in files)
                by_sub_run = defaultdict(dict)
                for f, b in zip(files, betas):
                    by_sub_run[rd.subject_of(f)][f.stem] = b
                done += len(files)
                dt = time.time() - tc
                print(f"  [{done}/{total}] {fam:16s} {con:8s} {nm:7s} "
                      f"{len(files)} Dateien in {dt:5.1f}s ({dt / len(files):4.1f}s/Fit)",
                      flush=True)
                _progress(done, total, t0, f"{fam} {con} {nm}")

                # Proband = Mittel ueber seine Durchgaenge
                by_sub = {s: xr.concat(list(r.values()), dim="run").mean("run")
                          for s, r in by_sub_run.items()}
                t, p_adj, rej, mean = group_test(by_sub)
                rel, n_pairs = split_run_reliability(by_sub_run)

                for ch in ("HbO", "HbR"):
                    m = mean.sel(chromo=ch).values
                    r = rej.sel(chromo=ch).values
                    row = dict(family=fam, constellation=con, noise_model=nm, chromo=ch,
                               n_subjects=len(by_sub), n_channels=int(m.size),
                               n_significant=int(r.sum()),
                               beta_mean=float(np.nanmean(m)),
                               beta_max=float(np.nanmax(np.abs(m))),
                               t_max=float(np.nanmax(np.abs(t.sel(chromo=ch).values))),
                               reliability_r=rel, n_run_pairs=n_pairs)
                    recs.append(row)

                # Plausibilitaet: HbR/HbO ueber die signifikanten HbO-Kanaele
                mo = mean.sel(chromo="HbO").values
                mr = mean.sel(chromo="HbR").values
                sig = rej.sel(chromo="HbO").values
                sel = sig if sig.any() else np.abs(mo) > np.nanpercentile(np.abs(mo), 75)
                ratio = float(np.nanmedian(mr[sel] / mo[sel])) if sel.any() else np.nan
                corr = float(np.corrcoef(mo, mr)[0, 1]) if np.isfinite(mo).all() else np.nan
                for row in recs[-2:]:
                    row["hbr_hbo_ratio"] = ratio
                    row["hbo_hbr_corr"] = corr

    df = pd.DataFrame(recs)
    out = RESULTS / ("realglm_summary_test.csv" if mode == "test"
                     else "realglm_summary.csv")
    df.to_csv(out, index=False)
    print(f"\n[OK] {total} Fits in {(time.time() - t0) / 60:.1f} min -> {out}")

    d = df[(df.chromo == "HbO") & (df.noise_model == noise_models[0])]
    print("\nHbO, Rauschmodell "
          f"{noise_models[0]} — signifikante Kanaele / Reproduzierbarkeit:")
    print(d.pivot_table(index="family", columns="constellation",
                        values=["n_significant", "reliability_r"])
           .to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
