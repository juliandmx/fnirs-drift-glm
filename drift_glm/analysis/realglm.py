"""GLM je Driftfamilie auf den realen Fingertapping-Daten (Khan), Gruppenebene.

Erste Ebene je Datei (Proband x Durchgang): GLM mit HRF-Regressor "Tapping" (auf Peak 1
normiert, beta in µM) plus Driftfamilie. Zweite Ebene: beta je Proband ueber die
Durchgaenge mitteln, Einstichproben-t-Test ueber Probanden je Kanal, Benjamini-Hochberg-
FDR ueber Kanaele. Ohne Ground Truth zaehlen drei Kriterien: Zahl signifikanter Kanaele,
Reproduzierbarkeit (Korrelation der beta-Karten zwischen Durchgaengen eines Probanden)
und HbO/HbR-Plausibilitaet (Verhaeltnis HbR/HbO, Korrelation). Konstellationen nur
baseline und global: der Datensatz hat weder Short-Channels (kuerzester Abstand 25.9 mm)
noch Bewegungs-Aux. Ergebnis: results/realglm_summary.csv.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.realglm test    # 2 Probanden, 2 Familien (Timing)
    conda run -n cedalion python -m drift_glm.analysis.realglm         # voll (Nachtlauf)
"""

from __future__ import annotations

import argparse
import importlib
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

from drift_glm.core import pipeline as pl
from drift_glm.core.filtering import apply_filter
from drift_glm.core.provenance import (archive_file, build_run_metadata, write_csv_atomic,
                                       write_metadata_atomic)
from drift_glm.data import realdata as rd
from drift_glm.analysis.sweep import drift_dm
from drift_glm import paths

RESULTS = paths.RESULTS
HRF_REG = "HRF Tapping"
ALPHA = 0.05          # FDR-Niveau q

FAMILIES = ["none", "poly:1", "poly:2", "poly:3", "poly:5",
            "dct:0.005", "dct:0.01", "dct:0.02",
            "legendre:1", "legendre:3", "bspline:5", "bspline:8",
            "butter:0.01", "bandpass:0.01-0.5"]

#: Ergaenzungsarme (Modus `supplement`, danach `merge`): der Tiefpass 0,5 Hz allein, den
#: die Betreuung ausdruecklich testen wollte (Chat 1.8.2026, 11:45), und der konsistente
#: Filter-Kontrollarm `butterxy:0.01`, der Daten UND alle nichtkonstanten Designspalten mit
#: demselben Filter behandelt (core/filtering.py). Der bestehende Arm `butter:0.01`
#: (nur Daten gefiltert, gaengige Praxis) bleibt unveraendert.
SUPPLEMENTAL_FAMILIES = ["lowpass:0.5", "butterxy:0.01"]
SUMMARY_KEY = ["family", "constellation", "noise_model", "chromo"]

#: Tiefpass- und Bandpass-Familien sind mit AR-IRLS nicht auswertbar: ein Tiefpass bei
#: 0.5 Hz (fs = 3.9 Hz) nimmt dem Residuum oberhalb der Grenze praktisch alle Leistung,
#: der Prewhitening-Filter verstaerkt dort unbegrenzt und beta kollabiert auf ~1e-5. Mit
#: OLS und mit reinem Hochpass tritt das nicht auf. Betroffene Zellen werden ausgewiesen
#: und nicht in Ranglisten gemischt.
AR_IRLS_INCOMPATIBLE = ("lowpass:", "bandpass:")


def is_degenerate(family: str, noise_model: str) -> bool:
    """True, wenn diese Kombination nicht auswertbar ist (s. AR_IRLS_INCOMPATIBLE)."""
    return noise_model == "ar_irls" and family.startswith(AR_IRLS_INCOMPATIBLE)
CONSTELLATIONS = ["baseline", "global"]
NOISE_MODELS = ["ar_irls", "ols"]
MOTION_METHOD = "wavelet"      # driftneutral, s. preprocess.DEFAULT_MOTION
#: Prozesse ueber Dateien; ~0.5 GB je Prozess, bei ~4 GB freiem RAM sind 5 die Obergrenze.
N_JOBS = 5


def _progress(done, total, t0, last):
    el = time.time() - t0
    eta = (el / done) * (total - done) if done else 0.0
    (paths.LOGS / "realglm_progress.txt").write_text(
        f"realglm: {done}/{total} ({100 * done / total:.0f} %)\n"
        f"verstrichen : {el / 60:5.1f} min\n"
        f"ETA (Rest)  : {eta / 60:5.1f} min\n"
        f"letzte Zelle: {last}\n")


def first_level(conc, stim, family, constellation, noise_model, ar_order=30,
                max_jobs=-1):
    """GLM einer Datei -> beta des Tapping-Regressors, Dims (channel, chromo).

    `max_jobs=1`, wenn bereits auf Datei-Ebene parallelisiert wird (s. `main`).
    """
    basis = glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=0 * units.s)
    dm_hrf = glm.design_matrix.hrf_regressors(conc, stim, basis)
    hrf_names = [r for r in dm_hrf.common.regressor.values if str(r).startswith("HRF")]
    dm_hrf, _ = pl._normalize_hrf_to_unit_peak(dm_hrf, hrf_names)

    dm_drift, filt = drift_dm(family, conc)
    consistent = family.startswith("butterxy:")
    # Bestehende Filter-Arme bilden den Global-Regressor aus der gefilterten Zeitreihe
    # (unveraendert). Der konsistente Kontrollarm baut X ungefiltert auf und filtert dann
    # y und alle nichtkonstanten Spalten mit demselben Nullphasenfilter (apply_filter).
    nuisance_ts = (conc.cd.freq_filter(filt[0] * units.Hz, filt[1] * units.Hz, 4)
                   if filt is not None and not consistent else conc)

    dm = dm_hrf & dm_drift
    if constellation == "global":
        dm = dm & glm.design_matrix.global_mean_regressor(nuisance_ts)

    ts, dm = apply_filter(conc, dm, filt, filter_design=consistent)
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

    Pearson-Korrelation der Kanal-beta-Karten fuer jedes Durchgangspaar innerhalb eines
    Probanden, danach EIN Median ueber alle Paare (gepoolt, nicht erst je Proband): ein
    Proband mit sechs Durchgaengen traegt 15 Paare bei, einer mit zwei Durchgaengen eines.
    Rueckgabe (Median, Zahl der Paare).
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


def _out_path(mode: str, output=None) -> Path:
    if output is not None:
        return Path(output)
    suffix = {"test": "_test", "supplement": "_supplement"}.get(mode, "")
    return RESULTS / f"realglm_summary{suffix}.csv"


def main(mode="full", *, families=None, output=None):
    """Raster rechnen. `mode`: full (14 Familien), test (2 Probanden), supplement
    (`SUPPLEMENTAL_FAMILIES`, eigene CSV, danach `merge_supplement`). Vorhandene
    Ergebnisdateien werden nicht ueberschrieben, sondern vorher archiviert; neben der CSV
    entsteht `<name>.meta.json` mit Code-/Daten-Fingerabdruck (core/provenance.py).
    """
    paths.ensure()
    files = rd.find_files()
    if not files:
        print(f"Keine Daten in {rd.DATA_DIR}"); return

    if families is None:
        families = {"test": ["poly:3", "none"],
                    "supplement": list(SUPPLEMENTAL_FAMILIES)}.get(mode, list(FAMILIES))
    families = list(families)
    noise_models = NOISE_MODELS if mode != "test" else ["ols"]
    if mode == "test":
        subs = sorted({rd.subject_of(f) for f in files})[:2]
        files = [f for f in files if rd.subject_of(f) in subs]
    out = _out_path(mode, output)
    for old in (out, out.with_suffix(".meta.json")):
        if old.exists():
            print(f"Archiviere {old.name} -> {archive_file(old, 'before-realglm-rerun')}")
    meta = build_run_metadata(dict(analysis="realglm", mode=mode, families=families,
                                   constellations=CONSTELLATIONS, noise_models=noise_models,
                                   motion_method=MOTION_METHOD, ar_order=30,
                                   n_files=len(files)), files)

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
    # Parallelisierung auf Datei-Ebene: glm.fit parallelisiert intern kaum (alle Kanaele
    # teilen eine Designmatrix), die Dateien sind unabhaengig; innen daher max_jobs=1.
    # backend="threading" bringt nichts, AR-IRLS ist GIL-gebunden (Python-Schleifen in
    # statsmodels), also loky. loky pickelt Funktionen aus __main__ per Wert und scheitert
    # an cedalion-Objekten; der Worker wird deshalb per importlib aus dem Modul geholt und
    # so per Referenz gepickelt.
    _self = __spec__.name if __spec__ else "drift_glm.analysis.realglm"
    worker = importlib.import_module(_self).first_level
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
    meta["elapsed_seconds"] = round(time.time() - t0, 1)
    write_csv_atomic(df, out)
    write_metadata_atomic(meta, out.with_suffix(".meta.json"))
    print(f"\n[OK] {total} Fits in {(time.time() - t0) / 60:.1f} min -> {out}")

    d = df[(df.chromo == "HbO") & (df.noise_model == noise_models[0])]
    print("\nHbO, Rauschmodell "
          f"{noise_models[0]} -- signifikante Kanaele / Reproduzierbarkeit:")
    print(d.pivot_table(index="family", columns="constellation",
                        values=["n_significant", "reliability_r"])
           .to_string(float_format=lambda x: f"{x:.3f}"))


def merge_supplement(addition="realglm_summary_supplement.csv", base="realglm_summary.csv"):
    """Ergaenzungsarme an die Haupttabelle anhaengen: Archivkopie der Basis, keine
    doppelten Zellen (SUMMARY_KEY), gleiche Spalten, atomares Schreiben, Metadaten."""
    base_p, add_p = RESULTS / base, RESULTS / addition
    b, a = pd.read_csv(base_p), pd.read_csv(add_p)
    if set(b.columns) != set(a.columns):
        raise ValueError(f"Spalten unterscheiden sich: {set(b.columns) ^ set(a.columns)}")
    kb = set(map(tuple, b[SUMMARY_KEY].itertuples(index=False)))
    ka = set(map(tuple, a[SUMMARY_KEY].itertuples(index=False)))
    if kb & ka:
        raise ValueError(f"Zellen bereits vorhanden, nichts ersetzt: {sorted(kb & ka)[:4]}")
    if len(ka) != len(a):
        raise ValueError("Ergaenzung enthaelt doppelte Zellen")
    archived = archive_file(base_p, "before-merge-" + add_p.stem)
    merged = pd.concat([b, a[b.columns]], ignore_index=True)
    write_csv_atomic(merged, base_p)
    meta_p = base_p.with_suffix(".meta.json")
    meta = {}
    if meta_p.exists():
        import json
        meta = json.loads(meta_p.read_text())
        archive_file(meta_p, "before-merge-" + add_p.stem)
    add_meta_p = add_p.with_suffix(".meta.json")
    meta.setdefault("merged_supplements", []).append(dict(
        addition=str(add_p), rows=int(len(a)), archived_base=str(archived),
        addition_meta=(__import__("json").loads(add_meta_p.read_text())
                       if add_meta_p.exists() else None)))
    write_metadata_atomic(meta, meta_p)
    print(f"[OK] {len(a)} Zeilen aus {add_p.name} an {base_p.name} angehaengt "
          f"(Archiv: {archived})")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="full",
                        choices=["full", "test", "supplement", "merge"])
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--addition", default="realglm_summary_supplement.csv")
    args = parser.parse_args()
    if args.mode == "merge":
        merge_supplement(args.addition)
    else:
        main(args.mode, families=args.families, output=args.output)
