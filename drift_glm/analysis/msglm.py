"""Multisubject-Fingertapping: GLM je Driftfamilie, Gruppenebene, Bildraum.

Der dritte Auswertungsstrang neben `sweep.py` (Simulation) und `realglm.py` (Khan). Wie
dort gibt es keine Ground Truth -- aber dieser Datensatz erlaubt zwei Kriterien, die auf
den anderen nicht verfuegbar waren:

  * **Halbierungs-Reproduzierbarkeit.** Jede Bedingung hat 30 Trials. Werden die geraden
    und ungeraden Trials getrennt gefittet, muessen beide Haelften dieselbe beta-Karte
    liefern. Das ist der Ersatz fuer die Durchgangspaare bei Khan (dort 2-3 Durchgaenge je
    Proband, hier nur eine Aufnahme) und dasselbe wahrheitsfreie Guetemass: eine Familie,
    die Rauschen als Aktivierung modelliert, ist zwischen den Haelften inkonsistent.

  * **Kontralaterale Vorhersage.** Getappt wird mit linker und rechter Hand getrennt, und
    Motorik ist kontralateral organisiert. Also MUSS Tapping/Left ueber C4 und
    Tapping/Right ueber C3 landen. Im Kanalraum als Lateralisierungsindex ueber die
    Kanalgruppen, im Bildraum als Abstand des rekonstruierten Maximums zur erwarteten
    Landmarke. Genau diese Kontrolle war bei Khan unmoeglich, weil dort die Landmarken
    fehlen (siehe BESPRECHUNG, Abb. 17) -- sie ist das inhaltliche Argument fuer den
    Bildraum.

DIE SYSTEMIK-ACHSE. Weil dieser Datensatz echte kurze Kanaele hat, sind erstmals alle
Varianten anwendbar, und zusaetzlich die Frage aus den Gespraechsnotizen ("global
components subtraction anschauen"): gehoert der systemische Anteil in die Designmatrix
oder wird er vorher abgezogen? Beides benutzt denselben Regressor, ist aber nicht
dasselbe -- Begruendung in `shortchannel.subtract_global_component`.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.msglm test    # 2 Probanden, 2 Familien (Timing)
    conda run -n cedalion python -m drift_glm.analysis.msglm         # voll (Nachtlauf)
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

import cedalion.models.glm as glm
import cedalion.nirs
from cedalion import units

from drift_glm.analysis import imageglm as ig
from drift_glm.core import imagespace as ims
from drift_glm.data import multisubject as ms
from drift_glm.core import pipeline as pl
from drift_glm.core import shortchannel as sc
from drift_glm.analysis.sweep import drift_dm
from drift_glm import paths

RESULTS = paths.RESULTS
ALPHA = 0.05

FAMILIES = ["none", "poly:1", "poly:3", "poly:5", "dct:0.005", "dct:0.01", "dct:0.02",
            "legendre:1", "legendre:3", "bspline:5", "bspline:8", "butter:0.01"]

#: Die Systemik-Achse: WELCHER Regressor und WIE er angewandt wird. Suffix `_dm` = bleibt
#: in der Designmatrix, `_sub` = wird vorher abgezogen (Notebook-50b-Variante).
SYSTEMIC = ["none", "global_dm", "short_avg_dm", "short_avg_sub",
            "short_maxcorr_dm", "short_closest_dm"]

NOISE_MODELS = ["ar_irls", "ols"]

#: AR-IRLS bekommt ein REDUZIERTES Raster, und das ist eine Messung, keine Bequemlichkeit.
#: Auf diesem Datensatz (2974 s bei 7,8125 Hz = 23 240 Samples je Kanal) gemessen, ein Fit
#: ueber alle Kanaele:
#:
#:     poly:3   + OLS       0,7 s        poly:3   + AR-IRLS   145,3 s
#:     dct:0.02 + OLS      53,8 s        dct:0.02 + AR-IRLS   > 500 s
#:
#: AR-IRLS kostet also rund das Zweihundertfache. Der Grund ist die Prewhitening-Stufe: sie
#: schaetzt je Kanal iterativ ein AR-Modell der Ordnung 30 ueber 23 000 Samples. Die Ordnung
#: liesse sich nicht einfach senken -- die Empfehlung ist ~4 x Abtastrate, bei 7,8 Hz also
#: genau 30.
#:
#: Das volle Raster (12 Familien x 6 Systemik-Stufen x 3 Fits x 5 Probanden) waere mit
#: AR-IRLS mehrere Tage. Deshalb: OLS ueber das VOLLE Raster, AR-IRLS ueber die Familien und
#: Stufen, an denen die Fragestellung haengt. Die Kanalraum-Rangliste aus `realglm.py` gibt
#: es fuer beide Rauschmodelle, der Vergleich ist also nicht verloren.
AR_IRLS_FAMILIES = ["none", "poly:3", "dct:0.02", "butter:0.01"]
AR_IRLS_SYSTEMIC = ["none", "short_avg_dm", "short_avg_sub"]


def grid_for(noise_model: str, families, systemic):
    """Raster je Rauschmodell -- fuer AR-IRLS reduziert (s. `AR_IRLS_FAMILIES`)."""
    if noise_model != "ar_irls":
        return list(families), list(systemic)
    return ([f for f in families if f in AR_IRLS_FAMILIES],
            [s for s in systemic if s in AR_IRLS_SYSTEMIC])

#: Prozesse fuer die Parallelisierung ueber Probanden. Wie in realglm.py: cedalions
#: glm.fit parallelisiert intern kaum, die Probanden sind dagegen unabhaengig.
#: Niedriger als die 5 in realglm.py, weil die Aufnahmen hier 8x laenger sind (2974 s bei
#: 7,8 Hz = 23 240 Samples gegen 1367) und jeder Prozess zusaetzlich den cedalion-Import
#: traegt. Mit 5 Prozessen wurde der Lauf auf dieser Maschine (7,8 GB) vom OOM-Killer
#: beendet.
N_JOBS = 3


def _basis():
    return glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=0 * units.s)


def first_level(conc, geo3d, stim, family: str, systemic: str, noise_model: str,
                *, ar_order: int = 30, max_jobs: int = 1, half: str | None = None):
    """GLM einer Aufnahme -> beta je (channel, chromo, trial_type).

    `half` waehlt fuer die Halbierungs-Reproduzierbarkeit jeden zweiten Trial je
    Bedingung aus (`"even"` / `"odd"`). Wichtig: es wird JE BEDINGUNG halbiert, nicht
    global -- sonst haette eine Haelfte mehr Trials einer Bedingung als die andere und der
    Unterschied waere ein Design-Effekt statt eines Rauscheffekts.
    """
    if half is not None:
        keep = []
        for _, g in stim.groupby("trial_type", sort=True):
            idx = list(g.index)
            keep += idx[0::2] if half == "even" else idx[1::2]
        stim = stim.loc[sorted(keep)].reset_index(drop=True)

    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        conc, geo3d, distance_threshold=ms.SHORT_THRESHOLD)

    ts = ts_long
    if systemic.endswith("_sub"):
        variant = systemic[: -len("_sub")]
        ts = sc.subtract_global_component(ts_long, ts_short, stim, _basis(),
                                          variant=variant, geo3d=geo3d)

    dm_drift, filt = drift_dm(family, ts)
    if filt is not None:
        ts = ts.cd.freq_filter(filt[0] * units.Hz, filt[1] * units.Hz, 4)

    dm_hrf = glm.design_matrix.hrf_regressors(ts, stim, _basis())
    hrf_names = [r for r in dm_hrf.common.regressor.values if str(r).startswith("HRF")]
    dm_hrf, _ = pl._normalize_hrf_to_unit_peak(dm_hrf, hrf_names)

    dm = dm_hrf & dm_drift
    if systemic == "global_dm":
        dm = dm & glm.design_matrix.global_mean_regressor(ts)
    elif systemic.endswith("_dm"):
        dm = dm & sc.short_dm(systemic[: -len("_dm")], ts, ts_short, geo3d)

    res = glm.fit(ts, dm, noise_model=noise_model, ar_order=ar_order, max_jobs=max_jobs)
    beta = res.sm.params.sel(regressor=hrf_names)
    return beta.assign_coords(
        trial_type=("regressor", [str(r).removeprefix("HRF ").strip()
                                  for r in hrf_names])).swap_dims(regressor="trial_type")


def group_test(betas: dict[str, xr.DataArray]):
    """Einstichproben-t-Test ueber die Probanden + Benjamini-Hochberg-FDR."""
    subs = sorted(betas)
    stack = xr.concat([betas[s] for s in subs], dim="subject").assign_coords(subject=subs)
    mean = stack.mean("subject")
    t, p = stats.ttest_1samp(stack.values, popmean=0.0, axis=0, nan_policy="omit")
    p = np.asarray(p, dtype=float)
    flat, ok = p.ravel(), np.isfinite(p.ravel())
    rej = np.zeros_like(flat, dtype=bool)
    if ok.any():
        rej[ok], _, _, _ = multipletests(flat[ok], alpha=ALPHA, method="fdr_bh")
    return (xr.DataArray(np.asarray(t), dims=mean.dims, coords=mean.coords),
            xr.DataArray(rej.reshape(p.shape), dims=mean.dims, coords=mean.coords),
            mean)


def half_reliability(pairs: list[tuple[xr.DataArray, xr.DataArray]], chromo="HbO"
                     ) -> tuple[float, int]:
    """Median-Korrelation der beta-Karten zwischen den beiden Trial-Haelften."""
    rs = []
    for a, b in pairs:
        x = np.asarray(a.sel(chromo=chromo).values, float).ravel()
        y = np.asarray(b.sel(chromo=chromo).values, float).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 3 and x[m].std() > 0 and y[m].std() > 0:
            rs.append(float(np.corrcoef(x[m], y[m])[0, 1]))
    return (float(np.median(rs)), len(rs)) if rs else (float("nan"), 0)


def hemisphere_of(channels, geo3d) -> np.ndarray:
    """+1 fuer Kanaele der linken, -1 fuer die der rechten Hemisphaere.

    Ueber die x-Koordinate des Kanalmittelpunkts, bezogen auf die Mitte zwischen LPA und
    RPA. Deshalb braucht dieser Test Landmarken -- und deshalb ist er beim Khan-Datensatz
    unmoeglich.
    """
    g = geo3d.pint.to("mm").pint.dequantify() if geo3d.pint.units is not None else geo3d
    lpa = np.asarray(g.sel(label="LPA").values, float)
    rpa = np.asarray(g.sel(label="RPA").values, float)
    axis = rpa - lpa
    axis = axis / np.linalg.norm(axis)                # zeigt nach RECHTS
    mid = 0.5 * (np.asarray(g.sel(label=channels.source.values).values, float)
                 + np.asarray(g.sel(label=channels.detector.values).values, float))
    proj = (mid - 0.5 * (lpa + rpa)) @ axis
    return np.where(proj < 0, 1.0, -1.0)


def lateralisation(mean_beta: xr.DataArray, geo3d, chromo="HbO") -> dict:
    """Kontralaterale Kontrolle im Kanalraum.

    Fuer jede Hand: mittleres beta auf der kontralateralen minus auf der ipsilateralen
    Hemisphaere. Positiv = die Erwartung ist erfuellt. Der Wert ist bewusst eine Differenz
    und kein Verhaeltnis -- bei kleinen Nennern wuerde ein Verhaeltnis explodieren.
    """
    side = hemisphere_of(mean_beta, geo3d)            # +1 links, -1 rechts
    out = {}
    for tt, expect in ms.EXPECTED_SIDE.items():
        if tt not in [str(x) for x in np.atleast_1d(mean_beta.trial_type.values)]:
            continue
        v = np.asarray(mean_beta.sel(trial_type=tt, chromo=chromo).values, float)
        contra = side > 0 if expect == "C3" else side < 0
        out[f"lat_{tt.split('/')[-1].lower()}"] = float(
            np.nanmean(v[contra]) - np.nanmean(v[~contra]))
    return out


def _fit_one(conc, geo3d, stim, family, systemic, noise_model, half, max_jobs=1):
    """Ein Proband, ein Fit. Ausgelagert, damit joblib ihn per Referenz picklen kann.

    Die Vorverarbeitung passiert bewusst NICHT hier: sie haengt weder an der Familie noch
    an der Systemik-Stufe, und die Wavelet-Korrektur ist der teuerste Einzelschritt. Wuerde
    sie je Zelle laufen, waere sie bei 144 Zellen der Kostentreiber -- dieselbe Lehre wie
    in `realglm.py`.
    """
    return first_level(conc, geo3d, stim, family, systemic, noise_model, half=half,
                       max_jobs=max_jobs)


def _out_path(mode: str) -> Path:
    return RESULTS / ("msglm_summary_test.csv" if mode == "test"
                      else "msglm_summary.csv")


def main(mode: str = "full", *, motion_method: str = "wavelet",
         with_halves: bool = True, with_image: bool = True):
    paths.ensure()
    files = ms.paths()
    families, systemic, noise_models = FAMILIES, SYSTEMIC, NOISE_MODELS
    if mode == "test":
        files = files[:2]
        families = ["poly:3", "dct:0.02"]
        systemic = ["none", "short_avg_dm", "short_avg_sub"]
        noise_models = ["ols"]

    # Raster je Rauschmodell -- AR-IRLS reduziert, s. AR_IRLS_FAMILIES.
    grids = {nm: grid_for(nm, families, systemic) for nm in noise_models}
    total = sum(len(f) * len(s) for f, s in grids.values())
    print(f"msglm [{mode}]: {len(files)} Probanden, {total} Zellen", flush=True)
    for nm, (f, s) in grids.items():
        print(f"  {nm:8s}: {len(f)} Familien x {len(s)} Systemik-Stufen "
              f"= {len(f) * len(s)} Zellen", flush=True)

    _self = __spec__.name if __spec__ else "drift_glm.analysis.msglm"
    worker = importlib.import_module(_self)._fit_one
    n_jobs = min(N_JOBS, len(files))

    # Vorverarbeitung EINMAL je Proband. Die Geometrie und die Kanalmenge sind ueber die
    # Probanden identisch (dieselbe Montage), der Referenz-Proband liefert sie also fuer
    # den Bildraum mit.
    t0 = time.time()
    prepped = {}
    for f in files:
        rec = ms.load(f)
        P, _ = ms.preprocess_recording(rec, motion_method=motion_method)
        prepped[f] = (P.conc, P.geo3d, ms.stim_df(rec, "hands"), P.od)
    conc0, geo0, _, od0 = prepped[files[0]]
    print(f"Vorverarbeitung: {len(files)} Probanden in {time.time() - t0:.0f}s",
          flush=True)

    long0, _ = cedalion.nirs.split_long_short_channels(
        conc0, geo0, distance_threshold=ms.SHORT_THRESHOLD)
    chans = [str(c) for c in long0.channel.values]

    recon = c_meas_ref = sens = xyz = seeds = None
    if with_image:
        adot_long = ims.adot(ms.DATASET).sel(channel=chans)
        recon, c_meas_ref = ims.recon_operator(adot_long, od0.sel(channel=chans))
        sens = ims.sensitivity_mask(adot_long)
        xyz = ims.vertex_coords_mm()
        head = ims.head()
        seeds = {lab: ims.seed_vertex(head, lab) for lab in ms.EXPECTED_SIDE.values()}
        print(f"Bildraum: {len(chans)} lange Kanaele, {int(sens.sum())} sichtbare "
              f"Vertices, alpha_meas={recon.alpha_meas:.3g}, "
              f"alpha_spatial={recon.alpha_spatial}", flush=True)

    # Wiederaufnahme: bereits gerechnete Zellen aus einer frueheren (z.B. vom OOM-Killer
    # beendeten) Tabelle uebernehmen statt sie neu zu rechnen. Der Volllauf am 12.08.
    # brach bei 81/84 Zellen ab -- ohne Wiederaufnahme wuerde jeder Neustart die komplette
    # OLS-Haelfte wiederholen, nur um an die 3 fehlenden AR-IRLS-Zellen zu kommen.
    # Neu rechnen erzwingen: die Summary-CSV loeschen oder umbenennen.
    out = _out_path(mode)
    done_cells, recs = set(), []
    if out.exists():
        prev = pd.read_csv(out)
        if {"family", "systemic", "noise_model"} <= set(prev.columns):
            done_cells = {(r.family, r.systemic, r.noise_model) for r in
                          prev[["family", "systemic", "noise_model"]]
                          .drop_duplicates().itertuples(index=False)}
            recs = prev.to_dict("records")
            print(f"Resume: {len(done_cells)} Zellen aus {out.name} uebernommen "
                  f"(neu rechnen: Datei loeschen)", flush=True)

    done = 0
    halves_wanted = ("even", "odd") if with_halves else ()
    # Rauschmodell aussen: OLS zuerst, damit bei einem Abbruch die vollstaendige
    # OLS-Tabelle schon geschrieben ist und nicht die teure AR-IRLS-Haelfte fehlt.
    for nm in sorted(noise_models, key=lambda x: x != "ols"):
        fams_nm, sys_nm = grids[nm]
        for fam in fams_nm:
            for sysm in sys_nm:
                if (fam, sysm, nm) in done_cells:
                    done += 1
                    print(f"  [{done}/{total}] {fam:14s} {sysm:16s} {nm:7s} "
                          f"uebernommen (Resume)", flush=True)
                    continue
                tc = time.time()
                # Vollfit und (falls gewuenscht) beide Haelften in EINEM Parallel-Aufruf:
                # loky startet dann einmal statt dreimal Prozesse.
                #
                # AR-IRLS laeuft bewusst mit EINEM Prozess und stattdessen mit
                # Thread-Parallelitaet UEBER DIE KANAELE (glm.fit, max_jobs=-1): drei
                # parallele AR-IRLS-Prozesse auf der dct:0.02-Designmatrix (~150
                # Spalten x 23 240 Samples) wurden auf dieser Maschine (7,8 GB) zweimal
                # vom OOM-Killer beendet (12.08. bei 81/84, erneut 08.09.) -- Threads
                # teilen sich den Speicher, Prozesse nicht.
                jobs = [(f, h) for f in files for h in (None, *halves_wanted)]
                # max_jobs=2 statt -1: jeder Kanal-Fit haelt transiente Kopien der
                # (grossen) Designmatrix; mit 4 Threads wurde der Prozess bei 5,1 GB
                # anon-rss vom OOM-Killer beendet (dmesg 08.09.), 2 Threads bleiben
                # unter ~3 GB. Waehrend des Laufs nichts anderes Grosses starten.
                nj, mj = (1, 2) if nm == "ar_irls" else (n_jobs, 1)
                res = Parallel(n_jobs=nj, backend="loky")(
                    delayed(worker)(prepped[f][0], prepped[f][1], prepped[f][2],
                                    fam, sysm, nm, h, mj) for f, h in jobs)
                got = dict(zip(jobs, res))
                by_sub = {ms.subject_of(f): got[(f, None)] for f in files}
                t, rej, mean = group_test(by_sub)

                rel, n_pairs = float("nan"), 0
                if with_halves:
                    rel, n_pairs = half_reliability(
                        [(got[(f, "even")], got[(f, "odd")]) for f in files])

                lat = lateralisation(mean, geo0)

                img_rows = {}
                if with_image:
                    for tt, expect in ms.EXPECTED_SIDE.items():
                        m = mean.sel(trial_type=tt)
                        # Wellenlaengen kommen aus der OD (long0 ist Konzentration und
                        # traegt nur chromo), Kanalkoordinaten aus long0.
                        od = ig.conc_map_to_od(m, geo0, od0.wavelength, like=long0)
                        img = recon.reconstruct(od, c_meas_ref)
                        for ch in ("HbO", "HbR"):
                            h = np.asarray(img.sel(chromo=ch).pint.dequantify().values,
                                           float)
                            idx = np.flatnonzero(sens)
                            j = idx[int(np.nanargmax(np.abs(h[sens])))]
                            img_rows[(tt, ch)] = dict(
                                loc_err_mm=float(np.linalg.norm(xyz[j]
                                                                - xyz[seeds[expect]])),
                                loc_err_wrong_mm=float(np.linalg.norm(
                                    xyz[j] - xyz[seeds["C3" if expect == "C4"
                                                       else "C4"]])),
                                peak_uM=float(np.abs(h[sens]).max()))

                for tt in [str(x) for x in np.atleast_1d(mean.trial_type.values)]:
                    for ch in ("HbO", "HbR"):
                        mv = mean.sel(trial_type=tt, chromo=ch).values
                        rv = rej.sel(trial_type=tt, chromo=ch).values
                        row = dict(family=fam, systemic=sysm, noise_model=nm,
                                   trial_type=tt, chromo=ch, n_subjects=len(by_sub),
                                   n_channels=int(mv.size),
                                   n_significant=int(rv.sum()),
                                   beta_mean=float(np.nanmean(mv)),
                                   beta_max=float(np.nanmax(np.abs(mv))),
                                   t_max=float(np.nanmax(np.abs(
                                       t.sel(trial_type=tt, chromo=ch).values))),
                                   reliability_r=rel, n_half_pairs=n_pairs, **lat)
                        row.update(img_rows.get((tt, ch), {}))
                        recs.append(row)

                # Plausibilitaet: HbR/HbO ueber die staerksten HbO-Kanaele
                mo = mean.sel(chromo="HbO").values.ravel()
                mr = mean.sel(chromo="HbR").values.ravel()
                good = np.isfinite(mo) & np.isfinite(mr)
                sel = good & (np.abs(mo) > np.nanpercentile(np.abs(mo[good]), 75))
                ratio = float(np.nanmedian(mr[sel] / mo[sel])) if sel.any() else np.nan
                corr = (float(np.corrcoef(mo[good], mr[good])[0, 1])
                        if good.sum() > 3 else np.nan)
                for row in recs[-6:]:
                    row["hbr_hbo_ratio"], row["hbo_hbr_corr"] = ratio, corr

                done += 1
                dt = time.time() - tc
                print(f"  [{done}/{total}] {fam:14s} {sysm:16s} {nm:7s} "
                      f"rel={rel:+.3f} lat_left={lat.get('lat_left', float('nan')):+.4f} "
                      f"lat_right={lat.get('lat_right', float('nan')):+.4f} ({dt:5.1f}s)",
                      flush=True)
                (paths.LOGS / "msglm_progress.txt").write_text(
                    f"msglm: {done}/{total}\n"
                    f"verstrichen: {(time.time() - t0) / 60:.1f} min\n"
                    f"ETA: {(time.time() - t0) / done * (total - done) / 60:.1f} min\n"
                    f"letzte Zelle: {fam} {sysm} {nm}\n")
                # Nach JEDER Zelle schreiben. Der Lauf dauert Stunden; ein Abbruch nach
                # zwei Dritteln soll nicht zwei Drittel der Ergebnisse mitnehmen.
                pd.DataFrame(recs).to_csv(_out_path(mode), index=False)

    df = pd.DataFrame(recs)
    out = _out_path(mode)
    df.to_csv(out, index=False)
    print(f"\n[OK] {total} Zellen in {(time.time() - t0) / 60:.1f} min -> {out}")
    d = df[(df.chromo == "HbO") & (df.trial_type == "Tapping/Right")]
    print("\nHbO, Tapping/Right -- Reproduzierbarkeit und Lateralisierung:")
    print(d.pivot_table(index="family", columns="systemic",
                        values=["reliability_r", "n_significant"])
           .to_string(float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
