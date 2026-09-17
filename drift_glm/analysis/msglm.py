"""Multisubject-Fingertapping: GLM je Driftfamilie, Gruppenebene, Kanal- und Bildraum.

Ohne Ground Truth zaehlen zwei Kriterien: Halbierungs-Reproduzierbarkeit (gerade und
ungerade Trials je Bedingung gemeinsam modelliert, Korrelation der beta-Karten) und
kontralaterale Vorhersage (Tapping/Left ueber C4, Tapping/Right ueber C3; im Kanalraum
als Lateralisierungsindex, im Bildraum als Abstand des rekonstruierten Maximums zur
Landmarke). Der Datensatz hat echte kurze Kanaele, daher ist die Systemik-Achse
(`SYSTEMIC`) voll besetzt: Regressor in der Designmatrix (`_dm`) oder vorab abgezogen
(`_sub`, s. `shortchannel.subtract_global_component`). Ergebnis: results/msglm_summary.csv.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.msglm test    # 2 Probanden, 2 Familien (Timing)
    conda run -n cedalion python -m drift_glm.analysis.msglm         # voll (Nachtlauf)
"""

from __future__ import annotations

import argparse
import importlib
import json
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
from drift_glm.core.filtering import apply_filter
from drift_glm.analysis.sweep import drift_dm
from drift_glm import paths

RESULTS = paths.RESULTS
ALPHA = 0.05

FAMILIES = ["none", "poly:1", "poly:3", "poly:5", "dct:0.005", "dct:0.01", "dct:0.02",
            "legendre:1", "legendre:3", "bspline:5", "bspline:8", "butter:0.01"]
SUPPLEMENTAL_FAMILIES = ["butterxy:0.01"]
RELIABILITY_METHOD = "joint_even_odd_v1"
CELL_COLUMNS = ["family", "systemic", "noise_model"]
ROW_COLUMNS = CELL_COLUMNS + ["trial_type", "chromo"]
RELIABILITY_COLUMNS = ["reliability_r", "n_half_pairs", "reliability_method"]

#: Systemik-Achse: Suffix `_dm` = Regressor in der Designmatrix, `_sub` = vorab abgezogen
#: (Variante aus Cedalion-Notebook 50b).
SYSTEMIC = ["none", "global_dm", "short_avg_dm", "short_avg_sub",
            "short_maxcorr_dm", "short_closest_dm"]

NOISE_MODELS = ["ar_irls", "ols"]

#: AR-IRLS bekommt ein reduziertes Raster: auf diesen Aufnahmen (2974 s bei 7,8 Hz =
#: 23 240 Samples je Kanal) kostet ein Fit ueber alle Kanaele mit AR-IRLS (Ordnung 30,
#: ~4 x Abtastrate) rund das Zweihundertfache von OLS (poly:3: 145 s gegen 0,7 s). OLS
#: laeuft ueber das volle Raster.
AR_IRLS_FAMILIES = ["none", "poly:3", "butter:0.01"]
AR_IRLS_SYSTEMIC = ["none", "short_avg_dm", "short_avg_sub"]


def grid_for(noise_model: str, families, systemic):
    """Raster je Rauschmodell -- fuer AR-IRLS reduziert (s. `AR_IRLS_FAMILIES`)."""
    if noise_model != "ar_irls":
        return list(families), list(systemic)
    return ([f for f in families if f in AR_IRLS_FAMILIES],
            [s for s in systemic if s in AR_IRLS_SYSTEMIC])

#: Prozesse ueber Probanden. Niedriger als in realglm.py, weil die Aufnahmen 8x laenger
#: sind und jeder Prozess den cedalion-Import traegt; 5 Prozesse wurden auf 7,8 GB RAM vom
#: OOM-Killer beendet.
N_JOBS = 3


def _basis():
    return glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=0 * units.s)


def joint_half_stimuli(stim: pd.DataFrame) -> pd.DataFrame:
    """Label alternating trials within each condition; retain every event.

    Alternation follows onset order, independently of the input index. The first
    event has the historical name ``even`` (zero-based index). Conditions with
    fewer than two events cannot provide both halves and are rejected.
    """
    result = stim.copy(deep=True).reset_index(drop=True)
    if result.empty:
        raise ValueError("Split-half reliability requires stimulus events")
    for condition, events in result.groupby("trial_type", sort=False):
        if len(events) < 2:
            raise ValueError(f"Condition {condition!r} needs at least two trials")
        if str(condition).endswith((" [even]", " [odd]")):
            raise ValueError("Stimulus labels already contain split-half suffixes")
        ordered = events.sort_values("onset", kind="stable").index
        for half, indices in (("even", ordered[::2]), ("odd", ordered[1::2])):
            result.loc[indices, "trial_type"] = f"{condition} [{half}]"
    return result


def split_joint_betas(beta: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Align the even and odd estimates by their original condition names."""
    halves = []
    for half in ("even", "odd"):
        suffix = f" [{half}]"
        names = [str(tt) for tt in beta.trial_type.values if str(tt).endswith(suffix)]
        selected = beta.sel(trial_type=names)
        selected = selected.assign_coords(trial_type=[s[:-len(suffix)] for s in names])
        # ``regressor`` retains the suffixed labels, so discard this redundant
        # coordinate before comparing corresponding even and odd maps.
        if "regressor" in selected.coords:
            selected = selected.drop_vars("regressor")
        halves.append(selected.sortby("trial_type"))
    if (not halves[0].sizes["trial_type"]
            or not np.array_equal(halves[0].trial_type, halves[1].trial_type)):
        raise ValueError("Every condition needs an even and an odd HRF estimate")
    return tuple(halves)


def first_level(conc, geo3d, stim, family: str, systemic: str, noise_model: str,
                *, ar_order: int = 30, max_jobs: int = 1, half: str | None = None):
    """GLM einer Aufnahme -> beta je (channel, chromo, trial_type).

    ``half=None`` is the unchanged full fit. ``half='joint'`` fits separate even
    and odd HRFs for every condition in one model over the complete recording.
    ``half='even'`` / ``'odd'`` select one half from that joint fit; they never
    omit the other half's task regressors. The full fit is kept separate for
    group tests, lateralisation and image reconstruction.
    """
    if half not in (None, "joint", "even", "odd"):
        raise ValueError(f"Unknown split-half selection: {half!r}")
    if half is not None:
        stim = joint_half_stimuli(stim)

    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        conc, geo3d, distance_threshold=ms.SHORT_THRESHOLD)

    ts = ts_long
    if systemic.endswith("_sub"):
        variant = systemic[: -len("_sub")]
        ts = sc.subtract_global_component(ts_long, ts_short, stim, _basis(),
                                          variant=variant, geo3d=geo3d)

    dm_drift, filt = drift_dm(family, ts)
    consistent = family.startswith("butterxy:")
    # Legacy arms formed global means and selected correlated short channels
    # from filtered y. Preserve that ordering while applying the shared helper
    # to the complete design below. The new control builds X before filtering.
    nuisance_ts = (ts.cd.freq_filter(filt[0] * units.Hz, filt[1] * units.Hz, 4)
                   if filt is not None and not consistent else ts)

    dm_hrf = glm.design_matrix.hrf_regressors(ts, stim, _basis())
    hrf_names = [r for r in dm_hrf.common.regressor.values if str(r).startswith("HRF")]
    dm_hrf, _ = pl._normalize_hrf_to_unit_peak(dm_hrf, hrf_names)

    dm = dm_hrf & dm_drift
    if systemic == "global_dm":
        dm = dm & glm.design_matrix.global_mean_regressor(nuisance_ts)
    elif systemic.endswith("_dm"):
        dm = dm & sc.short_dm(systemic[: -len("_dm")], nuisance_ts, ts_short, geo3d)

    ts, dm = apply_filter(ts, dm, filt, filter_design=consistent)

    res = glm.fit(ts, dm, noise_model=noise_model, ar_order=ar_order, max_jobs=max_jobs)
    beta = res.sm.params.sel(regressor=hrf_names)
    beta = beta.assign_coords(
        trial_type=("regressor", [str(r).removeprefix("HRF ").strip()
                                  for r in hrf_names])).swap_dims(regressor="trial_type")
    if half in ("even", "odd"):
        return split_joint_betas(beta)[0 if half == "even" else 1]
    return beta


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
    """Median of subject-wise Pearson correlations between the two HbO maps.

    The three condition maps are concatenated over long channels within each
    subject, as in the original summary; subjects receive equal weight. This
    is within-recording split-half reliability, not test-retest reliability.
    """
    rs = []
    for a, b in pairs:
        a, b = xr.align(a, b, join="exact")
        b = b.transpose(*a.dims)
        x = np.asarray(a.sel(chromo=chromo).values, float).ravel()
        y = np.asarray(b.sel(chromo=chromo).values, float).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 3 and x[m].std() > 0 and y[m].std() > 0:
            rs.append(float(np.corrcoef(x[m], y[m])[0, 1]))
    return (float(np.median(rs)), len(rs)) if rs else (float("nan"), 0)


def hemisphere_of(channels, geo3d) -> np.ndarray:
    """+1 fuer Kanaele der linken, -1 fuer die der rechten Hemisphaere.

    Ueber die x-Koordinate des Kanalmittelpunkts relativ zur Mitte zwischen LPA und RPA;
    braucht also Landmarken in geo3d.
    """
    g = geo3d.pint.to("mm").pint.dequantify() if geo3d.pint.units is not None else geo3d
    lpa = np.asarray(g.sel(label="LPA").values, float)
    rpa = np.asarray(g.sel(label="RPA").values, float)
    axis = rpa - lpa
    axis = axis / np.linalg.norm(axis)                # zeigt nach rechts
    mid = 0.5 * (np.asarray(g.sel(label=channels.source.values).values, float)
                 + np.asarray(g.sel(label=channels.detector.values).values, float))
    proj = (mid - 0.5 * (lpa + rpa)) @ axis
    return np.where(proj < 0, 1.0, -1.0)


def lateralisation(mean_beta: xr.DataArray, geo3d, chromo="HbO") -> dict:
    """Kontralaterale Kontrolle im Kanalraum.

    Je Hand: mittleres beta kontralateral minus ipsilateral; positiv = Erwartung erfuellt.
    Differenz statt Verhaeltnis, weil kleine Nenner ein Verhaeltnis explodieren liessen.
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
    """Ein Proband, ein Fit; ausgelagert, damit joblib ihn per Referenz picklen kann.

    Die Vorverarbeitung liegt bewusst nicht hier: sie haengt weder an Familie noch
    Systemik-Stufe, und die Wavelet-Korrektur ist der teuerste Einzelschritt.
    """
    return first_level(conc, geo3d, stim, family, systemic, noise_model, half=half,
                       max_jobs=max_jobs)


def _out_path(mode: str) -> Path:
    suffix = {"test": "_test", "butterxy": "_butterxy"}.get(mode, "")
    return RESULTS / f"msglm_summary{suffix}.csv"


def complete_cells(frame: pd.DataFrame, *, with_halves: bool, with_image: bool,
                   n_subjects: int, require_joint: bool = True) -> set[tuple]:
    """Only a complete, unique six-row cell can be resumed or reused."""
    required = set(ROW_COLUMNS + ["n_subjects", "beta_mean", "n_significant"])
    if with_halves:
        required.update(["reliability_r", "n_half_pairs"])
        if require_joint:
            required.add("reliability_method")
    image_columns = ["loc_err_mm", "loc_err_wrong_mm", "peak_uM"]
    if with_image:
        required.update(image_columns)
    if not required <= set(frame.columns):
        return set()
    expected = {(tt, chromo) for tt in ms.CONDITIONS for chromo in ("HbO", "HbR")}
    done = set()
    for key, cell in frame.groupby(CELL_COLUMNS, sort=False):
        if len(cell) != len(expected) or cell.duplicated(ROW_COLUMNS).any():
            continue
        if set(zip(cell.trial_type, cell.chromo)) != expected:
            continue
        if not (cell.n_subjects == n_subjects).all():
            continue
        if with_halves:
            if (not np.isfinite(cell.reliability_r).all()
                    or not (cell.n_half_pairs == n_subjects).all()):
                continue
            if require_joint and not (cell.reliability_method == RELIABILITY_METHOD).all():
                continue
        if with_image:
            tapping = cell.trial_type.isin(ms.TAPPING)
            if not np.isfinite(cell.loc[tapping, image_columns].to_numpy(float)).all():
                continue
        done.add(key)
    return done


def assert_full_metrics_unchanged(updated: pd.DataFrame, reference: pd.DataFrame):
    """A reliability-only refresh may change no stored full-fit/image metric."""
    columns = [c for c in reference.columns if c not in RELIABILITY_COLUMNS]
    if not set(columns) <= set(updated.columns):
        raise AssertionError("Reliability refresh removed full-fit columns")
    if updated.duplicated(ROW_COLUMNS).any() or reference.duplicated(ROW_COLUMNS).any():
        raise AssertionError("Duplicate summary rows prevent an unambiguous comparison")
    original = reference[columns].set_index(ROW_COLUMNS).sort_index()
    new = updated[columns].set_index(ROW_COLUMNS).sort_index()
    if not new.index.isin(original.index).all():
        raise AssertionError("Reliability refresh introduced a cell absent from its archive")
    pd.testing.assert_frame_equal(new, original.loc[new.index], check_dtype=False,
                                  check_exact=True)


def main(mode: str = "full", *, motion_method: str = "wavelet",
         with_halves: bool = True, with_image: bool = True,
         families=None, systemic=None, noise_models=None,
         output_path: str | Path | None = None,
         reuse_full_from: str | Path | None = None, resume: bool = True):
    """Run the primary grid or the separate OLS consistent-filter control.

    ``reuse_full_from`` explicitly identifies an archived, complete full-fit
    summary. Only the new joint half fits are computed; all group, channel and
    image results are retained byte-for-value in memory and asserted unchanged.
    ``with_image=True`` requires complete image results in that archive. This
    avoids repeating mathematically unchanged full fits and reconstructions.
    A fresh full run still computes and checks its own full-fit results.
    """
    from drift_glm.core.provenance import (
        archive_file, build_run_metadata, write_csv_atomic, write_metadata_atomic,
    )

    if mode not in ("full", "test", "reliability", "butterxy"):
        raise ValueError(f"Unknown mode: {mode!r}")
    if mode == "reliability" and reuse_full_from is None:
        raise ValueError("Reliability refresh requires --reuse-full-from ARCHIVED.csv")
    if reuse_full_from is not None and not with_halves:
        raise ValueError("Reusing full results is only useful with joint half fits")
    paths.ensure()
    files = ms.paths()
    families = list(FAMILIES if families is None else families)
    systemic = list(SYSTEMIC if systemic is None else systemic)
    noise_models = list(NOISE_MODELS if noise_models is None else noise_models)
    if mode == "test":
        files = files[:2]
        families = ["poly:3", "dct:0.02"]
        systemic = ["none", "short_avg_dm", "short_avg_sub"]
        noise_models = ["ols"]
    elif mode == "butterxy":
        families, noise_models = SUPPLEMENTAL_FAMILIES, ["ols"]

    # Raster je Rauschmodell -- AR-IRLS reduziert, s. AR_IRLS_FAMILIES.
    grids = {nm: grid_for(nm, families, systemic) for nm in noise_models}
    total = sum(len(f) * len(s) for f, s in grids.values())
    if total == 0:
        raise ValueError("Requested grid has no permitted cells")
    target_cells = {(family, sysm, nm) for nm, (fams, sysms) in grids.items()
                    for family in fams for sysm in sysms}
    reference = None
    if reuse_full_from is not None:
        reuse_full_from = Path(reuse_full_from).resolve()
        reference = pd.read_csv(reuse_full_from)
        valid = complete_cells(reference, with_halves=False, with_image=with_image,
                               n_subjects=len(files))
        if not target_cells <= valid:
            raise ValueError(f"Archive lacks complete full-fit cells: {target_cells - valid}")

    out = Path(output_path) if output_path is not None else _out_path(mode)
    if reuse_full_from is not None and out.resolve() == reuse_full_from:
        raise ValueError("The archived reference must not be overwritten")
    metadata_path = out.with_suffix(".meta.json")
    config = dict(analysis="msglm", mode=mode, motion_method=motion_method,
                  with_halves=with_halves, with_image=with_image,
                  cells=sorted(target_cells), reliability_method=RELIABILITY_METHOD,
                  ar_order=30, reuse_full_from=str(reuse_full_from))
    metadata = build_run_metadata(config, [*files, *([reuse_full_from]
                                                   if reuse_full_from else [])])
    done_cells, recs = set(), []
    if out.exists():
        if resume:
            if not metadata_path.exists():
                raise ValueError("Existing CSV has no run metadata; use an archived "
                                 "reference and a fresh output, or --no-resume")
            saved_metadata = json.loads(metadata_path.read_text())
            if saved_metadata["config_fingerprint"] != metadata["config_fingerprint"]:
                raise ValueError("Resume refused: code, data or configuration changed")
            previous = pd.read_csv(out)
            done_cells = complete_cells(previous, with_halves=with_halves,
                                        with_image=with_image, n_subjects=len(files))
            done_cells &= target_cells
            recs = [r for r in previous.to_dict("records")
                    if tuple(r[c] for c in CELL_COLUMNS) in done_cells]
            if reference is not None and recs:
                assert_full_metrics_unchanged(pd.DataFrame(recs), reference)
        else:
            archive_file(out, "before-msglm-rerun")
            if metadata_path.exists():
                archive_file(metadata_path, "before-msglm-rerun")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_metadata_atomic(metadata, metadata_path)
    print(f"msglm [{mode}]: {len(files)} Probanden, {total} Zellen", flush=True)
    for nm, (f, s) in grids.items():
        print(f"  {nm:8s}: {len(f)} Familien x {len(s)} Systemik-Stufen "
              f"= {len(f) * len(s)} Zellen", flush=True)

    _self = __spec__.name if __spec__ else "drift_glm.analysis.msglm"
    worker = importlib.import_module(_self)._fit_one
    n_jobs = min(N_JOBS, len(files))

    # Vorverarbeitung einmal je Proband. Geometrie und Kanalmenge sind ueber die Probanden
    # identisch (dieselbe Montage), der erste Proband liefert sie fuer den Bildraum.
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
    if reference is not None and not (reference.n_channels == len(chans)).all():
        raise ValueError("Archived and current full fits have different channel counts")

    recon = c_meas_ref = sens = xyz = seeds = None
    if with_image and reference is None:
        adot_long = ims.adot(ms.DATASET).sel(channel=chans)
        recon, c_meas_ref = ims.recon_operator(adot_long, od0.sel(channel=chans))
        sens = ims.sensitivity_mask(adot_long)
        xyz = ims.vertex_coords_mm()
        head = ims.head()
        seeds = {lab: ims.seed_vertex(head, lab) for lab in ms.EXPECTED_SIDE.values()}
        print(f"Bildraum: {len(chans)} lange Kanaele, {int(sens.sum())} sichtbare "
              f"Vertices, alpha_meas={recon.alpha_meas:.3g}, "
              f"alpha_spatial={recon.alpha_spatial}", flush=True)

    done = 0
    halves_wanted = ("joint",) if with_halves else ()
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
                # The six-HRF model represents all trials in one fit. Keep the
                # three-HRF full fit separate, or explicitly reuse its archived
                # summary (including the complete image columns).
                fit_modes = (() if reference is not None else (None,)) + halves_wanted
                jobs = [(f, h) for f in files for h in fit_modes]
                # AR-IRLS strikt sequenziell (1 Prozess, 1 Kanal-Thread): ein einzelner
                # Kanal-Fit auf der dct:0.02-Designmatrix (122 Spalten x 23 239 Samples)
                # belegt ~2,9 GB Peak-RSS; auf der 7,8-GB-Maschine ist nur ein Fit
                # gleichzeitig sicher, mehr Prozesse oder Threads wurden OOM-gekillt.
                nj, mj = (1, 1) if nm == "ar_irls" else (n_jobs, 1)
                res = Parallel(n_jobs=nj, backend="loky")(
                    delayed(worker)(prepped[f][0], prepped[f][1], prepped[f][2],
                                    fam, sysm, nm, h, mj) for f, h in jobs)
                got = dict(zip(jobs, res))

                rel, n_pairs = float("nan"), 0
                if with_halves:
                    rel, n_pairs = half_reliability(
                        [split_joint_betas(got[(f, "joint")]) for f in files])

                if reference is not None:
                    selected = reference[(reference.family == fam)
                                         & (reference.systemic == sysm)
                                         & (reference.noise_model == nm)].copy()
                    selected["reliability_r"], selected["n_half_pairs"] = rel, n_pairs
                    selected["reliability_method"] = RELIABILITY_METHOD
                    assert_full_metrics_unchanged(selected, reference)
                    recs.extend(selected.to_dict("records"))
                    done += 1
                    dt = time.time() - tc
                    print(f"  [{done}/{total}] {fam:14s} {sysm:16s} {nm:7s} "
                          f"joint rel={rel:+.3f}; full/image preserved ({dt:.1f}s)",
                          flush=True)
                    write_csv_atomic(pd.DataFrame(recs), out)
                    continue

                by_sub = {ms.subject_of(f): got[(f, None)] for f in files}
                t, rej, mean = group_test(by_sub)

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
                                   reliability_r=rel, n_half_pairs=n_pairs,
                                   reliability_method=(RELIABILITY_METHOD
                                                       if with_halves else "disabled"),
                                   **lat)
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
                # Nach jeder Zelle schreiben; der Lauf dauert Stunden.
                write_csv_atomic(pd.DataFrame(recs), out)

    df = pd.DataFrame(recs)
    if reference is not None:
        assert_full_metrics_unchanged(df, reference)
    write_csv_atomic(df, out)
    print(f"\n[OK] {total} Zellen in {(time.time() - t0) / 60:.1f} min -> {out}")
    d = df[(df.chromo == "HbO") & (df.trial_type == "Tapping/Right")]
    print("\nHbO, Tapping/Right -- Reproduzierbarkeit und Lateralisierung:")
    print(d.pivot_table(index="family", columns="systemic",
                        values=["reliability_r", "n_significant"])
           .to_string(float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="full",
                        choices=["full", "test", "reliability", "butterxy"])
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--systemic", nargs="+")
    parser.add_argument("--noise-models", nargs="+", choices=NOISE_MODELS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-full-from", type=Path)
    parser.add_argument("--motion-method", default="wavelet")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    main(args.mode, motion_method=args.motion_method, families=args.families,
         systemic=args.systemic, noise_models=args.noise_models,
         output_path=args.output, reuse_full_from=args.reuse_full_from,
         resume=not args.no_resume)
