"""Abbildungen zu den realen Daten (Khan-Datensatz, Abb. 15-18).

Liest `results/realglm_summary.csv` und rechnet fuer eine Konfiguration die
Gruppen-beta-Karte nach (Abb. 17, ~13 min; Zwischenergebnis in results/realglm_group_map.nc).

  Abb. 15  Reproduzierbarkeit je Driftfamilie (Hauptkriterium ohne Ground Truth)
  Abb. 16  Signifikante Kanaele nach FDR je Familie x Konstellation
  Abb. 17  Gruppen-beta-Karte im Montage-Layout (HbO/HbR)
  Abb. 18  HbO/HbR-Plausibilitaet

Aufruf:
    conda run -n cedalion python -m drift_glm.reports.realglm_report          # alles
    conda run -n cedalion python -m drift_glm.reports.realglm_report quick    # ohne Abb. 17
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
from joblib import Parallel, delayed

import cedalion
import cedalion.vis.anatomy

from drift_glm.data import realdata as rd
from drift_glm.analysis import realglm as rg
from drift_glm import paths

RES = paths.RESULTS
OUT = paths.FIGURES

ORDER = ["none", "poly:1", "poly:2", "poly:3", "poly:5",
         "dct:0.005", "dct:0.01", "dct:0.02", "legendre:1", "legendre:3",
         "bspline:5", "bspline:8", "butter:0.01", "butterxy:0.01",
         "lowpass:0.5", "bandpass:0.01-0.5"]


def _color(fam):
    if fam.startswith("poly"):     return "#4c72b0"
    if fam.startswith("dct"):      return "#dd8452"
    if fam.startswith("legendre"): return "#55a868"
    if fam.startswith("bspline"):  return "#8172b3"
    if fam == "none":              return "#937860"
    if fam.startswith("butterxy"): return "#7b2d8e"   # konsistenter Filter-Kontrollarm
    return "#c44e52"               # Filter-Alternativen


def load():
    df = pd.read_csv(RES / "realglm_summary.csv")
    df["degeneriert"] = [rg.is_degenerate(f, n)
                         for f, n in zip(df.family, df.noise_model)]
    return df


def fig15_reliability(df):
    """Abb. 15 -- Reproduzierbarkeit je Familie (Hauptkriterium ohne Ground Truth)."""
    d = df[(df.chromo == "HbO") & (df.constellation == "baseline")]
    fams = [f for f in ORDER if f in set(d.family)]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, (nm, hatch) in enumerate([("ar_irls", None), ("ols", "//")]):
        v, deg = [], []
        for f in fams:
            r = d[(d.family == f) & (d.noise_model == nm)]
            v.append(float(r.reliability_r.iloc[0]) if len(r) else np.nan)
            deg.append(bool(r.degeneriert.iloc[0]) if len(r) else False)
        v = np.array(v, dtype=float)
        v_plot = np.where(deg, np.nan, v)          # degenerierte Zellen nicht zeichnen
        ax.bar(x + (k - 0.5) * 0.4, v_plot, width=0.38,
               color=[_color(f) for f in fams], alpha=1.0 if k == 0 else 0.55,
               hatch=hatch, edgecolor="white",
               label=f"{nm}" + (" (schraffiert)" if k else ""))
        for xi, (dg, f) in enumerate(zip(deg, fams)):
            if dg:
                ax.text(xi + (k - 0.5) * 0.4, 0.02, "n.a.", ha="center",
                        fontsize=8, rotation=90, color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=55, ha="right", fontsize=9)
    ax.set_ylabel("Median-Korrelation der β-Karten zwischen Durchgängen")
    ax.set_title("Reproduzierbarkeit je Driftfamilie — 25 Probanden, 74 Durchgangspaare\n"
                 "(höher = stabilere Schätzung; „n.a.\" = mit AR-IRLS nicht auswertbar)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "15_real_reliability.png", dpi=130)
    plt.close(fig)


def fig16_significant(df):
    """Abb. 16 -- signifikante Kanaele nach FDR je Familie und Konstellation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, nm in zip(axes, ["ar_irls", "ols"]):
        d = df[(df.chromo == "HbO") & (df.noise_model == nm) & (~df.degeneriert)]
        fams = [f for f in ORDER if f in set(d.family)]
        x = np.arange(len(fams))
        for k, con in enumerate(["baseline", "global"]):
            v = [float(d[(d.family == f) & (d.constellation == con)].n_significant.iloc[0])
                 for f in fams]
            ax.bar(x + (k - 0.5) * 0.4, v, width=0.38, label=con,
                   color="#4c72b0" if k == 0 else "#dd8452")
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=55, ha="right", fontsize=8)
        ax.set_title(f"{nm}"); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("signifikante Kanäle von 48 (FDR, q = 0,05)")
    axes[0].legend()
    fig.suptitle("Aktivierungs-Detektion auf den realen Daten: signifikante Kanäle "
                 "je Driftfamilie und Konstellation")
    fig.tight_layout(); fig.savefig(OUT / "16_real_significant.png", dpi=130)
    plt.close(fig)


def fig18_plausibility(df):
    """Abb. 18 -- HbR/HbO-Verhaeltnis und HbO/HbR-Korrelation je Driftfamilie."""
    d = df[(df.chromo == "HbO") & (~df.degeneriert)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (col, ttl, ref) in zip(axes, [
            ("hbr_hbo_ratio", "HbR/HbO-Verhältnis", -0.4),
            ("hbo_hbr_corr", "Korrelation HbO ↔ HbR", None)]):
        for k, nm in enumerate(["ar_irls", "ols"]):
            for j, con in enumerate(["baseline", "global"]):
                s = d[(d.noise_model == nm) & (d.constellation == con)][col]
                ax.scatter([k * 2 + j] * len(s), s, alpha=0.6,
                           color="#4c72b0" if j == 0 else "#dd8452", s=28)
        if ref is not None:
            ax.axhline(ref, ls="--", color="k", lw=1,
                       label=f"physiologisch erwartet ({ref})")
            ax.legend(fontsize=8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["ar_irls\nbaseline", "ar_irls\nglobal",
                            "ols\nbaseline", "ols\nglobal"], fontsize=8)
        ax.set_title(ttl); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Plausibilität auf den realen Daten (je Punkt eine Driftfamilie)")
    fig.tight_layout(); fig.savefig(OUT / "18_real_plausibility.png", dpi=130)
    plt.close(fig)


def _montage_xy(conc, geo3d):
    """2D-Layout der Kanaele aus den Optodenpositionen, anatomisch orientiert.

    Cedalions `scalp_plot` braucht die Landmarken Nz/LPA/RPA, die dieser Datensatz nicht
    hat. Die Hauptkomponenten-Projektion wird deshalb ueber die Kanalzuordnung aus Khan
    et al. (2026), Tabelle 3 (`realdata.khan_hemisphere`, 10-10-Positionen) ausgerichtet:
    linke Hemisphaere links (negatives x), frontale Positionen (F*, FC*, FT*) oben.
    """
    g = geo3d.pint.dequantify() if hasattr(geo3d, "pint") else geo3d
    pos = {str(l): np.asarray(v, float) for l, v in zip(g.label.values, g.values)}
    mid = np.array([0.5 * (pos[str(s)] + pos[str(d)])
                    for s, d in zip(conc.source.values, conc.detector.values)])
    c = mid - mid.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ vt[:2].T          # (channel, 2)
    chans = [str(ch) for ch in conc.channel.values]
    side = rd.khan_hemisphere(chans)                       # +1 links
    if np.mean(xy[side > 0, 0]) > np.mean(xy[side < 0, 0]):
        xy[:, 0] *= -1
    table = rd.khan_channel_table(chans)
    frontal = np.array([lab.startswith("F") for lab in table.source_1010]) \
        | np.array([lab.startswith("F") for lab in table.detector_1010])
    if np.mean(xy[frontal, 1]) < np.mean(xy[~frontal, 1]):
        xy[:, 1] *= -1
    return xy


def _montage_plot(xy, values, ax, *, vmin, vmax, cmap, title, cb_label):
    """Kanalwerte als Streudiagramm im Montage-Layout."""
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=260, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.4)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.colorbar(sc, ax=ax, label=cb_label, fraction=0.046, pad=0.04)


def fig17_scalp(family="dct:0.02", constellation="baseline", noise_model="ar_irls",
                reuse=True):
    """Abb. 17 -- Gruppen-beta-Karte im Montage-Layout fuer eine Konfiguration.

    Das Ergebnis (~13 min) wird vor dem Zeichnen als NetCDF gespeichert; mit `reuse=True`
    wird eine vorhandene Datei wiederverwendet.
    """
    cache = RES / "realglm_group_map.nc"
    if reuse and cache.exists():
        ds = xr.open_dataset(cache)
        print(f"Abb. 17: nutze vorhandene Gruppenkarte {cache.name} "
              f"({ds.attrs.get('family')} / {ds.attrs.get('constellation')} / "
              f"{ds.attrs.get('noise_model')})", flush=True)
        rec = rd.load(rd.find_files()[0])
        P, _ = rd.preprocess_recording(rec, motion_method=rg.MOTION_METHOD)
        _draw_fig17(ds, P.conc, P.geo3d)
        return

    files = rd.find_files()
    print(f"Abb. 17: {family} / {constellation} / {noise_model} — "
          f"{len(files)} Dateien werden nachgerechnet ...", flush=True)
    t0 = time.time()
    prepped, geo3d, conc0 = {}, None, None
    for f in files:
        rec = rd.load(f)
        P, _ = rd.preprocess_recording(rec, motion_method=rg.MOTION_METHOD)
        prepped[f] = (P.conc, rd.stim_df(rec, rd.subject_of(f), "tapping"))
        if geo3d is None:
            geo3d, conc0 = P.geo3d, P.conc

    worker = rg.first_level
    betas = Parallel(n_jobs=rg.N_JOBS, backend="loky")(
        delayed(worker)(prepped[f][0], prepped[f][1], family, constellation,
                        noise_model, 30, 1) for f in files)

    by_sub = {}
    for f, b in zip(files, betas):
        by_sub.setdefault(rd.subject_of(f), []).append(b)
    by_sub = {s: xr.concat(v, dim="run").mean("run") for s, v in by_sub.items()}
    t, p_adj, rej, mean = rg.group_test(by_sub)
    print(f"  fertig in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Erst speichern, dann zeichnen.
    ds = xr.Dataset(dict(beta=mean, t=t, p_fdr=p_adj,
                         significant=rej.astype("int8")))
    ds.attrs.update(family=family, constellation=constellation,
                    noise_model=noise_model, n_subjects=len(by_sub))
    ds.to_netcdf(cache)
    print(f"  -> {cache}", flush=True)
    _draw_fig17(ds, conc0, geo3d)


def _draw_fig17(ds, conc, geo3d):
    xy = _montage_xy(conc, geo3d)
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for a in ax.ravel():
        a.text(0.02, 0.98, "L", transform=a.transAxes, fontsize=13, fontweight="bold",
               va="top"); a.text(0.98, 0.98, "R", transform=a.transAxes, fontsize=13,
                                 fontweight="bold", va="top", ha="right")
    for j, ch in enumerate(["HbO", "HbR"]):
        m = np.asarray(ds["beta"].sel(chromo=ch).values, float)
        lim = float(np.nanpercentile(np.abs(m), 98)) or 1.0
        _montage_plot(xy, m, ax[0, j], vmin=-lim, vmax=lim, cmap="RdBu_r",
                      title=f"{ch}: Gruppen-β über {ds.attrs.get('n_subjects', '?')} "
                            f"Probanden", cb_label="β [µM]")
        tv = np.asarray(ds["t"].sel(chromo=ch).values, float)
        tl = float(np.nanpercentile(np.abs(tv), 98)) or 1.0
        n_sig = int(np.asarray(ds["significant"].sel(chromo=ch).values).sum())
        _montage_plot(xy, tv, ax[1, j], vmin=-tl, vmax=tl, cmap="RdBu_r",
                      title=f"{ch}: t-Wert  ({n_sig} von {len(tv)} signifikant "
                            f"nach FDR)", cb_label="t")
    fig.suptitle(
        f"Reale Daten, Finger-Tapping rechte Hand — {ds.attrs.get('family')} / "
        f"{ds.attrs.get('constellation')} / {ds.attrs.get('noise_model')}\n"
        "Montage-Layout aus den Optodenpositionen; Orientierung nach Khan et al. Tab. 3 "
        "(linke Hemisphäre links, frontal oben)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "17_real_scalp.png", dpi=130)
    plt.close(fig)


def main(quick=False):
    paths.ensure()
    df = load()
    fig15_reliability(df)
    fig16_significant(df)
    fig18_plausibility(df)
    print("Abb. 15, 16, 18 erzeugt.")
    if not quick:
        fig17_scalp()
    print("\nGespeichert:", *(p.name for p in sorted(OUT.glob("1?_real_*.png"))))


if __name__ == "__main__":
    main(quick=len(sys.argv) > 1 and sys.argv[1] == "quick")
