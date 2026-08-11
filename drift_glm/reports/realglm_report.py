"""Abbildungen zu den realen Daten (Stufe 2) -- Gespraechsgrundlage.

Erzeugt aus `results/realglm_summary.csv` die Uebersichts-Abbildungen und rechnet fuer
EINE repraesentative Konfiguration die Gruppen-beta-Karte nach, damit die Aktivierung auf
dem Kopf sichtbar wird (das ist die Abbildung, die man im Gespraech zeigt).

  Abb. 15  Reproduzierbarkeit je Driftfamilie (das wahrheitsfreie Hauptkriterium)
  Abb. 16  Signifikante Kanaele nach FDR je Familie x Konstellation
  Abb. 17  Gruppen-beta-Karte auf dem Kopf (HbO/HbR), beste Familie
  Abb. 18  HbO/HbR-Plausibilitaet

Aufruf:
    conda run -n cedalion python -m drift_glm.reports.realglm_report          # alles (Abb. 17 ~13 min)
    conda run -n cedalion python -m drift_glm.reports.realglm_report quick    # ohne Abb. 17
"""

from __future__ import annotations

import importlib
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
         "bspline:5", "bspline:8", "butter:0.01", "bandpass:0.01-0.5"]


def _color(fam):
    if fam.startswith("poly"):     return "#4c72b0"
    if fam.startswith("dct"):      return "#dd8452"
    if fam.startswith("legendre"): return "#55a868"
    if fam.startswith("bspline"):  return "#8172b3"
    if fam == "none":              return "#937860"
    return "#c44e52"               # Filter-Alternativen


def load():
    df = pd.read_csv(RES / "realglm_summary.csv")
    df["degeneriert"] = [rg.is_degenerate(f, n)
                         for f, n in zip(df.family, df.noise_model)]
    return df


def fig30_reliability(df):
    """Reproduzierbarkeit je Familie -- das Hauptkriterium ohne Ground Truth."""
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
        v_plot = np.where(deg, np.nan, v)          # kaputte Zellen nicht zeichnen
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


def fig31_significant(df):
    """Wie viele Kanaele ueberstehen die FDR-Korrektur?"""
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
    fig.suptitle("Aktivierungs-Detektion auf den realen Daten: der systemische Regressor "
                 "entfernt einen großen Teil der Signifikanz")
    fig.tight_layout(); fig.savefig(OUT / "16_real_significant.png", dpi=130)
    plt.close(fig)


def fig33_plausibility(df):
    """HbR/HbO-Verhaeltnis und Antikorrelation."""
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
    """2D-Layout der Kanaele aus den Optodenpositionen.

    Cedalions `scalp_plot` braucht die Landmarken Nz/LPA/RPA, um auf den Kopf zu
    projizieren -- dieser Datensatz liefert nur die 32 Optodenpositionen, keine
    Landmarken. Deshalb hier eine Hauptkomponenten-Projektion: die Kanal-Mittelpunkte
    liegen naeherungsweise auf einer Kappenflaeche, deren zwei groesste
    Hauptkomponenten das Layout aufspannen.

    WICHTIG UND EHRLICH ZU BENENNEN: Diese Projektion ist **nicht anatomisch
    orientiert**. Ohne Landmarken laesst sich links/rechts und vorn/hinten nicht
    bestimmen. Die Abbildung zeigt also die raeumliche STRUKTUR der Aktivierung
    (fokal? verteilt?), nicht ihre Lage am Kopf. Fuer die Zuordnung zu einer
    Hirnregion braucht es entweder Landmarken oder den Bildraum.
    """
    g = geo3d.pint.dequantify() if hasattr(geo3d, "pint") else geo3d
    pos = {str(l): np.asarray(v, float) for l, v in zip(g.label.values, g.values)}
    mid = np.array([0.5 * (pos[str(s)] + pos[str(d)])
                    for s, d in zip(conc.source.values, conc.detector.values)])
    c = mid - mid.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T          # (channel, 2)


def _montage_plot(xy, values, ax, *, vmin, vmax, cmap, title, cb_label):
    """Kanalwerte als Streudiagramm im Montage-Layout."""
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=260, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.4)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.colorbar(sc, ax=ax, label=cb_label, fraction=0.046, pad=0.04)


def fig32_scalp(family="dct:0.02", constellation="baseline", noise_model="ar_irls",
                reuse=True):
    """Gruppen-beta-Karte im Montage-Layout -- rechnet die eine Konfiguration nach.

    Das Ergebnis wird als NetCDF gespeichert, BEVOR gezeichnet wird: die Rechnung kostet
    ~13 min, ein Fehler beim Zeichnen soll sie nicht vernichten. Mit `reuse=True` wird
    eine vorhandene Datei wiederverwendet.
    """
    cache = RES / "realglm_group_map.nc"
    if reuse and cache.exists():
        ds = xr.open_dataset(cache)
        print(f"Abb. 17: nutze vorhandene Gruppenkarte {cache.name} "
              f"({ds.attrs.get('family')} / {ds.attrs.get('constellation')} / "
              f"{ds.attrs.get('noise_model')})", flush=True)
        rec = rd.load(rd.find_files()[0])
        P, _ = rd.preprocess_recording(rec, motion_method=rg.MOTION_METHOD)
        _draw_fig32(ds, P.conc, P.geo3d)
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

    worker = importlib.import_module("realglm").first_level
    betas = Parallel(n_jobs=rg.N_JOBS, backend="loky")(
        delayed(worker)(prepped[f][0], prepped[f][1], family, constellation,
                        noise_model, 30, 1) for f in files)

    by_sub = {}
    for f, b in zip(files, betas):
        by_sub.setdefault(rd.subject_of(f), []).append(b)
    by_sub = {s: xr.concat(v, dim="run").mean("run") for s, v in by_sub.items()}
    t, p_adj, rej, mean = rg.group_test(by_sub)
    print(f"  fertig in {(time.time() - t0) / 60:.1f} min", flush=True)

    # ZUERST speichern, dann zeichnen -- die Rechnung ist teuer, das Zeichnen billig.
    ds = xr.Dataset(dict(beta=mean, t=t, p_fdr=p_adj,
                         significant=rej.astype("int8")))
    ds.attrs.update(family=family, constellation=constellation,
                    noise_model=noise_model, n_subjects=len(by_sub))
    ds.to_netcdf(cache)
    print(f"  -> {cache}", flush=True)
    _draw_fig32(ds, conc0, geo3d)


def _draw_fig32(ds, conc, geo3d):
    xy = _montage_xy(conc, geo3d)
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
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
        "Montage-Layout aus den Optodenpositionen (Hauptkomponenten-Projektion). "
        "Der Datensatz enthält keine Landmarken —\ndie Darstellung zeigt die räumliche "
        "Struktur der Aktivierung, ist aber NICHT anatomisch orientiert "
        "(links/rechts nicht bestimmbar).", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "17_real_scalp.png", dpi=130)
    plt.close(fig)


def main(quick=False):
    paths.ensure()
    df = load()
    fig30_reliability(df)
    fig31_significant(df)
    fig33_plausibility(df)
    print("Abb. 15, 16, 18 erzeugt.")
    if not quick:
        fig32_scalp()
    print("\nGespeichert:", *(p.name for p in sorted(OUT.glob("1?_real_*.png"))))


if __name__ == "__main__":
    main(quick=len(sys.argv) > 1 and sys.argv[1] == "quick")
