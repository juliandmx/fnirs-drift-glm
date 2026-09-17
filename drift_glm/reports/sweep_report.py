"""Auswertung/Abbildungen des Driftregressor-Sweeps (liest results/sweep_summary.csv).

Erzeugt PNGs in figures/ und results/tables.md:
   6) RMSE je Driftfamilie x Fenster (Konstellation baseline)
   7) Bias-Varianz-Zerlegung je Familie (baseline)
   8) Konstellations-Effekt (RMSE je Familie x Konstellation)
   9) HbO/HbR-Plausibilitaet (rueckgewonnenes Ratio je Familie)
  10) Motion-Correction-Achse: Bias und RMSE
  25) Variance explained (adj. R^2) je Familie x Fenster
  26) Residuen vs. GT-Abweichung (zwischen und innerhalb der Zellen)

Abb. 25/26 brauchen die Spalten r2_adj_med, resid_rms_med und resid_err_corr; auf einer
sweep_summary.csv ohne sie werden sie uebersprungen.

Aufruf: conda run -n cedalion python -m drift_glm.reports.sweep_report
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from drift_glm import figstyle, paths

RES = paths.RESULTS
OUT = paths.FIGURES

# Familien-Farben/-Reihenfolge zentral in figstyle (mehrere Module teilen sie sich).
ORDER = figstyle.FAMILY_ORDER
_color = figstyle.family_color


def _order(df):
    return [f for f in ORDER if f in df.family.values]


def _md_table(df, cols):
    """Einfache Markdown-Tabelle (ohne externe Abhaengigkeit wie tabulate)."""
    def cell(v):
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else f"{v:.3f}"
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(cell(r[c]) for c in cols) + " |"
            for _, r in df.iterrows()]
    return "\n".join([head, sep] + body)


def _write_tables(df):
    """Markdown-Ergebnistabellen -> results/tables.md."""
    d = df.copy()
    d["window_s"] = d["window_s"].astype(int)
    lines = ["# Ergebnistabellen (erzeugt von sweep_report.py)", ""]

    idx = d.groupby(["chromo", "window_s", "constellation"])["rmse_med"].idxmin()
    best = d.loc[idx, ["chromo", "window_s", "constellation", "family",
                       "rmse_med", "absbias_med", "var_med"]].sort_values(
                           ["chromo", "window_s", "constellation"])
    lines += ["## Beste Driftfamilie je Bedingung (minimaler RMSE_med)", "",
              _md_table(best, list(best.columns)), ""]

    for ch in ["HbO", "HbR"]:
        piv = (d[(d.chromo == ch) & (d.constellation == "baseline")]
               .pivot_table(index="family", columns="window_s", values="rmse_med"))
        piv = piv.reindex([f for f in ORDER if f in piv.index]).reset_index()
        piv.columns = ["family"] + [f"{int(w)}s" for w in piv.columns[1:]]
        lines += [f"## RMSE_med je Familie × Fenster — {ch} (baseline)", "",
                  _md_table(piv, list(piv.columns)), ""]

    con = (d.groupby(["chromo", "constellation"])["rmse_med"].mean()
           .reset_index().rename(columns={"rmse_med": "rmse_med_mean"}))
    lines += ["## Konstellations-Effekt (Mittel RMSE_med über Familien)", "",
              _md_table(con, list(con.columns)), ""]

    if "r2_adj_med" in d.columns:
        for ch in ["HbO", "HbR"]:
            piv = (d[(d.chromo == ch) & (d.constellation == "baseline")]
                   .pivot_table(index="family", columns="window_s",
                                values="r2_adj_med"))
            piv = piv.reindex([f for f in ORDER if f in piv.index]).reset_index()
            piv.columns = ["family"] + [f"{int(w)}s" for w in piv.columns[1:]]
            lines += [f"## Variance explained (adj. R²) je Familie × Fenster — {ch} "
                      "(baseline; Filter-Arme: R² auf der gefilterten Zeitreihe)", "",
                      _md_table(piv, list(piv.columns)), ""]
        rc = (d.groupby(["chromo", "family"])["resid_err_corr"].mean()
              .reset_index())
        lines += ["## Residuen ↔ GT-Abweichung: corr(Residual-RMS, |β̂−GT|) "
                  "über Seeds × Kanäle (Mittel über Fenster, baseline)", "",
                  _md_table(rc, list(rc.columns)), ""]

    for fname, title in [("flex_basis_summary.csv", "Flexible Recovery-Basis (Form-Treue)"),
                         ("detection_summary.csv", "Detektion nach FDR (q=0.05)")]:
        p = RES / fname
        if p.exists():
            extra = pd.read_csv(p)
            lines += [f"## {title}", "", _md_table(extra, list(extra.columns)), ""]

    (RES / "tables.md").write_text("\n".join(lines))
    print(f"-> {RES/'tables.md'}")


def main():
    paths.ensure()
    df_all = pd.read_csv(RES / "sweep_summary.csv")
    # Die Familien-/Konstellations-Abbildungen zeigen nur die erste Motion-Stufe (per
    # Konvention "wavelet"), damit sie nicht ueber zwei Vorverarbeitungen mitteln;
    # die Motion-Achse selbst zeigt Abb. 10.
    if "motion" not in df_all.columns:
        df_all = df_all.assign(motion="n/a")
    motions = list(dict.fromkeys(df_all.motion))
    ref_motion = motions[0]
    df = df_all[df_all.motion == ref_motion].copy()
    if len(motions) > 1:
        print(f"Motion-Achse: {motions} -> Abb. 6-9 zeigen '{ref_motion}'")
    wins = sorted(df.window_s.unique())
    fams = _order(df)
    colors = [_color(f) for f in fams]
    x = np.arange(len(fams))

    # ---- Abb. 6: RMSE je Familie x Fenster, Konstellation baseline ----
    # Gemeinsame y-Skala je Chromophor-Zeile, damit der Fenstereffekt ablesbar bleibt;
    # HbO und HbR getrennt, weil sie sich um etwa das Fuenffache unterscheiden.
    fig, axes = plt.subplots(2, len(wins), figsize=(6 * len(wins), 8),
                             sharex=True, sharey="row")
    for i, ch in enumerate(["HbO", "HbR"]):
        row = df[(df.chromo == ch) & (df.constellation == "baseline")]
        ymax = float(row.rmse_med.max()) * 1.08
        for j, w in enumerate(wins):
            ax = axes[i, j]
            d = row[row.window_s == w].set_index("family").loc[fams]
            ax.bar(x, d.rmse_med.values, color=colors)
            best = d.rmse_med.idxmin()
            ax.set_title(f"{ch} | window {w:g} s | baseline  (best: {best})")
            ax.set_ylabel("median RMSE [µM]")
            ax.set_ylim(0, ymax)
            ax.grid(axis="y", alpha=0.3)
    axes[-1, 0].set_xticks(x); axes[-1, 0].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    if len(wins) > 1:
        axes[-1, 1].set_xticks(x); axes[-1, 1].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    fig.suptitle("Drift-regressor comparison: RMSE of the β recovery (AR-IRLS, Monte Carlo over seeds)")
    fig.tight_layout(); fig.savefig(OUT / "06_sweep_rmse_by_family.png", dpi=130); plt.close(fig)

    # ---- Abb. 7: Bias-Varianz-Zerlegung (baseline, laengstes Fenster) ----
    w = wins[-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, ch in enumerate(["HbO", "HbR"]):
        d = df[(df.chromo == ch) & (df.window_s == w) & (df.constellation == "baseline")]
        d = d.set_index("family").loc[fams]
        ax = axes[i]
        ax.bar(x - 0.2, d.absbias_med.values, width=0.4, label="median |bias|", color="#e76f51")
        ax.bar(x + 0.2, np.sqrt(d.var_med.values), width=0.4, label="median SD (√var)", color="#457b9d")
        ax.set_title(f"{ch} | window {w:g} s | baseline")
        ax.set_ylabel("[µM]"); ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.suptitle("Bias-variance decomposition per drift family")
    fig.tight_layout(); fig.savefig(OUT / "07_sweep_bias_var.png", dpi=130); plt.close(fig)

    # ---- Abb. 8: Konstellations-Effekt (RMSE je Familie x Konstellation) ----
    # Konstellationen aus den Daten nehmen, feste Reihenfolge nur fuer die Darstellung.
    order = ["baseline", "motion", "global", "short_avg", "short_maxcorr", "short_closest"]
    present = list(df.constellation.unique())
    cons = [c for c in order if c in present] + [c for c in present if c not in order]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    width = 0.8 / max(len(cons), 1)
    for i, ch in enumerate(["HbO", "HbR"]):
        ax = axes[i]
        for k, con in enumerate(cons):
            d = df[(df.chromo == ch) & (df.window_s == w) & (df.constellation == con)]
            d = d.set_index("family").reindex(fams)
            ax.bar(x + (k - (len(cons) - 1) / 2) * width, d.rmse_med.values,
                   width=width * 0.92, label=con)
        ax.set_title(f"{ch} | window {w:g} s")
        ax.set_ylabel("median RMSE [µM]"); ax.grid(axis="y", alpha=0.3)
        ax.legend(ncol=min(len(cons), 5), fontsize=8)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    fig.suptitle("Effect of the regressor constellation on the β recovery: RMSE per drift "
                 "family and constellation")
    fig.tight_layout(); fig.savefig(OUT / "08_sweep_constellation_effect.png", dpi=130); plt.close(fig)

    # ---- Abb. 9: HbO/HbR-Plausibilitaet (rueckgew. Ratio) ----
    fig, axes = plt.subplots(1, len(wins), figsize=(6 * len(wins), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for j, wv in enumerate(wins):
        ax = axes[j]
        d = df[(df.chromo == "HbO") & (df.window_s == wv) & (df.constellation == "baseline")]
        d = d.set_index("family").loc[fams]
        ax.bar(x, d.hbr_hbo_ratio_med.values, color=colors)
        ax.axhline(-0.4, color="k", ls="--", lw=1.5, label="true ratio (−0.4)")
        ax.set_title(f"window {wv:g} s | baseline")
        ax.set_ylabel("median(β_HbR / β_HbO)")
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.suptitle("HbO/HbR plausibility: recovered amplitude ratio (target −0.4)")
    fig.tight_layout(); fig.savefig(OUT / "09_sweep_plausibility.png", dpi=130); plt.close(fig)

    # ---- Abb. 10: Motion-Achse -- Bias und RMSE, getrennt nach Chromophor ----
    # Bias und RMSE zusammen: TDDR daempft die eingemischte HRF (~70 %) und kompensiert so
    # die systemische HbO-Ueberschaetzung -- der RMSE sinkt, ohne dass besser geschaetzt wird.
    if len(motions) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        width = 0.8 / len(motions)
        for i, ch in enumerate(["HbO", "HbR"]):
            ax = axes[i]
            for k, mm in enumerate(motions):
                d = df_all[(df_all.chromo == ch) & (df_all.motion == mm)
                           & (df_all.constellation == "baseline")]
                g = d.groupby("family")[["bias_med", "rmse_med"]].mean()
                g = g.reindex([f for f in fams if f in g.index])
                pos = np.arange(len(g)) + (k - (len(motions) - 1) / 2) * width
                ax.bar(pos, g.bias_med.values, width=width * 0.92,
                       label=f"{mm} · bias", alpha=0.85)
                ax.plot(pos, g.rmse_med.values, "k_", markersize=7,
                        label="RMSE" if k == 0 else None)
            truth = float(df_all[df_all.chromo == ch].beta_true_peak.iloc[0])
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{ch} | baseline | truth {truth:+.3f} µM")
            ax.set_ylabel("median bias [µM]  (ticks: median RMSE)")
            ax.set_xticks(np.arange(len(g)))
            ax.set_xticklabels(g.index, rotation=60, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Motion correction as an axis: bias and RMSE per drift family")
        fig.tight_layout(); fig.savefig(OUT / "10_sweep_motion_axis.png", dpi=130)
        plt.close(fig)

        print("\n=== Motion-Achse (Mittel über Familien × Fenster × Konstellationen) ===")
        print(df_all.groupby(["chromo", "motion"])[["bias_med", "rmse_med"]].mean()
              .to_string(float_format=lambda v: f"{v:+.4f}"))

    # ---- Abb. 25: Variance explained (adj. R^2) je Familie x Fenster ----
    # Adjustiertes R^2, weil die Familien verschieden viele Spalten haben. Die Filter-Arme
    # sind schraffiert: ihr R^2 bezieht sich auf die gefilterte Zeitreihe.
    if "r2_adj_med" in df.columns:
        fig, axes = plt.subplots(2, len(wins), figsize=(6 * len(wins), 8),
                                 sharex=True, sharey="row")
        axes = np.atleast_2d(axes)
        for i, ch in enumerate(["HbO", "HbR"]):
            row = df[(df.chromo == ch) & (df.constellation == "baseline")]
            for j, wv in enumerate(wins):
                ax = axes[i, j]
                d = row[row.window_s == wv].set_index("family").loc[fams]
                hatches = ["//" if f.startswith(("butter", "lowpass", "bandpass"))
                           else None for f in fams]
                bars = ax.bar(x, d.r2_adj_med.values, color=colors)
                for bar, h in zip(bars, hatches):
                    if h:
                        bar.set_hatch(h)
                ax.set_title(f"{ch} | window {wv:g} s | baseline")
                ax.set_ylabel("median adjusted R²")
                # Negative Werte mitzeigen (unter AR-IRLS bei kurzen Fenstern normal, R^2
                # im Rohdatenraum nach Prewhitening); Boden bei -0.5, Extremwerte laufen aus.
                ax.set_ylim(-0.5, 1)
                ax.axhline(0, color="k", lw=0.8)
                ax.grid(axis="y", alpha=0.3)
        for j in range(len(wins)):
            axes[-1, j].set_xticks(x)
            axes[-1, j].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
        fig.suptitle("Variance explained by the GLM (adjusted R²) per drift family — "
                     "hatched: filter arms (R² on the filtered series)")
        fig.tight_layout()
        fig.savefig(OUT / "25_sweep_r2.png", dpi=130)
        plt.close(fig)
    else:
        print("(Abb. 25 uebersprungen: sweep_summary.csv ohne r2_adj_med -- "
              "Sweep neu laufen lassen)")

    # ---- Abb. 26: Residuen vs. Abweichung von der Ground Truth ----
    # Links zwischen den Modellen (ein Punkt je Zelle Familie x Fenster), rechts innerhalb
    # der Zellen (Korrelation ueber Seeds x Kanaele aus dem Sweep).
    if "resid_rms_med" in df.columns:
        from scipy import stats as sstats
        markers = {90.0: "o", 180.0: "s", 368.0: "^"}
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for i, ch in enumerate(["HbO", "HbR"]):
            d = df[(df.chromo == ch) & (df.constellation == "baseline")]
            ax = axes[i, 0]
            for f in fams:
                for wv in wins:
                    r = d[(d.family == f) & (d.window_s == wv)]
                    if r.empty:
                        continue
                    ax.scatter(r.resid_rms_med, r.rmse_med, s=45, color=_color(f),
                               marker=markers.get(wv, "o"), edgecolor="white",
                               linewidth=0.6, zorder=3)
            rho = sstats.spearmanr(d.resid_rms_med, d.rmse_med)
            ax.set_title(f"{ch}: between models (baseline) — "
                         f"Spearman ρ = {rho.statistic:+.2f}")
            # Log-Skala: die AR-IRLS-Instabilitaet roher Polynome bei kurzen Fenstern
            # erzeugt Residual-Ausreisser (bis ~40 µM), die sonst alles stauchen.
            ax.set_xscale("log")
            ax.set_xlabel("residual RMS [µM] (median, log scale)")
            ax.set_ylabel("RMSE of β̂ vs. truth [µM] (median)")
            ax.grid(alpha=0.3, which="both")
            ax = axes[i, 1]
            g = (d.groupby("family")["resid_err_corr"].mean().reindex(fams))
            ax.bar(x, g.values, color=colors)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{ch}: within cells (correlation over seeds × channels)")
            ax.set_ylabel("corr(residual RMS, |β̂ − truth|)")
            ax.set_ylim(-1, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        handles = [plt.Line2D([], [], color="0.4", marker=m, ls="", label=f"{int(w)} s")
                   for w, m in markers.items() if w in wins]
        axes[0, 0].legend(handles=handles, title="window", fontsize=8)
        fig.suptitle("Residual RMS against deviation from the ground truth\n"
                     "(colours = drift families as in the other sweep figures)")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUT / "26_sweep_resid_vs_error.png", dpi=130)
        plt.close(fig)
    else:
        print("(Abb. 26 uebersprungen: sweep_summary.csv ohne resid_rms_med)")

    # ---- Text-Zusammenfassung ----
    print("=== Ranking nach RMSE_med (Mittel über baseline+motion, je chromo × Fenster) ===")
    clean = df[df.constellation.isin(["baseline", "motion"])]
    for ch in ["HbO", "HbR"]:
        for wv in wins:
            g = (clean[(clean.chromo == ch) & (clean.window_s == wv)]
                 .groupby("family")["rmse_med"].mean().sort_values())
            top = " > ".join(f"{f}({v:.3f})" for f, v in g.head(4).items())
            print(f"  {ch} {wv:g}s:  {top}")
    _write_tables(df)
    print("\nGespeichert:", *(p.name for p in sorted(OUT.glob('0?_sweep_*.png'))))


if __name__ == "__main__":
    main()
