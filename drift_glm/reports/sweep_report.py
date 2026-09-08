"""Auswertung/Abbildungen des Driftregressor-Sweeps (liest results/).

Erzeugt PNGs in figures/:
   6) RMSE je Driftfamilie x Fenster (Konstellation baseline)        -> Kernresultat
   7) Bias-Varianz-Zerlegung je Familie (baseline)
   8) Konstellations-Effekt (RMSE je Familie x Konstellation)
   9) HbO/HbR-Plausibilitaet (rueckgew. Ratio je Familie)
  10) Motion-Correction-Achse: Bias UND RMSE (der RMSE allein taeuscht)
  25) Variance explained (adj. R^2) je Familie x Fenster             -> Modellfit-Guete
  26) Residuen vs. GT-Abweichung: prueft die Erwartung aus den Gespraechsnotizen
      2026-09-08 (kleine Residuen <-> kleine Abweichung von der Ground Truth)

Abb. 25/26 brauchen die Metrik-Spalten aus dem Sweep-Re-Run (r2_adj_med, resid_rms_med,
resid_err_corr); auf einer aelteren sweep_summary.csv werden sie uebersprungen.

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
    """Thesis-taugliche Markdown-Ergebnistabellen -> results/tables.md."""
    d = df.copy()
    d["window_s"] = d["window_s"].astype(int)
    lines = ["# Ergebnistabellen (auto-generiert von sweep_report.py)", ""]

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
    # Seit v4 ist die Motion Correction eine eigene Achse. Die Familien-/Konstellations-
    # Abbildungen zeigen EINE Stufe (die erste, per Konvention die driftneutrale
    # "wavelet"), damit sie nicht ueber zwei Vorverarbeitungen hinweg mitteln. Die
    # Motion-Achse selbst bekommt eine eigene Abbildung.
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
    # Gemeinsame y-Skala je Chromophor-Zeile (Betreuungsvorgabe "ueber alle bilder
    # gleiche skala"): nur so ist der Effekt der FENSTERLAENGE ablesbar -- bei
    # teilbildweiser Autoskalierung sehen 90 s und 368 s gleich schlecht aus, obwohl
    # sich der Fehler halbiert. HbO und HbR bekommen getrennte Skalen, weil sie sich um
    # etwa das Fuenffache unterscheiden und HbR sonst zu einer flachen Linie wuerde.
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
            ax.set_title(f"{ch} | Fenster {w:g}s | baseline  (best: {best})")
            ax.set_ylabel("RMSE_med [µM]")
            ax.set_ylim(0, ymax)
            ax.grid(axis="y", alpha=0.3)
    axes[-1, 0].set_xticks(x); axes[-1, 0].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    if len(wins) > 1:
        axes[-1, 1].set_xticks(x); axes[-1, 1].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    fig.suptitle("Driftregressor-Vergleich: RMSE der β-Rückgewinnung (AR-IRLS, MC über Seeds)")
    fig.tight_layout(); fig.savefig(OUT / "06_sweep_rmse_by_family.png", dpi=130); plt.close(fig)

    # ---- Abb. 7: Bias-Varianz-Zerlegung (baseline, laengstes Fenster) ----
    w = wins[-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, ch in enumerate(["HbO", "HbR"]):
        d = df[(df.chromo == ch) & (df.window_s == w) & (df.constellation == "baseline")]
        d = d.set_index("family").loc[fams]
        ax = axes[i]
        ax.bar(x - 0.2, d.absbias_med.values, width=0.4, label="|Bias|_med", color="#e76f51")
        ax.bar(x + 0.2, np.sqrt(d.var_med.values), width=0.4, label="Std_med (√Var)", color="#457b9d")
        ax.set_title(f"{ch} | Fenster {w:g}s | baseline")
        ax.set_ylabel("[µM]"); ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.suptitle("Bias-Varianz-Zerlegung je Driftfamilie")
    fig.tight_layout(); fig.savefig(OUT / "07_sweep_bias_var.png", dpi=130); plt.close(fig)

    # ---- Abb. 8: Konstellations-Effekt (RMSE je Familie x Konstellation) ----
    # Konstellationen aus den Daten nehmen, nicht hartkodiert: das Raster aendert sich
    # zwischen den Sweep-Versionen (v4 ersetzt "motion+global" durch die echten
    # Short-Channel-Varianten).
    order = ["baseline", "motion", "global", "motion+global",
             "short_avg", "short_maxcorr", "short_closest"]
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
        ax.set_title(f"{ch} | Fenster {w:g}s")
        ax.set_ylabel("RMSE_med [µM]"); ax.grid(axis="y", alpha=0.3)
        ax.legend(ncol=min(len(cons), 5), fontsize=8)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    fig.suptitle("Konstellations-Effekt auf die β-Rückgewinnung: ein systemischer "
                 "Regressor (Short-Channel oder Global) senkt den HbO-Fehler deutlich;\n"
                 "Motion-Regressoren bleiben ~ohne Wirkung")
    fig.tight_layout(); fig.savefig(OUT / "08_sweep_constellation_effect.png", dpi=130); plt.close(fig)

    # ---- Abb. 9: HbO/HbR-Plausibilitaet (rueckgew. Ratio) ----
    fig, axes = plt.subplots(1, len(wins), figsize=(6 * len(wins), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for j, wv in enumerate(wins):
        ax = axes[j]
        d = df[(df.chromo == "HbO") & (df.window_s == wv) & (df.constellation == "baseline")]
        d = d.set_index("family").loc[fams]
        ax.bar(x, d.hbr_hbo_ratio_med.values, color=colors)
        ax.axhline(-0.4, color="k", ls="--", lw=1.5, label="wahres Ratio (−0.4)")
        ax.set_title(f"Fenster {wv:g}s | baseline")
        ax.set_ylabel("median(β_HbR / β_HbO)")
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.suptitle("HbO/HbR-Plausibilität: rückgewonnenes Amplituden-Ratio (Ziel −0.4)")
    fig.tight_layout(); fig.savefig(OUT / "09_sweep_plausibility.png", dpi=130); plt.close(fig)

    # ---- Abb. 10: Motion-Achse -- Bias UND RMSE, getrennt nach Chromophor ----
    # Eigene Abbildung, weil hier der RMSE allein in die Irre führt: TDDR dämpft die
    # eingemischte HRF auf ~70 % und kompensiert damit zufällig die systemisch bedingte
    # HbO-Überschätzung. Der RMSE sinkt, obwohl nicht besser geschätzt wird. Sichtbar
    # wird das erst am Bias — und daran, dass HbR (ohne Überschätzung) sich verschlechtert.
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
                       label=f"{mm} · Bias", alpha=0.85)
                ax.plot(pos, g.rmse_med.values, "k_", markersize=7,
                        label="RMSE" if k == 0 else None)
            truth = float(df_all[df_all.chromo == ch].beta_true_peak.iloc[0])
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{ch} | baseline | Wahrheit {truth:+.3f} µM")
            ax.set_ylabel("Bias_med [µM]  (Striche: RMSE_med)")
            ax.set_xticks(np.arange(len(g)))
            ax.set_xticklabels(g.index, rotation=60, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Motion Correction als Achse: Bias verrät, was der RMSE verdeckt")
        fig.tight_layout(); fig.savefig(OUT / "10_sweep_motion_axis.png", dpi=130)
        plt.close(fig)

        print("\n=== Motion-Achse (Mittel über Familien × Fenster × Konstellationen) ===")
        print(df_all.groupby(["chromo", "motion"])[["bias_med", "rmse_med"]].mean()
              .to_string(float_format=lambda v: f"{v:+.4f}"))

    # ---- Abb. 25: Variance explained (adj. R^2) je Familie x Fenster ----
    # Modellfit-Guete OHNE Ground Truth -- die Metrik, die es auch auf realen Daten
    # gibt. Adjustiert, weil die Familien verschieden viele Spalten haben (dct:0.02
    # hat bei 368 s ein Vielfaches von poly:1 -- unadjustiert gewinnt sonst mechanisch
    # die groesste Designmatrix). butter ist markiert: sein R^2 bezieht sich auf die
    # GEFILTERTE Zeitreihe, ein Teil der Varianz ist dort schon entfernt.
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
                ax.set_title(f"{ch} | Fenster {wv:g}s | baseline")
                ax.set_ylabel("median adj. R²")
                ax.set_ylim(0, 1)
                ax.grid(axis="y", alpha=0.3)
        for j in range(len(wins)):
            axes[-1, j].set_xticks(x)
            axes[-1, j].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
        fig.suptitle("Variance explained des GLM (adj. R²) je Driftfamilie — "
                     "schraffiert: Filter-Arme (R² auf der gefilterten Zeitreihe)")
        fig.tight_layout()
        fig.savefig(OUT / "25_sweep_r2.png", dpi=130)
        plt.close(fig)
    else:
        print("(Abb. 25 uebersprungen: sweep_summary.csv ohne r2_adj_med -- "
              "Sweep neu laufen lassen)")

    # ---- Abb. 26: Residuen vs. Abweichung von der Ground Truth ----
    # Die pruefbare Erwartung: je kleiner die Residuen, desto kleiner die Abweichung.
    # Links ZWISCHEN den Modellen (jeder Punkt eine Zelle: Familie x Fenster),
    # rechts INNERHALB der Zellen (Korrelation ueber Seeds x Kanaele, aus dem Sweep).
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
            ax.set_title(f"{ch}: zwischen den Modellen (baseline) — "
                         f"Spearman ρ = {rho.statistic:+.2f}")
            ax.set_xlabel("Residual-RMS [µM] (median)")
            ax.set_ylabel("RMSE β̂ vs. GT [µM] (median)")
            ax.grid(alpha=0.3)
            ax = axes[i, 1]
            g = (d.groupby("family")["resid_err_corr"].mean().reindex(fams))
            ax.bar(x, g.values, color=colors)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{ch}: innerhalb der Zellen (Korrelation über Seeds × Kanäle)")
            ax.set_ylabel("corr(Residual-RMS, |β̂ − GT|)")
            ax.set_ylim(-1, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        handles = [plt.Line2D([], [], color="0.4", marker=m, ls="", label=f"{int(w)} s")
                   for w, m in markers.items() if w in wins]
        axes[0, 0].legend(handles=handles, title="Fenster", fontsize=8)
        fig.suptitle("Erwartung geprüft: kleinere Residuen ↔ kleinere Abweichung von "
                     "der Ground Truth?\n(Farben = Driftfamilien wie in Abb. 6–10)")
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
