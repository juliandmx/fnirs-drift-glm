"""Auswertung/Abbildungen des Driftregressor-Sweeps (liest results/).

Erzeugt PNGs in figures/:
  10) RMSE je Driftfamilie x Fenster (Konstellation baseline)        -> Kernresultat
  11) Bias-Varianz-Zerlegung je Familie (baseline)
  12) Konstellations-Effekt (RMSE je Familie x Konstellation)        -> Global-Artefakt
  13) HbO/HbR-Plausibilitaet (rueckgew. Ratio je Familie)

Aufruf: conda run -n cedalion python sweep_report.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "figures"

ORDER = ["none", "poly:1", "poly:2", "poly:3", "poly:4", "poly:5",
         "dct:0.005", "dct:0.01", "dct:0.02",
         "legendre:1", "legendre:3", "legendre:5",
         "bspline:5", "bspline:8", "butter:0.01"]


def _color(fam):
    if fam.startswith("poly"):
        return plt.cm.Blues(0.4 + 0.12 * int(fam.split(":")[1]))
    if fam.startswith("dct"):
        return {"dct:0.005": "#f4a259", "dct:0.01": "#e76f51", "dct:0.02": "#bc4749"}[fam]
    if fam.startswith("legendre"):
        return plt.cm.Greens(0.4 + 0.15 * int(fam.split(":")[1]))
    if fam.startswith("bspline"):
        return {"bspline:5": "#8d6e63", "bspline:8": "#4e342e"}.get(fam, "#795548")
    return {"none": "0.6", "butter:0.01": "#8338ec"}[fam]


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

    for fname, title in [("flex_basis_summary.csv", "Flexible Recovery-Basis (Form-Treue)"),
                         ("detection_summary.csv", "Detektion nach FDR (q=0.05)")]:
        p = RES / fname
        if p.exists():
            extra = pd.read_csv(p)
            lines += [f"## {title}", "", _md_table(extra, list(extra.columns)), ""]

    (RES / "tables.md").write_text("\n".join(lines))
    print(f"-> {RES/'tables.md'}")


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(RES / "sweep_summary.csv")
    wins = sorted(df.window_s.unique())
    fams = _order(df)
    colors = [_color(f) for f in fams]
    x = np.arange(len(fams))

    # ---- Abb. 10: RMSE je Familie x Fenster, Konstellation baseline ----
    fig, axes = plt.subplots(2, len(wins), figsize=(6 * len(wins), 8), sharex=True)
    for i, ch in enumerate(["HbO", "HbR"]):
        for j, w in enumerate(wins):
            ax = axes[i, j]
            d = df[(df.chromo == ch) & (df.window_s == w) & (df.constellation == "baseline")]
            d = d.set_index("family").loc[fams]
            ax.bar(x, d.rmse_med.values, color=colors)
            best = d.rmse_med.idxmin()
            ax.set_title(f"{ch} | Fenster {w:g}s | baseline  (best: {best})")
            ax.set_ylabel("RMSE_med [µM]")
            ax.grid(axis="y", alpha=0.3)
    axes[-1, 0].set_xticks(x); axes[-1, 0].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    if len(wins) > 1:
        axes[-1, 1].set_xticks(x); axes[-1, 1].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    fig.suptitle("Driftregressor-Vergleich: RMSE der β-Rückgewinnung (AR-IRLS, MC über Seeds)")
    fig.tight_layout(); fig.savefig(OUT / "10_sweep_rmse_by_family.png", dpi=130); plt.close(fig)

    # ---- Abb. 11: Bias-Varianz-Zerlegung (baseline, laengstes Fenster) ----
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
    fig.tight_layout(); fig.savefig(OUT / "11_sweep_bias_var.png", dpi=130); plt.close(fig)

    # ---- Abb. 12: Konstellations-Effekt (RMSE je Familie x Konstellation) ----
    cons = ["baseline", "motion", "global", "motion+global"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    width = 0.2
    for i, ch in enumerate(["HbO", "HbR"]):
        ax = axes[i]
        for k, con in enumerate(cons):
            d = df[(df.chromo == ch) & (df.window_s == w) & (df.constellation == con)]
            d = d.set_index("family").loc[fams]
            ax.bar(x + (k - 1.5) * width, d.rmse_med.values, width=width, label=con)
        ax.set_title(f"{ch} | Fenster {w:g}s")
        ax.set_ylabel("RMSE_med [µM]"); ax.grid(axis="y", alpha=0.3); ax.legend(ncol=4)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(fams, rotation=60, ha="right", fontsize=8)
    fig.suptitle("Konstellations-Effekt auf die β-Rückgewinnung: systemischer "
                 "Global-Regressor senkt den HbO-RMSE deutlich\n(entfernt geteilte "
                 "Physiologie im fokalen Aktivierungscluster); Motion ~ohne Wirkung")
    fig.tight_layout(); fig.savefig(OUT / "12_sweep_constellation_effect.png", dpi=130); plt.close(fig)

    # ---- Abb. 13: HbO/HbR-Plausibilitaet (rueckgew. Ratio) ----
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
    fig.tight_layout(); fig.savefig(OUT / "13_sweep_plausibility.png", dpi=130); plt.close(fig)

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
    print("\nGespeichert:", *(p.name for p in sorted(OUT.glob('1?_sweep_*.png'))))


if __name__ == "__main__":
    main()
