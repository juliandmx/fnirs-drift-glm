"""Pipeline-Flowchart als Abbildung fuer die Arbeit (Abb. 00; englische Beschriftung).

Erzeugt figures/00_pipeline_flowchart.pdf (Vektor, fuer LaTeX) und .png (Vorschau) mit
reinem matplotlib, ohne Zusatzabhaengigkeit. Gezeigt werden beide Straenge (Simulation mit
Ground Truth auf nn22; Realdaten Khan/Multisubject) durch die gemeinsame Vorverarbeitung
und das GLM bis zu den jeweiligen Auswertungsmetriken; die Injektion der synthetischen
Aktivierung ist als Seitenpfad in die optische Dichte vor der Motion Correction gezeichnet.

Aufruf: conda run -n cedalion python -m drift_glm.reports.flowchart
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from drift_glm import paths

OUT = paths.FIGURES

# Ein Farbton je Schritt-Kategorie (gedeckt, druckfaehig; Text bleibt dunkel).
C_DATA = "#e8eef7"      # Datenquellen
C_PREP = "#e6f2ea"      # Vorverarbeitung
C_GT = "#fdecec"        # Ground Truth / Injektion
C_GLM = "#fff4e0"       # Modell
C_EVAL = "#f1e8f7"      # Auswertung
EDGE = "#4a4a4a"


def _box(ax, x, y, w, h, text, fc, fontsize=8.3, bold_first=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.9, edgecolor=EDGE, facecolor=fc, zorder=2))
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x + w / 2, y + h - 0.012, lines[0], ha="center", va="top",
                fontsize=fontsize, fontweight="bold", zorder=3, wrap=True)
        ax.text(x + w / 2, y + h - 0.012 - 0.030, "\n".join(lines[1:]), ha="center",
                va="top", fontsize=fontsize - 0.8, zorder=3, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, zorder=3, linespacing=1.3)


def _arrow(ax, p0, p1, label=None, style="-|>", ls="-", label_dx=0.012):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=11, linewidth=1.1,
        linestyle=ls, color=EDGE, zorder=1,
        shrinkA=0, shrinkB=0))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + label_dx, (p0[1] + p1[1]) / 2, label,
                ha="left", va="center", fontsize=7.3, color="#333333",
                style="italic", zorder=3)


def main():
    paths.ensure()
    fig, ax = plt.subplots(figsize=(9.6, 12.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    xl, xr = 0.03, 0.52          # linke/rechte Spalte
    wl = wr = 0.45
    xm, wm = 0.03, 0.94          # breite (gemeinsame) Boxen

    # --- row 1: data sources ------------------------------------------------------
    _box(ax, xl, 0.925, wl, 0.062,
         "Strand A — simulation (ground truth known)\n"
         "nn22 resting state: 520 channels, ≈9 Hz, 368 s\n"
         "analysis windows 90 / 180 / 368 s, several seeds", C_DATA)
    _box(ax, xr, 0.925, wr, 0.062,
         "Strand B — real data (no ground truth)\n"
         "Khan: 25 subjects, 48 channels · multisubject:\n"
         "5 subjects, 28 channels, 8 genuine short channels", C_DATA)

    # --- side path: ground truth / injection (strand A only) -----------------------
    _box(ax, xl, 0.790, wl, 0.100,
         "Synthetic activation (ground truth)\n"
         "Gaussian blob on the cortex beneath C3/C4 (ICBM152)\n"
         "→ forward model (sensitivity A) → channel space\n"
         "→ converted back to optical density\n"
         "β_true known per channel and on the cortex", C_GT)

    # --- shared preprocessing ------------------------------------------------------
    _box(ax, xm, 0.700, wm, 0.052,
         "Intensity → optical density (OD)\n"
         "Strand A: activation injected into the OD before motion correction",
         C_PREP)
    _box(ax, xm, 0.628, wm, 0.042,
         "Motion correction on OD\n"
         "TDDR and/or wavelet (separate comparison axis)", C_PREP)
    _box(ax, xm, 0.556, wm, 0.042,
         "Channel pruning and concentration\n"
         "SNR, amplitude and distance masks → Beer–Lambert → Δ[HbO], Δ[HbR] in µM",
         C_PREP)

    # --- GLM ------------------------------------------------------------------------
    _box(ax, xm, 0.408, wm, 0.118,
         "GLM per channel × chromophore\n"
         "Design matrix  =  HRF regressor (gamma, peak = 1; shape analysis: flexible"
         " Gaussian basis)\n"
         "⊕  drift family (independent variable): polynomial n = 1…5 · Legendre 1/3/5 ·"
         " DCT 0.005/0.01/0.02 Hz · B-spline 5/8 · none\n"
         "     alternative to regressors: Butterworth high-pass (or low-/band-pass) on the"
         " time series; control: data and design filtered alike\n"
         "⊕  systemic: global mean · short channel (avg / max-corr / closest; in the DM"
         " or subtracted beforehand)   ⊕  motion regressors\n"
         "Estimators: AR-IRLS (AR order up to 30) and OLS", C_GLM)

    # --- evaluation -----------------------------------------------------------------
    _box(ax, xl, 0.235, wl, 0.130,
         "Evaluation, strand A (against the ground truth)\n"
         "RMSE / bias / variance of β̂ (over seeds)\n"
         "scalp plots: truth next to estimate, deviation\n"
         "detection: t-test per channel + FDR (sensitivity/specificity)\n"
         "HRF shape: flexible basis vs. injected HRF\n"
         "model fit: R², residuals (↔ deviation from the truth)\n"
         "image space: reconstruction → location of the maximum", C_EVAL)
    _box(ax, xr, 0.235, wr, 0.130,
         "Evaluation, strand B (truth-free criteria)\n"
         "run-to-run / split-half reproducibility of the β maps\n"
         "contralateral expectation: left-hand tapping → C4,\n"
         "right-hand tapping → C3 (channel and image space)\n"
         "model fit: R², residuals\n"
         "HRF curves per drift family (flexible basis)\n"
         "against the block average of the data", C_EVAL)

    # --- outcome --------------------------------------------------------------------
    _box(ax, xm, 0.125, wm, 0.052,
         "Comparison of the drift-regressor families\n"
         "per window length and regressor constellation → recommendation for practice",
         C_EVAL)

    # --- arrows ---------------------------------------------------------------------
    a_x = xl + wl / 2
    b_x = xr + wr / 2
    _arrow(ax, (a_x, 0.925), (a_x, 0.890))                     # A: data -> truth/injection
    _arrow(ax, (a_x, 0.790), (a_x, 0.752),
           label="injection into the OD")                      # truth -> OD stage
    _arrow(ax, (b_x, 0.925), (b_x, 0.752))                     # B: data -> OD stage
    for y0, y1 in [(0.700, 0.670), (0.628, 0.598), (0.556, 0.526)]:
        _arrow(ax, (0.5, y0), (0.5, y1))
    _arrow(ax, (a_x, 0.408), (a_x, 0.365))                     # GLM -> evaluation A
    _arrow(ax, (b_x, 0.408), (b_x, 0.365))                     # GLM -> evaluation B
    _arrow(ax, (a_x, 0.235), (a_x, 0.177))
    _arrow(ax, (b_x, 0.235), (b_x, 0.177))

    ax.set_title("Processing pipeline: simulation (strand A) and real data (strand B)\n"
                 "with shared preprocessing and shared GLM",
                 fontsize=11.5, pad=14)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        p = OUT / f"00_pipeline_flowchart.{suffix}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"-> {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
