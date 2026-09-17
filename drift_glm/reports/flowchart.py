"""Pipeline-Flowchart als Abbildung fuer die Arbeit (Abb. 00).

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

    # --- Zeile 1: Datenquellen -------------------------------------------------
    _box(ax, xl, 0.925, wl, 0.062,
         "Strang A — Simulation (Ground Truth bekannt)\n"
         "nn22-Ruhedaten: 544 Kanäle, ≈9 Hz, 368 s\n"
         "Analysefenster 90 / 180 / 368 s, mehrere Seeds", C_DATA)
    _box(ax, xr, 0.925, wr, 0.062,
         "Strang B — Realdaten (ohne Ground Truth)\n"
         "Khan: 25 Probanden, 48 Kanäle · Multisubject:\n"
         "5 Probanden, 28 Kanäle, 8 echte Short-Channels", C_DATA)

    # --- Seitenpfad: Ground Truth / Injektion (nur Strang A) --------------------
    _box(ax, xl, 0.790, wl, 0.100,
         "Synthetische Aktivierung (Ground Truth)\n"
         "Gauß-Blob auf dem Kortex unter C3/C4 (ICBM152)\n"
         "→ Vorwärtsmodell (Sensitivität A) → Kanalraum\n"
         "→ Rückrechnung in optische Dichte\n"
         "β_true je Kanal und auf dem Kortex bekannt", C_GT)

    # --- Gemeinsame Vorverarbeitung ---------------------------------------------
    _box(ax, xm, 0.700, wm, 0.052,
         "Intensität → optische Dichte (OD)\n"
         "Strang A: Injektion der Aktivierung in die OD, vor der Motion Correction",
         C_PREP)
    _box(ax, xm, 0.628, wm, 0.042,
         "Motion Correction auf OD\n"
         "TDDR und/oder Wavelet (eigene Vergleichsachse)", C_PREP)
    _box(ax, xm, 0.556, wm, 0.042,
         "Kanal-Pruning und Konzentration\n"
         "SNR-, Amplituden-, Abstandsmaske → Beer-Lambert → Δ[HbO], Δ[HbR] in µM",
         C_PREP)

    # --- GLM ---------------------------------------------------------------------
    _box(ax, xm, 0.408, wm, 0.118,
         "GLM je Kanal × Chromophor\n"
         "Designmatrix  =  HRF-Regressor (Gamma, Peak = 1; Formanalyse: flexible"
         " Gauß-Basis)\n"
         "⊕  Driftfamilie (unabhängige Variable): Polynom n=1…5 · Legendre 1/3/5 ·"
         " DCT 0,005/0,01/0,02 Hz · B-Spline 5/8 · none\n"
         "     Alternative statt Regressoren: Butterworth-Hochpass (bzw. Tief-/Bandpass)"
         " auf der Zeitreihe\n"
         "⊕  Systemik: Global-Mean · Short-Channel (avg / maxcorr / closest; in der DM"
         " oder vorab abgezogen)   ⊕  Motion-Regressoren\n"
         "Schätzer: AR-IRLS (AR-Ordnung 30) und OLS", C_GLM)

    # --- Auswertung ---------------------------------------------------------------
    _box(ax, xl, 0.235, wl, 0.130,
         "Auswertung Strang A (gegen Ground Truth)\n"
         "RMSE / Bias / Varianz von β̂ (über Seeds)\n"
         "Scalp-Plots: GT neben Schätzung, Abweichung\n"
         "Detektion: t-Test je Kanal + FDR (Sensitivität/Spezifität)\n"
         "HRF-Form: flexible Basis vs. injizierte HRF\n"
         "Modellfit: R², Residuen (↔ GT-Abweichung)\n"
         "Bildraum: Rekonstruktion → Ort des Maximums", C_EVAL)
    _box(ax, xr, 0.235, wr, 0.130,
         "Auswertung Strang B (wahrheitsfreie Kriterien)\n"
         "Split-Half-Reproduzierbarkeit der β-Karten\n"
         "Kontralaterale Erwartung: Tapping links → C4,\n"
         "rechts → C3 (Kanalraum und Bildraum)\n"
         "Modellfit: R², Residuen\n"
         "HRF-Kurven je Driftfamilie (flexible Basis)\n"
         "gegen das Block-Mittel der Daten", C_EVAL)

    # --- Ergebnis ------------------------------------------------------------------
    _box(ax, xm, 0.125, wm, 0.052,
         "Vergleich der Driftregressor-Familien\n"
         "je Fensterlänge und Regressor-Konstellation → Nutzungsempfehlung",
         C_EVAL)

    # --- Pfeile ---------------------------------------------------------------------
    a_x = xl + wl / 2
    b_x = xr + wr / 2
    _arrow(ax, (a_x, 0.925), (a_x, 0.890))                     # A: Daten -> GT/Injektion
    _arrow(ax, (a_x, 0.790), (a_x, 0.752),
           label="Injektion in die OD")                        # GT -> OD-Stufe
    _arrow(ax, (b_x, 0.925), (b_x, 0.752))                     # B: Daten -> OD-Stufe
    for y0, y1 in [(0.700, 0.670), (0.628, 0.598), (0.556, 0.526)]:
        _arrow(ax, (0.5, y0), (0.5, y1))
    _arrow(ax, (a_x, 0.408), (a_x, 0.365))                     # GLM -> Auswertung A
    _arrow(ax, (b_x, 0.408), (b_x, 0.365))                     # GLM -> Auswertung B
    _arrow(ax, (a_x, 0.235), (a_x, 0.177))
    _arrow(ax, (b_x, 0.235), (b_x, 0.177))

    ax.set_title("Verarbeitungspipeline: Simulation (Strang A) und Realdaten (Strang B)\n"
                 "mit gemeinsamer Vorverarbeitung und gemeinsamem GLM",
                 fontsize=11.5, pad=14)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        p = OUT / f"00_pipeline_flowchart.{suffix}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"-> {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
