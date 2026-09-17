"""Farb- und Reihenfolge-Konventionen der Driftfamilien fuer alle Abbildungen.

Eine Familie traegt in jeder Abbildung dieselbe Farbe; verwandte Familien teilen sich
einen Farbton in Stufen (poly blau, legendre gruen, dct orange/rot, bspline braun,
Filter violett).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Kanonische Reihenfolge auf Achsen (nach Familie gruppiert, Ordnung steigend).
FAMILY_ORDER = [
    "none",
    "poly:1", "poly:2", "poly:3", "poly:4", "poly:5",
    "dct:0.005", "dct:0.01", "dct:0.02",
    "legendre:1", "legendre:3", "legendre:5",
    "bspline:5", "bspline:8",
    "butter:0.01", "butterxy:0.01", "lowpass:0.5", "bandpass:0.01-0.5",
]

_FIXED = {
    "none": "0.6",
    "dct:0.005": "#f4a259", "dct:0.01": "#e76f51", "dct:0.02": "#bc4749",
    "bspline:5": "#8d6e63", "bspline:8": "#4e342e",
    "butter:0.01": "#8338ec", "butterxy:0.01": "#3a0ca3", "lowpass:0.5": "#b5179e",
    "bandpass:0.01-0.5": "#560bad",
}


def family_color(fam: str):
    """Feste Farbe einer Driftfamilie (Ramps fuer poly/legendre, sonst Festwerte)."""
    if fam.startswith("poly"):
        return plt.cm.Blues(0.4 + 0.12 * int(fam.split(":")[1]))
    if fam.startswith("legendre"):
        return plt.cm.Greens(0.4 + 0.15 * int(fam.split(":")[1]))
    if fam.startswith("bspline"):
        return _FIXED.get(fam, "#795548")
    return _FIXED.get(fam, "0.4")


def order(families) -> list[str]:
    """Vorhandene Familien in kanonischer Reihenfolge (unbekannte hinten anhaengen)."""
    known = [f for f in FAMILY_ORDER if f in set(families)]
    return known + [f for f in families if f not in FAMILY_ORDER]
