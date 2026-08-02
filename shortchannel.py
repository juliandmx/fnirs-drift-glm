"""Short-Channel-Regression: Long/Short-Split und Regressor-Varianten.

Hintergrund (Betreuungsgespraech): Ein Kanal mit kleinem Quell-Detektor-Abstand hat eine
flache "Banane" -- sein Licht erreicht den Kortex nicht und misst nur Kopfhaut und
Schaedel, also die SYSTEMISCHE Physiologie (Herzschlag, Atmung, Mayer-Wellen, Hautdurch-
blutung). Nimmt man ihn als Regressor in die Designmatrix auf, laesst sich dieser Anteil
aus den langen Kanaelen herausrechnen; ausgewertet wird dann nur ueber die langen.

Auf nn22 gibt es keine echten Short-Separation-Kanaele (<10 mm). Der kuerzeste Abstand
betraegt 15,55 mm. Die Betreuungsvorgabe ist daher, **1,8 cm als Schwelle zu testen** --
dieselbe Konstruktion verwendet das Cedalion-Workshop-Notebook 36 auf `fingertappingDOT`
mit 22,5 mm ("The montage has longer (3-3.5cm) and shorter (~1.7-2.2cm) distance
channels. Define a cut-off at 22.5 mm").

LIMITATION, die in die Arbeit gehoert: 15,5-18 mm sehen noch etwas Kortex. Der Regressor
entfernt daher potenziell auch echtes Hirnsignal, nicht nur Systemik. Er ist ein
Naeherungs-Surrogat fuer eine echte Short-Separation-Montage, kein Ersatz.

Aufruf:  conda run -n cedalion python shortchannel.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import cedalion
import cedalion.nirs
from cedalion import units

# Betreuungsvorgabe: 1,8 cm testen (kuerzester Abstand im Datensatz: 15,55 mm)
SHORT_THRESHOLD = 1.8 * units.cm


def distances_mm(ts: xr.DataArray, geo3d) -> np.ndarray:
    """Quell-Detektor-Abstaende je Kanal in mm (dequantifiziert)."""
    d = cedalion.nirs.channel_distances(ts, geo3d).pint.to("mm")
    return np.asarray(d.pint.dequantify().values, dtype=float)


def split(ts: xr.DataArray, geo3d, threshold=SHORT_THRESHOLD):
    """Zerlegt eine Zeitreihe in lange und kurze Kanaele.

    Duenner Wrapper um `cedalion.nirs.split_long_short_channels`, damit die Schwelle an
    genau einer Stelle steht. Rueckgabe-Reihenfolge wie in Cedalion: **(long, short)**.
    Der Vergleich ist einheitenbewusst, `threshold` muss eine pint-Laenge sein.
    """
    return cedalion.nirs.split_long_short_channels(ts, geo3d, distance_threshold=threshold)


def report(ts: xr.DataArray, geo3d, thresholds_cm=(1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2)) -> dict:
    """Kennzahlen zur Schwellenwahl: wie viele Kanaele gelten bei welcher Schwelle als kurz.

    Belegt, dass 1,8 cm eine sinnvolle Wahl ist und nicht auf einer Kante sitzt.
    """
    d = distances_mm(ts, geo3d)
    return {
        "n_channels": int(d.size),
        "min_mm": float(d.min()),
        "median_mm": float(np.median(d)),
        "max_mm": float(d.max()),
        "counts": {c: int((d < c * 10.0).sum()) for c in thresholds_cm},
        "distances_mm": d,
    }


VARIANTS = ("short_avg", "short_maxcorr", "short_closest")


def short_dm(variant: str, ts_long: xr.DataArray, ts_short: xr.DataArray, geo3d):
    """Short-Channel-Regressor als DesignMatrix. Alle drei Varianten sind nativ.

      * `short_avg`     -- Mittel ueber ALLE kurzen Kanaele
        (`average_short_channel_regressor`). Ein gemeinsamer Regressor fuer alle Kanaele;
        robust, aber ohne raeumliche Zuordnung. Entspricht der Notiz "oder einfach short
        avg nutzen".
      * `short_maxcorr` -- je langem Kanal der am staerksten mit ihm korrelierende kurze
        Kanal (`max_corr_short_channel_regressor`). Entspricht der Notiz "der am meisten
        zu ihm korreliert". Das Maximum wird ueber die Chromophore gebildet, damit HbO
        und HbR denselben kurzen Kanal zugewiesen bekommen.
      * `short_closest` -- je langem Kanal der raeumlich naechste kurze Kanal
        (`closest_short_channel_regressor`). Nicht in der Notiz, aber die dritte native
        Variante und die in den Cedalion-Notebooks 32/34/35 verwendete.

    `maxcorr` und `closest` liefern KANALWEISE Designmatrizen (`channel_wise`), d.h. jeder
    lange Kanal bekommt seinen eigenen Regressor namens "short"; `avg` liefert einen
    gemeinsamen `common`-Regressor. `glm.fit` verarbeitet beides, gruppiert bei den
    kanalweisen aber intern nach `comp_group`.
    """
    import cedalion.models.glm as glm

    dmx = glm.design_matrix
    if variant == "short_avg":
        return dmx.average_short_channel_regressor(ts_short)
    if variant == "short_maxcorr":
        return dmx.max_corr_short_channel_regressor(ts_long, ts_short)
    if variant == "short_closest":
        return dmx.closest_short_channel_regressor(ts_long, ts_short, geo3d)
    raise ValueError(f"Unbekannte Short-Channel-Variante: {variant!r} "
                     f"(erlaubt: {VARIANTS})")


if __name__ == "__main__":
    import cedalion.data

    import preprocess as prep

    P = prep.run(cedalion.data.get_nn22_resting_state())
    r = report(P.conc, P.geo3d)

    print(f"Kanaele nach Preprocessing : {r['n_channels']}")
    print(f"Abstaende [mm]             : min {r['min_mm']:.2f}  "
          f"median {r['median_mm']:.2f}  max {r['max_mm']:.2f}")
    print("\nWie viele Kanaele gelten als 'kurz'?")
    for cm, n in r["counts"].items():
        mark = "   <-- Vorgabe" if abs(cm - 1.8) < 1e-9 else ""
        print(f"  < {cm:.1f} cm : {n:4d} kurz / {r['n_channels'] - n:4d} lang{mark}")

    ts_long, ts_short = split(P.conc, P.geo3d)
    print(f"\nSplit bei {SHORT_THRESHOLD}: {ts_short.sizes['channel']} kurz, "
          f"{ts_long.sizes['channel']} lang")

    d = r["distances_mm"]
    short = np.sort(d[d < 18.0])
    print(f"Abstaende der kurzen Kanaele: {short.min():.2f} .. {short.max():.2f} mm "
          f"(Median {np.median(short):.2f})")
    print("\nHinweis: das sind keine echten Short-Separation-Kanaele (<10 mm) --")
    print("sie sehen noch etwas Kortex. Als Limitation dokumentieren.")
